"""
train_model.py  -  Bank customer churn prediction: full pipeline.

Run once from the project folder (with your venv active):
    python train_model.py

Needs European_Bank.csv in the same folder. Creates:
    churn_model.pkl, scaler.pkl, feature_columns.pkl, all_probabilities.pkl   (used by app.py)
    results/metrics.csv, results/threshold_table.csv, results/results.json    (numbers for the paper)
    figures/*.png                                                             (charts for the paper)
"""
import json
import os
import warnings

import joblib
import matplotlib
matplotlib.use("Agg")  # draw to files, no window needed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import PartialDependenceDisplay
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import (StratifiedKFold, cross_val_predict,
                                     cross_val_score, train_test_split)
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")
RANDOM_STATE = 42
CSV_PATH = "European_Bank.csv"

os.makedirs("figures", exist_ok=True)
os.makedirs("results", exist_ok=True)

# ----------------------------------------------------------------------------
# 1. Load and inspect
# ----------------------------------------------------------------------------
raw = pd.read_csv(CSV_PATH)
print("Loaded:", raw.shape)
print("Missing values:", int(raw.isnull().sum().sum()), "| Duplicates:", int(raw.duplicated().sum()))

churn_rate = raw["Exited"].mean()
print(f"Overall churn rate: {churn_rate:.4f}")

# ----------------------------------------------------------------------------
# 2. EDA figures
# ----------------------------------------------------------------------------
def bar_rate(series_groups, title, fname, xlabel):
    rates = raw.groupby(series_groups)["Exited"].mean() * 100
    fig, ax = plt.subplots(figsize=(6, 4))
    rates.plot(kind="bar", ax=ax, color="#2E5C8A", edgecolor="black")
    ax.axhline(churn_rate * 100, color="red", linestyle="--", label="Overall churn rate")
    ax.set_ylabel("Churn rate (%)")
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    ax.legend()
    plt.xticks(rotation=0)
    plt.tight_layout()
    fig.savefig(f"figures/{fname}", dpi=150)
    plt.close(fig)
    return rates

age_bands = pd.cut(raw["Age"], bins=[17, 30, 40, 50, 60, 100],
                   labels=["18-30", "31-40", "41-50", "51-60", "60+"])
eda = {
    "geography": bar_rate("Geography", "Churn rate by country", "eda_geography.png", "Country"),
    "gender": bar_rate("Gender", "Churn rate by gender", "eda_gender.png", "Gender"),
    "active": bar_rate("IsActiveMember", "Churn rate by activity (1 = active)", "eda_active.png", "IsActiveMember"),
    "products": bar_rate("NumOfProducts", "Churn rate by number of products", "eda_products.png", "Number of products"),
    "age": bar_rate(age_bands, "Churn rate by age band", "eda_age.png", "Age band"),
}
fig, ax = plt.subplots(figsize=(4, 4))
raw["Exited"].value_counts().sort_index().plot(kind="bar", ax=ax, color=["#2E5C8A", "#C0392B"], edgecolor="black")
ax.set_xticklabels(["Retained (0)", "Churned (1)"], rotation=0)
ax.set_ylabel("Customers")
ax.set_title("Class balance")
plt.tight_layout()
fig.savefig("figures/eda_class_balance.png", dpi=150)
plt.close(fig)

# ----------------------------------------------------------------------------
# 3. Preprocessing and feature set
# ----------------------------------------------------------------------------
df = raw.drop(columns=["Year", "CustomerId", "Surname"], errors="ignore")
df = pd.get_dummies(df, columns=["Geography", "Gender"], drop_first=True)
for c in df.columns:  # make sure dummies are 0/1 numbers
    if df[c].dtype == bool:
        df[c] = df[c].astype(int)

X = df.drop(columns=["Exited"])
y = df["Exited"]
feature_columns = list(X.columns)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)
print("Train:", X_train.shape, "Test:", X_test.shape,
      "| churn train/test:", round(y_train.mean(), 4), round(y_test.mean(), 4))

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)   # fit on train only (no leakage)
X_test_s = scaler.transform(X_test)

# ----------------------------------------------------------------------------
# 4. Models
# ----------------------------------------------------------------------------
models = {
    "Logistic Regression": (LogisticRegression(random_state=RANDOM_STATE, max_iter=1000), True),
    "Decision Tree": (DecisionTreeClassifier(random_state=RANDOM_STATE), False),
    "Random Forest": (RandomForestClassifier(random_state=RANDOM_STATE), False),
    "Gradient Boosting": (GradientBoostingClassifier(random_state=RANDOM_STATE), False),
}

rows, probas, fitted = [], {}, {}
for name, (model, needs_scaling) in models.items():
    Xtr, Xte = (X_train_s, X_test_s) if needs_scaling else (X_train, X_test)
    model.fit(Xtr, y_train)
    pred = model.predict(Xte)
    proba = model.predict_proba(Xte)[:, 1]
    probas[name], fitted[name] = proba, model
    rows.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, pred),
        "Precision": precision_score(y_test, pred),
        "Recall": recall_score(y_test, pred),
        "F1": f1_score(y_test, pred),
        "ROC-AUC": roc_auc_score(y_test, proba),
    })
metrics = pd.DataFrame(rows).round(4)
metrics.to_csv("results/metrics.csv", index=False)
print("\n", metrics.to_string(index=False))

# ROC curves
fig, ax = plt.subplots(figsize=(6, 5))
for name, proba in probas.items():
    fpr, tpr, _ = roc_curve(y_test, proba)
    ax.plot(fpr, tpr, label=f"{name} (AUC {roc_auc_score(y_test, proba):.3f})")
ax.plot([0, 1], [0, 1], "k--", label="Random guess")
ax.set_xlabel("False positive rate")
ax.set_ylabel("True positive rate")
ax.set_title("ROC curves on the held-out test set")
ax.legend(loc="lower right")
plt.tight_layout()
fig.savefig("figures/roc_curves.png", dpi=150)
plt.close(fig)

# ----------------------------------------------------------------------------
# 5. Best model: cross-validation, threshold study
# ----------------------------------------------------------------------------
gb = fitted["Gradient Boosting"]
gb_proba = probas["Gradient Boosting"]

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
cv_auc = cross_val_score(GradientBoostingClassifier(random_state=RANDOM_STATE),
                         X, y, cv=cv, scoring="roc_auc")
print(f"\n5-fold CV ROC-AUC (Gradient Boosting): {cv_auc.mean():.4f} +/- {cv_auc.std():.4f}")

thr_rows = []
for t in [0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]:
    pred = (gb_proba >= t).astype(int)
    thr_rows.append({
        "Threshold": t,
        "Precision": precision_score(y_test, pred),
        "Recall": recall_score(y_test, pred),
        "F1": f1_score(y_test, pred),
        "Flagged customers": int(pred.sum()),
    })
thr = pd.DataFrame(thr_rows).round(4)
thr.to_csv("results/threshold_table.csv", index=False)
print("\n", thr.to_string(index=False))

fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(thr["Threshold"], thr["Precision"], marker="o", label="Precision")
ax.plot(thr["Threshold"], thr["Recall"], marker="o", label="Recall")
ax.plot(thr["Threshold"], thr["F1"], marker="o", label="F1")
ax.set_xlabel("Decision threshold")
ax.set_title("Precision / recall trade-off (Gradient Boosting)")
ax.legend()
plt.tight_layout()
fig.savefig("figures/threshold_tradeoff.png", dpi=150)
plt.close(fig)

# ----------------------------------------------------------------------------
# 6. Explainability: importance, SHAP, partial dependence
# ----------------------------------------------------------------------------
imp = pd.Series(gb.feature_importances_, index=feature_columns).sort_values()
fig, ax = plt.subplots(figsize=(6, 5))
imp.plot(kind="barh", ax=ax, color="#2E5C8A", edgecolor="black")
ax.set_xlabel("Importance")
ax.set_title("Gradient Boosting feature importance")
plt.tight_layout()
fig.savefig("figures/feature_importance.png", dpi=150)
plt.close(fig)

try:
    import shap
    explainer = shap.TreeExplainer(gb)
    shap_values = explainer.shap_values(X_test)
    plt.figure()
    shap.summary_plot(shap_values, X_test, show=False)
    plt.tight_layout()
    plt.savefig("figures/shap_summary.png", dpi=150, bbox_inches="tight")
    plt.close()
    mean_abs = pd.Series(np.abs(shap_values).mean(axis=0), index=feature_columns).sort_values(ascending=False)
    mean_abs.round(4).to_csv("results/shap_mean_abs.csv", header=["mean_abs_shap"])
    print("\nSHAP done. Top drivers:\n", mean_abs.head(5).round(3).to_string())
except Exception as e:  # SHAP is optional; do not stop the pipeline
    print("SHAP skipped:", e)

fig, ax = plt.subplots(1, 2, figsize=(10, 4))
PartialDependenceDisplay.from_estimator(gb, X_train.astype(float), ["Age", "NumOfProducts"], ax=ax)
plt.tight_layout()
fig.savefig("figures/partial_dependence.png", dpi=150)
plt.close(fig)

# ----------------------------------------------------------------------------
# 7. Save artefacts for the Streamlit app
# ----------------------------------------------------------------------------
# Out-of-fold probabilities: every customer is scored by a model that never saw them.
oof = cross_val_predict(GradientBoostingClassifier(random_state=RANDOM_STATE),
                        X, y, cv=cv, method="predict_proba")[:, 1]
joblib.dump(gb, "churn_model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(feature_columns, "feature_columns.pkl")
joblib.dump(oof, "all_probabilities.pkl")

summary = {
    "n_rows": int(len(raw)),
    "churn_rate": round(float(churn_rate), 4),
    "cv_auc_mean": round(float(cv_auc.mean()), 4),
    "cv_auc_std": round(float(cv_auc.std()), 4),
    "high_risk_at_0.3": int((oof >= 0.3).sum()),
    "high_risk_at_0.5": int((oof >= 0.5).sum()),
}
with open("results/results.json", "w") as f:
    json.dump(summary, f, indent=2)
print("\nSaved model files, results/ and figures/. Summary:", summary)
