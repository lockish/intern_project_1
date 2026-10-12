"""
Bank Customer Churn Intelligence Dashboard
Run:  streamlit run app.py      (after running  python train_model.py  once)
"""
import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Churn Intelligence Dashboard", layout="wide")


@st.cache_resource
def load_artifacts():
    model = joblib.load("churn_model.pkl")
    cols = joblib.load("feature_columns.pkl")
    probs = joblib.load("all_probabilities.pkl")
    return model, cols, probs


try:
    model, feature_columns, all_probs = load_artifacts()
except FileNotFoundError:
    st.error("Model files not found. Run `python train_model.py` first, in this same folder.")
    st.stop()


def build_row(credit_score, age, tenure, balance, num_products,
              has_cr_card, is_active, salary, geography, gender):
    """Turn form values into a one-row table in the exact column order used for training."""
    row = {
        "CreditScore": credit_score,
        "Age": age,
        "Tenure": tenure,
        "Balance": balance,
        "NumOfProducts": num_products,
        "HasCrCard": 1 if has_cr_card == "Yes" else 0,
        "IsActiveMember": 1 if is_active == "Yes" else 0,
        "EstimatedSalary": salary,
        "Geography_Germany": 1 if geography == "Germany" else 0,
        "Geography_Spain": 1 if geography == "Spain" else 0,
        "Gender_Male": 1 if gender == "Male" else 0,
    }
    return pd.DataFrame([row])[feature_columns]


def risk_label(p, threshold):
    if p >= 0.5:
        return "High risk"
    if p >= threshold:
        return "Elevated risk"
    return "Low risk"


# ----------------------------- Sidebar: customer profile -----------------------------
st.sidebar.header("Customer profile")
credit_score = st.sidebar.slider("Credit score", 300, 850, 650)
age = st.sidebar.slider("Age", 18, 92, 38)
tenure = st.sidebar.slider("Tenure (years)", 0, 10, 5)
balance = st.sidebar.number_input("Balance", min_value=0.0, value=0.0, step=1000.0)
salary = st.sidebar.number_input("Estimated salary", min_value=0.0, value=100000.0, step=1000.0)
num_products = st.sidebar.selectbox("Number of products", [1, 2, 3, 4])
has_cr_card = st.sidebar.selectbox("Has credit card?", ["Yes", "No"])
is_active = st.sidebar.selectbox("Active member?", ["Yes", "No"])
geography = st.sidebar.selectbox("Country", ["France", "Germany", "Spain"])
gender = st.sidebar.selectbox("Gender", ["Female", "Male"])
threshold = st.sidebar.slider("Alert threshold", 0.10, 0.60, 0.30, 0.05,
                              help="Customers at or above this probability are flagged for retention action.")

base_row = build_row(credit_score, age, tenure, balance, num_products,
                     has_cr_card, is_active, salary, geography, gender)
base_prob = float(model.predict_proba(base_row)[0][1])

st.title("Bank Customer Churn Intelligence Dashboard")
tab1, tab2, tab3, tab4 = st.tabs(
    ["Risk calculator", "Probability distribution", "Feature importance", "What-if simulator"])

# ----------------------------- Tab 1: calculator -----------------------------
with tab1:
    st.subheader("Churn risk for this customer")
    c1, c2, c3 = st.columns(3)
    c1.metric("Churn probability", f"{base_prob:.1%}")
    c2.metric("Churn flag (threshold %.2f)" % threshold, "CHURN" if base_prob >= threshold else "RETAIN")
    c3.metric("Risk band", risk_label(base_prob, threshold))
    st.progress(min(base_prob, 1.0))
    if base_prob >= 0.5:
        st.error("High risk: recommend proactive, personalised retention outreach.")
    elif base_prob >= threshold:
        st.warning("Elevated risk: monitor and consider an engagement offer.")
    else:
        st.success("Low risk: no action needed.")
    st.caption("Change the profile in the left sidebar; results update immediately.")

# ----------------------------- Tab 2: distribution -----------------------------
with tab2:
    st.subheader("Predicted churn probability across all customers")
    st.caption("Out-of-fold predictions: each customer was scored by a model that never saw them.")
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(all_probs, bins=40, color="#2E5C8A", edgecolor="black")
    ax.axvline(threshold, color="orange", linestyle="--", label=f"Alert threshold = {threshold:.2f}")
    ax.axvline(base_prob, color="red", linestyle="-", label=f"Selected customer = {base_prob:.1%}")
    ax.set_xlabel("Predicted churn probability")
    ax.set_ylabel("Number of customers")
    ax.legend()
    st.pyplot(fig)
    plt.close(fig)
    a, b, c = st.columns(3)
    a.metric("Customers", f"{len(all_probs):,}")
    b.metric(f"Flagged at {threshold:.2f}", f"{int((all_probs >= threshold).sum()):,}")
    c.metric("High risk (>= 0.50)", f"{int((all_probs >= 0.5).sum()):,}")
    pct = float((all_probs <= base_prob).mean()) * 100
    st.write(f"The selected customer is riskier than **{pct:.0f}%** of the customer base.")

# ----------------------------- Tab 3: importance -----------------------------
with tab3:
    st.subheader("What drives the model's churn predictions")
    imp = pd.Series(model.feature_importances_, index=feature_columns).sort_values()
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.barh(imp.index, imp.values, color="#2E5C8A", edgecolor="black")
    ax.set_xlabel("Importance")
    st.pyplot(fig)
    plt.close(fig)
    st.write("Longer bar = the model relies on this feature more to separate churners from loyal customers.")
    for path, caption in [("figures/shap_summary.png", "SHAP summary: direction and size of each feature's effect"),
                          ("figures/partial_dependence.png", "Partial dependence: how risk changes with Age and NumOfProducts")]:
        if os.path.exists(path):
            st.image(path, caption=caption)

# ----------------------------- Tab 4: what-if -----------------------------
with tab4:
    st.subheader("What-if scenario simulator")
    st.write("Start from the profile in the sidebar, change engagement and product values, and compare.")
    s1, s2 = st.columns(2)
    sc_products = s1.selectbox("Scenario: number of products", [1, 2, 3, 4],
                               index=[1, 2, 3, 4].index(num_products), key="sc_p")
    sc_active = s1.selectbox("Scenario: active member?", ["Yes", "No"],
                             index=["Yes", "No"].index(is_active), key="sc_a")
    sc_balance = s2.number_input("Scenario: balance", min_value=0.0, value=float(balance), step=1000.0, key="sc_b")
    sc_tenure = s2.slider("Scenario: tenure (years)", 0, 10, tenure, key="sc_t")

    sc_row = build_row(credit_score, age, sc_tenure, sc_balance, sc_products,
                       has_cr_card, sc_active, salary, geography, gender)
    sc_prob = float(model.predict_proba(sc_row)[0][1])
    delta = sc_prob - base_prob

    m1, m2, m3 = st.columns(3)
    m1.metric("Current probability", f"{base_prob:.1%}")
    m2.metric("Scenario probability", f"{sc_prob:.1%}", f"{delta * 100:+.1f} pts", delta_color="inverse")
    m3.metric("Scenario risk band", risk_label(sc_prob, threshold))
    fig, ax = plt.subplots(figsize=(6, 2.5))
    ax.barh(["Scenario", "Current"], [sc_prob, base_prob], color=["#C0392B", "#2E5C8A"], edgecolor="black")
    ax.axvline(threshold, color="orange", linestyle="--")
    ax.set_xlim(0, 1)
    ax.set_xlabel("Churn probability")
    st.pyplot(fig)
    plt.close(fig)
