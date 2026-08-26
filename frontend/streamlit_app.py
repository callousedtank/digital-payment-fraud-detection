import os

import requests
import streamlit as st

DEFAULT_API_URL = "https://digital-payment-fraud-detection.onrender.com/predict"
API_URL = os.getenv("API_URL", DEFAULT_API_URL)

st.set_page_config(page_title="Digital Payment Fraud Detection", page_icon="💳")
st.title("Transaction fraud review")
st.caption("Assess a payment using the deployed fraud-risk model. This is a decision-support signal, not an automatic block.")

with st.expander("How to use this tool", expanded=False):
    st.write("Enter the transaction details and review the fraud probability alongside the predicted risk decision. Escalate high-risk results for manual review.")

with st.form("transaction_form"):
    transaction_amount = st.number_input(
        "Transaction amount", min_value=0.0, value=5000.0, step=100.0
    )
    transaction_type = st.selectbox("Transaction type", ["Payment", "Transfer", "Withdrawal"])
    payment_mode = st.selectbox("Payment mode", ["Card", "NetBanking", "UPI", "Wallet"])
    device_type = st.selectbox("Device type", ["Android", "iOS", "Web"])
    device_location = st.text_input("Device location", value="Chennai")
    account_age_days = st.number_input("Account age (days)", min_value=0, value=365, step=1)
    transaction_hour = st.slider("Transaction hour", min_value=0, max_value=23, value=14)
    previous_failed_attempts = st.number_input("Previous failed attempts", min_value=0, value=0, step=1)
    avg_transaction_amount = st.number_input(
        "Average transaction amount", min_value=0.0, value=4500.0, step=100.0
    )
    is_international = st.selectbox(
        "International transaction?",
        options=[0, 1],
        format_func=lambda value: "Yes" if value else "No",
    )
    ip_risk_score = st.slider("IP risk score", min_value=0.0, max_value=1.0, value=0.2, step=0.01)
    login_attempts_last_24h = st.number_input(
        "Login attempts in last 24h", min_value=0, value=2, step=1
    )

    submitted = st.form_submit_button("Check transaction")

if submitted:
    payload = {
        "transaction_amount": transaction_amount,
        "transaction_type": transaction_type,
        "payment_mode": payment_mode,
        "device_type": device_type,
        "device_location": device_location,
        "account_age_days": account_age_days,
        "transaction_hour": transaction_hour,
        "previous_failed_attempts": previous_failed_attempts,
        "avg_transaction_amount": avg_transaction_amount,
        "is_international": is_international,
        "ip_risk_score": ip_risk_score,
        "login_attempts_last_24h": login_attempts_last_24h,
    }

    try:
        response = requests.post(API_URL, json=payload, timeout=30)
        response.raise_for_status()
    except requests.RequestException as exc:
        st.error(f"Prediction request failed: {exc}")
    else:
        result = response.json()
        prediction = result.get("fraud_prediction")
        probability = result.get("fraud_probability")
        threshold = result.get("decision_threshold")
        if prediction == 1:
            st.error("Potentially fraudulent — send for review")
        else:
            st.success("Likely legitimate transaction")

        if isinstance(probability, (int, float)):
            st.metric("Estimated fraud probability", f"{probability:.1%}")
            st.progress(min(max(float(probability), 0.0), 1.0))
        if isinstance(threshold, (int, float)):
            st.caption(f"Decision threshold: {threshold:.1%}. Scores at or above this value are marked for review.")

        with st.expander("Prediction details"):
            st.json(result)
