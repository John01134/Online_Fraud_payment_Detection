import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ==============================
# Load model, scaler
# ==============================
MODEL_PATH = r"C:\Users\John\Downloads\Online_Fraud_payment_Detection_C_P-2\fraud_model.pkl"
SCALER_PATH = r"C:\Users\John\Downloads\Online_Fraud_payment_Detection_C_P-2\scaler.pkl"

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

# ==============================
# Manual Mapping for Transaction Types
# ==============================
transaction_mapping = {
    "Payment": 0,
    "Cash Withdrawal (Cash-Out)": 1,
    "Cash Deposit (Cash-In)": 2,
    "Account Transfer": 3,
    "Debit Transaction": 4
}

# ==============================
# Streamlit UI
# ==============================
st.set_page_config(page_title="Fraud Detection App", page_icon="💳", layout="centered")

st.title("💳 Smart Fraud Risk Detection System")
st.write("This app uses **Machine Learning + Business Rules** to detect potentially fraudulent transactions in real-time.")

# Transaction input fields
st.subheader("📥 Enter Transaction Details")

amount = st.number_input("💰 Transaction Amount", step=100.0, value=5000.0)
oldbalanceOrg = st.number_input("🏦 Sender's Balance Before Transaction", step=100.0, value=20000.0)
newbalanceOrig = st.number_input("🏦 Sender's Balance After Transaction", step=100.0, value=15000.0)
oldbalanceDest = st.number_input("📥 Receiver's Balance Before Transaction", step=100.0, value=10000.0)
newbalanceDest = st.number_input("📥 Receiver's Balance After Transaction", step=100.0, value=15000.0)

transaction_type = st.selectbox("📂 Transaction Type", list(transaction_mapping.keys()))

# ==============================
# Prediction + Rules
# ==============================
if st.button("🔍 Analyze Transaction"):
    type_encoded = transaction_mapping[transaction_type]

    # Build input
    input_data = np.array([[type_encoded, amount, oldbalanceOrg, newbalanceOrig,
                            oldbalanceDest, newbalanceDest]])
    input_data_scaled = scaler.transform(input_data)

    # Model prediction
    prediction = model.predict(input_data_scaled)[0]
    prob = model.predict_proba(input_data_scaled)[0][1] * 100

    # Rule-based checks
    reasons = []
    sender_diff = oldbalanceOrg - newbalanceOrig
    receiver_diff = newbalanceDest - oldbalanceDest

    if not np.isclose(sender_diff, amount, atol=1e-2):
        reasons.append("❌ Sender’s balance change does not match transaction amount.")
    if not np.isclose(receiver_diff, amount, atol=1e-2):
        reasons.append("❌ Receiver’s balance change does not match transaction amount.")
    if oldbalanceOrg < amount:
        reasons.append("❌ Sender had insufficient balance for this transaction.")
    if all(v == 0 for v in [amount, oldbalanceOrg, newbalanceOrig, oldbalanceDest, newbalanceDest]):
        reasons.append("❌ All values are zero (invalid transaction).")
    if amount > 0.8 * oldbalanceOrg and oldbalanceOrg > 0:
        reasons.append("⚠️ Transaction drains more than 80% of sender’s balance (unusual behavior).")

    # ==============================
    # Risk category (Low / Medium / High) - Text only
    # ==============================
    if any("❌" in r for r in reasons):   # 🚨 Critical errors = High Risk
        risk_level = "🔴 High Risk"
    elif prob < 30 and not reasons:
        risk_level = "🟢 Low Risk"
    elif prob < 70 or any("⚠️" in r for r in reasons):
        risk_level = "🟡 Medium Risk"
    else:
        risk_level = "🔴 High Risk"

    # ==============================
    # Display Results
    # ==============================
    st.subheader("📊 Transaction Analysis Result")

    if prediction == 1 or reasons:
        st.error(f"🚨 **Alert! Suspicious Transaction Detected**\n\nRisk Level: {risk_level}")
    else:
        st.success(f"✅ **Transaction Appears Safe**\n\nRisk Level: {risk_level}")

    # Show reasons if any
    if reasons:
        st.warning("### ⚠️ Additional Checks Triggered")
        for r in reasons:
            st.write(r)
    else:
        st.info("✅ All consistency checks passed.")

    # ==============================
    # Transaction Summary
    # ==============================
    st.subheader("📑 Transaction Summary")
    summary = pd.DataFrame({
        "Feature": ["Transaction Type", "Amount", "Sender Old Balance", "Sender New Balance",
                    "Receiver Old Balance", "Receiver New Balance"],
        "Value": [transaction_type, amount, oldbalanceOrg, newbalanceOrig, oldbalanceDest, newbalanceDest]
    })
    st.table(summary)
