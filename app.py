import streamlit as st
import pandas as pd
import numpy as np
import joblib

# -----------------------------
# Load Trained Model
# -----------------------------
model = joblib.load("models/fraud_model.pkl")

st.set_page_config(page_title="Fraud Detection App", layout="centered")

st.title("💳 Fraud Detection System")
st.write("Predict whether a transaction is Fraudulent or Legitimate")

st.markdown("---")

# -----------------------------
# User Input Section
# -----------------------------
st.subheader("Enter Transaction Details")

# Example features (Modify based on your dataset)
amount = st.number_input("Transaction Amount", min_value=0.0, value=100.0)
oldbalanceOrg = st.number_input("Old Balance (Sender)", min_value=0.0, value=1000.0)
newbalanceOrig = st.number_input("New Balance (Sender)", min_value=0.0, value=900.0)
oldbalanceDest = st.number_input("Old Balance (Receiver)", min_value=0.0, value=0.0)
newbalanceDest = st.number_input("New Balance (Receiver)", min_value=0.0, value=100.0)

# -----------------------------
# Prediction Button
# -----------------------------
if st.button("Predict Fraud"):

    # Create DataFrame
    input_data = pd.DataFrame({
        "amount": [amount],
        "oldbalanceOrg": [oldbalanceOrg],
        "newbalanceOrig": [newbalanceOrig],
        "oldbalanceDest": [oldbalanceDest],
        "newbalanceDest": [newbalanceDest]
    })

    # Prediction
    prediction = model.predict(input_data)
    probability = model.predict_proba(input_data)[0][1]

    st.markdown("---")

    # Output
    if prediction[0] == 1:
        st.error("⚠️ Fraudulent Transaction Detected!")
    else:
        st.success("✅ Legitimate Transaction")

    st.write(f"Fraud Probability: **{probability:.2%}**")

    # Risk Category
    if probability > 0.75:
        st.error("🔴 High Risk")
    elif probability > 0.40:
        st.warning("🟠 Medium Risk")
    else:
        st.success("🟢 Low Risk")

