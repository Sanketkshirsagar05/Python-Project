import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load model and scaler
model = joblib.load("fraud_model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("💳 Credit Card Fraud Detection System")

st.write("Enter transaction details to predict whether the transaction is Fraud or Legitimate.")

# User Inputs
amt = st.number_input("Transaction Amount", min_value=0.0)

gender = st.selectbox("Gender", ["Male", "Female"])

category = st.number_input("Merchant Category (Encoded)", min_value=0)

merchant = st.number_input("Merchant ID (Encoded)", min_value=0)

city = st.number_input("City (Encoded)", min_value=0)

state = st.number_input("State (Encoded)", min_value=0)

job = st.number_input("Job (Encoded)", min_value=0)

city_pop = st.number_input("City Population", min_value=0)

unix_time = st.number_input("Unix Time", min_value=0)

hour = st.slider("Transaction Hour", 0, 23)

day = st.slider("Transaction Day", 1, 31)

month = st.slider("Transaction Month", 1, 12)

is_weekend = st.selectbox("Weekend Transaction", [0,1])

age = st.number_input("Customer Age", min_value=18)

distance_km = st.number_input("Distance Between Customer & Merchant (km)", min_value=0.0)

# Encode gender
if gender == "Male":
    gender = 1
else:
    gender = 0


if st.button("Predict Fraud"):

    input_data = pd.DataFrame([[
        merchant,
        category,
        amt,
        gender,
        city,
        state,
        city_pop,
        job,
        unix_time,
        hour,
        day,
        month,
        is_weekend,
        age,
        distance_km
    ]], columns=[
        "merchant",
        "category",
        "amt",
        "gender",
        "city",
        "state",
        "city_pop",
        "job",
        "unix_time",
        "hour",
        "day",
        "month",
        "is_weekend",
        "age",
        "distance_km"
    ])

    input_scaled = scaler.transform(input_data)

    prediction = model.predict(input_scaled)
    probability = model.predict_proba(input_scaled)[0][1]

    if prediction[0] == 1:
        st.error(f"⚠️ Fraudulent Transaction Detected (Probability: {probability:.2f})")
    else:
        st.success(f"✅ Legitimate Transaction (Fraud Probability: {probability:.2f})")
