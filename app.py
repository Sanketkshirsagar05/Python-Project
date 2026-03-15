import streamlit as st
import pandas as pd
import joblib
import os


MODEL_PATH = "PKL Files"

# Load Model and Scaler
with open(os.path.join(MODEL_PATH, "fraud_model.pkl"), "rb") as f:
    model = joblib.load(f)

with open(os.path.join(MODEL_PATH, "scaler.pkl"), "rb") as f:
    scaler = joblib.load(f)

with open(os.path.join(MODEL_PATH, "merchant_encoder.pkl"), "rb") as f:
    merchant_encoder = joblib.load(f)

with open(os.path.join(MODEL_PATH, "category_encoder.pkl"), "rb") as f:
    category_encoder = joblib.load(f)

with open(os.path.join(MODEL_PATH, "city_encoder.pkl"), "rb") as f:
    city_encoder = joblib.load(f)

with open(os.path.join(MODEL_PATH, "state_encoder.pkl"), "rb") as f:
    state_encoder = joblib.load(f)

with open(os.path.join(MODEL_PATH, "job_encoder.pkl"), "rb") as f:
    job_encoder = joblib.load(f)

# UI
st.title("💳 Credit Card Fraud Detection System")

st.write("Enter transaction details manually to predict fraud.")

merchant = st.selectbox("Merchant", merchant_encoder.classes_)
category = st.selectbox("Category", category_encoder.classes_)
city = st.selectbox("City", city_encoder.classes_)
state = st.selectbox("State", state_encoder.classes_)
job = st.selectbox("Job", job_encoder.classes_)

gender = st.selectbox("Gender", ["Male","Female"])

amt = st.number_input("Transaction Amount", 0.0)

zip_code = st.number_input("Zip Code", 0)

city_pop = st.number_input("City Population", 0)

unix_time = st.number_input("Unix Time", 0)

hour = st.slider("Transaction Hour", 0, 23)

day = st.slider("Day of Month", 1, 31)

month = st.slider("Month", 1, 12)

is_weekend = st.selectbox("Is Weekend Transaction?", [0,1])

age = st.number_input("Customer Age", 18)

distance_km = st.number_input("Distance Between Customer & Merchant (km)", 0.0)


# Encode Categorical Values
merchant = merchant_encoder.transform([merchant])[0]
category = category_encoder.transform([category])[0]
city = city_encoder.transform([city])[0]
state = state_encoder.transform([state])[0]
job = job_encoder.transform([job])[0]

gender = 1 if gender == "Male" else 0


# Prediction
if st.button("Predict Fraud"):

    input_data = pd.DataFrame([[

        merchant,
        category,
        amt,
        gender,
        city,
        state,
        zip_code,
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
        "zip",
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

   
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]

    if prediction == 1:
        st.error(f"⚠️ Fraudulent Transaction Detected | Probability: {probability:.2f}")
    else:
        st.success(f"✅ Legitimate Transaction | Fraud Probability: {probability:.2f}")


