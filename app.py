import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime
from math import radians, sin, cos, sqrt, atan2

# ------------------------------------
# Paths
# ------------------------------------

data_path = r"C:\Users\sanke\OneDrive\Documents\Fraud Detection System\Dataset\fraudTest.csv"

pkl_path = r"C:\Users\sanke\OneDrive\Documents\Fraud Detection System\PKL Files"

# ------------------------------------
# Load Dataset
# ------------------------------------

df = pd.read_csv(data_path)

# ------------------------------------
# Load Model and Encoders
# ------------------------------------

model = joblib.load(os.path.join(pkl_path,"fraud_model.pkl"))
scaler = joblib.load(os.path.join(pkl_path,"scaler.pkl"))

merchant_encoder = joblib.load(os.path.join(pkl_path,"merchant_encoder.pkl"))
category_encoder = joblib.load(os.path.join(pkl_path,"category_encoder.pkl"))
city_encoder = joblib.load(os.path.join(pkl_path,"city_encoder.pkl"))
state_encoder = joblib.load(os.path.join(pkl_path,"state_encoder.pkl"))
job_encoder = joblib.load(os.path.join(pkl_path,"job_encoder.pkl"))

# ------------------------------------
# Distance Function
# ------------------------------------

def haversine(lat1, lon1, lat2, lon2):

    R = 6371

    dlat = radians(lat2-lat1)
    dlon = radians(lon2-lon1)

    a = sin(dlat/2)**2 + cos(radians(lat1))*cos(radians(lat2))*sin(dlon/2)**2

    c = 2*atan2(sqrt(a),sqrt(1-a))

    return R*c

# ------------------------------------
# Streamlit UI
# ------------------------------------

st.title("💳 Credit Card Fraud Detection System")

merchant = st.selectbox("Merchant", df["merchant"].unique())

category = st.selectbox("Category", df["category"].unique())

city = st.selectbox("City", df["city"].unique())

state = st.selectbox("State", df["state"].unique())

job = st.selectbox("Job", df["job"].unique())

gender = st.selectbox("Gender", ["M","F"])

amt = st.number_input("Transaction Amount",0.0)

dob = st.date_input("Customer DOB")

# ------------------------------------
# Get city information
# ------------------------------------

city_data = df[df["city"]==city].iloc[0]

zip_code = city_data["zip"]

city_pop = city_data["city_pop"]

lat = city_data["lat"]
long = city_data["long"]

merch_lat = city_data["merch_lat"]
merch_long = city_data["merch_long"]

# ------------------------------------
# Auto Generated Features
# ------------------------------------

now = datetime.now()

unix_time = int(now.timestamp())

hour = now.hour
day = now.day
month = now.month

is_weekend = 1 if now.weekday()>=5 else 0

age = now.year - dob.year

distance_km = haversine(lat,long,merch_lat,merch_long)

# ------------------------------------
# Show Auto Generated Values
# ------------------------------------

st.write("Zip Code:",zip_code)
st.write("City Population:",city_pop)
st.write("Unix Time:",unix_time)
st.write("Distance (km):",round(distance_km,2))

# ------------------------------------
# Encode Categorical Values
# ------------------------------------

merchant = merchant_encoder.transform([merchant])[0]
category = category_encoder.transform([category])[0]
city = city_encoder.transform([city])[0]
state = state_encoder.transform([state])[0]
job = job_encoder.transform([job])[0]

gender = 1 if gender=="M" else 0

# ------------------------------------
# Prediction
# ------------------------------------

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

    ]],columns=[

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

    prob = model.predict_proba(input_scaled)[0][1]

    if prediction==1:

        st.error(f"⚠️ Fraud Detected | Probability {prob:.2f}")

    else:

        st.success(f"✅ Legitimate Transaction | Fraud Probability {prob:.2f}")
