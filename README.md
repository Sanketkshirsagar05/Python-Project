# 💳 Credit Card Fraud Detection System

A Machine Learning project that detects **fraudulent credit card transactions** using transaction data, customer information, and location-based features. The model predicts whether a transaction is **fraudulent or legitimate** and is deployed using a **Streamlit web application**.

---

## 📌 Project Overview
This project builds an end-to-end fraud detection pipeline:

- Data preprocessing and feature engineering  
- Handling imbalanced datasets using **SMOTE**  
- Training ML models (Logistic Regression, Random Forest, XGBoost)  
- Selecting the best model  
- Deploying with **Streamlit**

---


---

## ⚙️ Key Features
- Feature engineering (age, time, distance between customer and merchant)
- Imbalanced data handling with **SMOTE**
- Fraud prediction using **XGBoost**
- Interactive **Streamlit web app**

---

## ▶️ Run the Project

Install dependencies:
pip install -r requirements.txt

Run the app:
streamlit run app.py
