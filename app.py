import streamlit as st
import pickle
import pandas as pd
import matplotlib.pyplot as plt

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Churn Predictor", page_icon="📊", layout="centered")

# ---------------- LOAD MODEL ----------------
model = pickle.load(open('model.pkl', 'rb'))
columns = pickle.load(open('columns.pkl', 'rb'))

# ---------------- TITLE ----------------
st.title("📊 Customer Churn Prediction System")

st.markdown("""
This application predicts whether a customer is likely to churn or stay using a Machine Learning model.

👉 Enter the customer details from the sidebar and click **Predict**.
""")

# ---------------- SIDEBAR INPUT ----------------
st.sidebar.header("🧾 Customer Details")

tenure = st.sidebar.slider("Tenure (months)", 1, 72, 12)
monthly_charges = st.sidebar.number_input("Monthly Charges", min_value=0.0, value=50.0)
total_charges = st.sidebar.number_input("Total Charges", min_value=0.0, value=500.0)

# ---------------- CREATE INPUT DATA ----------------
input_dict = {col: 0 for col in columns}

# Basic features mapping (adjust if dataset different)
if 'tenure' in input_dict:
    input_dict['tenure'] = tenure
if 'MonthlyCharges' in input_dict:
    input_dict['MonthlyCharges'] = monthly_charges
if 'TotalCharges' in input_dict:
    input_dict['TotalCharges'] = total_charges

input_df = pd.DataFrame([input_dict])

# ---------------- PREDICTION ----------------
if st.button("🚀 Predict"):

    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    st.markdown("---")

    # Result Display
    if prediction == 1:
        st.error(f"❌ Customer is likely to CHURN\n\nProbability: {probability:.2f}")
    else:
        st.success(f"✅ Customer is likely to STAY\n\nProbability: {probability:.2f}")

    # ---------------- CHART ----------------
    st.subheader("📊 Prediction Probability")

    fig, ax = plt.subplots()
    ax.bar(["Stay", "Churn"], [1 - probability, probability])
    ax.set_ylabel("Probability")
    st.pyplot(fig)

# ---------------- ABOUT SECTION ----------------
st.markdown("---")
st.subheader("ℹ️ About Project")

st.write("""
This project uses a Random Forest machine learning model to predict customer churn.

It helps businesses identify customers who are likely to leave and take necessary actions to retain them.

Built as part of B.Tech Gen AI coursework.
""")
