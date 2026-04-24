import streamlit as st
import pickle
import pandas as pd
import matplotlib.pyplot as plt

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Customer Churn Predictor", page_icon="📊", layout="wide")

# ---------------- LOAD MODEL ----------------
model = pickle.load(open('model.pkl', 'rb'))
columns = pickle.load(open('columns.pkl', 'rb'))

# ---------------- HEADER ----------------
st.title("📊 Customer Churn Prediction Dashboard")
st.markdown("Predict whether a customer will churn using Machine Learning")

st.markdown("---")

# ---------------- SIDEBAR ----------------
st.sidebar.header("🧾 Customer Details")

tenure = st.sidebar.slider("Tenure (Months)", 1, 72, 12)
monthly_charges = st.sidebar.number_input("Monthly Charges", 0.0, 10000.0, 50.0)
total_charges = st.sidebar.number_input("Total Charges", 0.0, 100000.0, 500.0)

gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
contract = st.sidebar.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])

# ---------------- INPUT DATA ----------------
input_dict = {col: 0 for col in columns}

# Basic mapping
if 'tenure' in input_dict:
    input_dict['tenure'] = tenure
if 'MonthlyCharges' in input_dict:
    input_dict['MonthlyCharges'] = monthly_charges
if 'TotalCharges' in input_dict:
    input_dict['TotalCharges'] = total_charges

# Example categorical encoding (adjust if needed)
if 'gender_Male' in input_dict:
    input_dict['gender_Male'] = 1 if gender == "Male" else 0

if 'Contract_One year' in input_dict:
    input_dict['Contract_One year'] = 1 if contract == "One year" else 0
if 'Contract_Two year' in input_dict:
    input_dict['Contract_Two year'] = 1 if contract == "Two year" else 0

input_df = pd.DataFrame([input_dict])

# ---------------- MAIN LAYOUT ----------------
col1, col2 = st.columns(2)

with col1:
    st.subheader("📥 Input Summary")
    st.write(input_df)

with col2:
    st.subheader("📊 Prediction Output")

# ---------------- PREDICTION ----------------
if st.button("🚀 Predict Churn"):

    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    with col2:

        # Result
        if prediction == 1:
            st.error("❌ High Risk: Customer may churn")
            st.warning("Possible reasons: Low tenure or high charges.")
        else:
            st.success("✅ Low Risk: Customer likely to stay")
            st.info("Customer is stable based on current usage.")

        # KPI Metrics
        st.metric("Churn Probability", f"{probability:.2f}")
        st.metric("Tenure", tenure)
        st.metric("Monthly Charges", monthly_charges)

        # Chart
        st.subheader("📈 Probability Chart")
        fig, ax = plt.subplots()
        ax.bar(["Stay", "Churn"], [1 - probability, probability])
        ax.set_ylabel("Probability")
        st.pyplot(fig)

        # Feature Importance
        st.subheader("🔥 Top Factors Affecting Churn")
        importance = pd.Series(model.feature_importances_, index=columns)
        top_features = importance.nlargest(5)
        st.bar_chart(top_features)

# ---------------- FOOTER ----------------
st.markdown("---")
st.caption("Made by Your Name | B.Tech Gen AI Project")
