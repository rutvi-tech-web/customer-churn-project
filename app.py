import streamlit as st
import pickle
import pandas as pd
import matplotlib.pyplot as plt

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="📊",
    layout="wide"
)

# ---------------- LOAD MODEL ----------------
model = pickle.load(open('model.pkl', 'rb'))
columns = pickle.load(open('columns.pkl', 'rb'))

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>
.main {
    background-color: #0E1117;
}
h1, h2, h3 {
    color: white;
}
.stButton>button {
    background-color: #4CAF50;
    color: white;
    border-radius: 10px;
    height: 3em;
    width: 100%;
    font-size: 18px;
}
</style>
""", unsafe_allow_html=True)

# ---------------- HEADER ----------------
st.title("📊 Customer Churn Prediction Dashboard")
st.markdown("Predict whether a customer will churn using Machine Learning")

st.markdown("---")

# ---------------- LAYOUT ----------------
col1, col2 = st.columns(2)

with col1:
    st.subheader("📥 Customer Input")

    tenure = st.slider("Tenure (Months)", 1, 72, 12)
    monthly_charges = st.number_input("Monthly Charges", 0.0, 10000.0, 50.0)
    total_charges = st.number_input("Total Charges", 0.0, 100000.0, 500.0)

    predict_btn = st.button("🚀 Predict Churn")

with col2:
    st.subheader("📊 Prediction Output")

# ---------------- INPUT PROCESS ----------------
input_dict = {col: 0 for col in columns}

if 'tenure' in input_dict:
    input_dict['tenure'] = tenure
if 'MonthlyCharges' in input_dict:
    input_dict['MonthlyCharges'] = monthly_charges
if 'TotalCharges' in input_dict:
    input_dict['TotalCharges'] = total_charges

input_df = pd.DataFrame([input_dict])

# ---------------- PREDICTION ----------------
if predict_btn:
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    with col2:

        if prediction == 1:
            st.error("❌ High Risk: Customer may churn")
        else:
            st.success("✅ Low Risk: Customer likely to stay")

        st.metric("Churn Probability", f"{probability:.2f}")

        # Chart
        fig, ax = plt.subplots()
        ax.bar(["Stay", "Churn"], [1 - probability, probability])
        ax.set_ylabel("Probability")
        st.pyplot(fig)

# ---------------- FOOTER ----------------
st.markdown("---")

st.markdown("""
### ℹ️ About Project
This application uses a Random Forest Machine Learning model to predict customer churn.

It helps businesses identify at-risk customers and improve retention strategies.
""")
