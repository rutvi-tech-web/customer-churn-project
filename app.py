import streamlit as st
import pickle
import pandas as pd

model = pickle.load(open('model.pkl', 'rb'))
columns = pickle.load(open('columns.pkl', 'rb'))

st.title("Customer Churn Prediction")

tenure = st.slider("Tenure", 1, 72)
monthly_charges = st.number_input("Monthly Charges")
total_charges = st.number_input("Total Charges")

input_dict = {col:0 for col in columns}

if 'tenure' in input_dict:
    input_dict['tenure'] = tenure
if 'MonthlyCharges' in input_dict:
    input_dict['MonthlyCharges'] = monthly_charges
if 'TotalCharges' in input_dict:
    input_dict['TotalCharges'] = total_charges

input_df = pd.DataFrame([input_dict])

if st.button("Predict"):
    prediction = model.predict(input_df)[0]

    if prediction == 1:
        st.error("Customer will churn")
    else:
        st.success("Customer will stay")