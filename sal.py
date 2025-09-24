# streamlit_salary.py
import streamlit as st
import numpy as np
import joblib

# Load model, scaler, and columns
@st.cache_resource
def load_artifacts():
    model = joblib.load("Salary_model.pkl")
    scaler = joblib.load("scaler_salary.pkl")
    columns = joblib.load("columns_salary.pkl")
    return model, scaler, columns

model, scaler, columns = load_artifacts()

st.title("📈 Salary Prediction App by Satyam")

st.write("Enter your **Years of Experience** to predict Salary:")

# Input
years_exp = st.number_input("Years of Experience", min_value=0.0, max_value=40.0, step=0.1)

if st.button("Predict Salary"):
    # Convert input into DataFrame-like array
    input_data = np.array([[years_exp]])
    
    # Apply same scaler
    input_scaled = scaler.transform(input_data)
    
    # Predict
    prediction = model.predict(input_scaled)[0]
    
    st.success(f"💰 Predicted Salary: ₹ {prediction:,.2f}")
