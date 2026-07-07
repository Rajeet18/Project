import streamlit as st
import numpy as np
import joblib

# Load trained model
model = joblib.load(r"C:\Users\RAJEET\OneDrive\Desktop\A,A,P   tops\ML project\loan_model.pkl")

st.title("Loan Approval Prediction")

# User inputs
dependents = st.number_input("Dependents", 0, 10)
income = st.number_input("Income")
loan_amount = st.number_input("Loan Amount")
loan_term = st.number_input("Loan Term")
cibil = st.number_input("CIBIL Score", 300, 900)

residential = st.number_input("Residential Assets")
commercial = st.number_input("Commercial Assets")
luxury = st.number_input("Luxury Assets")
bank = st.number_input("Bank Assets")

# Predict
if st.button("Predict"):

    data = np.array([[
        dependents,
        income,
        loan_amount,
        loan_term,
        cibil,
        residential,
        commercial,
        luxury,
        bank
    ]])

    prediction = model.predict(data)

    if prediction[0] == 1:
        st.success("Loan Approved")
    else:
        st.error("Loan Rejected")