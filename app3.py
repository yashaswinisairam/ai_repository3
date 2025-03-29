import streamlit as st
import numpy as np
import pickle

st.set_page_config(page_title="Credit Risk Estimator", layout="centered")

@st.cache_resource
def load_model():
    with open('model.pkl', 'rb') as f:
        return pickle.load(f)

model = load_model()

st.title("💳 AI Credit Risk Estimator")
st.markdown("Estimate creditworthiness using income, expenses, UPI activity, and rent.")

# Input fields (no recharge)
income = st.number_input("Monthly Income (₹)", min_value=0, step=100)
expense = st.number_input("Monthly Expense (₹)", min_value=0, step=100)
upi_txns = st.number_input("UPI Transactions/Month", min_value=0, step=1)
rent = st.number_input("Monthly Rent Paid (₹)", min_value=0, step=100)

if st.button("Predict Credit Risk"):
    input_data = np.array([[income, expense, upi_txns, rent]])
    prediction = model.predict(input_data)[0]

    if prediction == 1:
        st.success("✅ Likely Creditworthy")
    else:
        st.error("⚠️ High Credit Risk")
