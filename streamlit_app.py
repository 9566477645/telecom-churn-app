import os
import requests
import streamlit as st

# Page Configuration
st.set_page_config(
    page_title="Indian Telecom Customer Churn Prediction", page_icon="📡", layout="wide"
)

st.title("IN Indian Telecom Customer Churn Prediction")
st.markdown(
    "Predict whether a customer of **Jio / Airtel / Vodafone / BSNL** is likely to churn "
    "and automatically generate a personalized Gemini AI retention offer."
)

# ------------------------------------------------------------------
# Sidebar Configuration
# ------------------------------------------------------------------
st.sidebar.header("Configuration")
FASTAPI_URL = st.sidebar.text_input(
    "FastAPI URL",
    value="https://telecom-churn-app-1-wkem.onrender.com/predict",
    help="Change if your backend is running on a different host/port",
)

st.sidebar.header("Customer Input Features")

# Example input fields (ungaloda model features-ku etha mathiri change pannikonga)
gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
senior_citizen = st.sidebar.selectbox("Senior Citizen", [0, 1])
partner = st.sidebar.selectbox("Partner", ["Yes", "No"])
dependents = st.sidebar.selectbox("Dependents", ["Yes", "No"])
tenure = st.sidebar.slider("Tenure (Months)", 0, 72, 12)
monthly_charges = st.sidebar.number_input(
    "Monthly Charges", 0.0, 200.0, 50.0
)
total_charges = st.sidebar.number_input("Total Charges", 0.0, 5000.0, 500.0)

# ------------------------------------------------------------------
# Build the Request Data Dictionary
# ------------------------------------------------------------------
data = {
    "gender": gender,
    "SeniorCitizen": senior_citizen,
    "Partner": partner,
    "Dependents": dependents,
    "tenure": tenure,
    "MonthlyCharges": monthly_charges,
    "TotalCharges": total_charges,
}

# ------------------------------------------------------------------
# Prediction Trigger
# ------------------------------------------------------------------
if st.button("Predict Churn & Generate Offer", type="primary"):
  try:
    with st.spinner("Connecting to FastAPI backend..."):
      response = requests.post(FASTAPI_URL, json=data, timeout=30)

    if response.status_code == 200:
      result = response.json()
      st.success("Prediction completed successfully!")

      # Display Results (Customize based on your backend output keys)
      st.subheader("Results")
      st.json(result)
    else:
      st.error(
          f"Backend Error (Status {response.status_code}): {response.text}"
      )

  except requests.exceptions.ConnectionError:
    st.error(
        "Failed to connect to the FastAPI backend. Please check if the Render"
        " URL is active."
    )
  except Exception as e:
    st.error(f"An unexpected error occurred: {e}")