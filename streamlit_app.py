"""
Indian Telecom Customer Churn Prediction & GenAI Retention
Streamlit frontend that talks to the FastAPI backend.
"""

import streamlit as st
import requests

# ------------------------------------------------------------------
# Page config
# ------------------------------------------------------------------
st.set_page_config(
    page_title="Indian Telecom Churn Prediction & GenAI",
    page_icon="📱",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ------------------------------------------------------------------
# Constants
# ------------------------------------------------------------------
FASTAPI_URL = st.sidebar.text_input(
    "FastAPI URL",
    value="http://127.0.0.1:8000/predict",
    help="Change if your backend is running on a different host/port",
)

ALL_STATES = [
    "Andhra Pradesh",
    "Arunachal Pradesh",
    "Assam",
    "Bihar",
    "Chhattisgarh",
    "Goa",
    "Gujarat",
    "Haryana",
    "Himachal Pradesh",
    "Jharkhand",
    "Karnataka",
    "Kerala",
    "Madhya Pradesh",
    "Maharashtra",
    "Manipur",
    "Meghalaya",
    "Mizoram",
    "Nagaland",
    "Odisha",
    "Punjab",
    "Rajasthan",
    "Sikkim",
    "Tamil Nadu",
    "Telangana",
    "Tripura",
    "Uttar Pradesh",
    "Uttarakhand",
    "West Bengal",
]

PARTNERS = [
    "Reliance Jio",
    "Airtel",
    "Vodafone",
    "BSNL",
]

# ------------------------------------------------------------------
# Header
# ------------------------------------------------------------------
st.title("🇮🇳 Indian Telecom Customer Churn Prediction")
st.markdown(
    """
    Predict whether a customer of **Jio / Airtel / Vodafone / BSNL** is likely to churn  
    and automatically generate a **personalized Gemini AI retention offer**.
    """
)

# ------------------------------------------------------------------
# Sidebar inputs
# ------------------------------------------------------------------
st.sidebar.header("📋 Customer Profile")

telecom_partner = st.sidebar.selectbox("Telecom Partner", PARTNERS, index=0)
gender = st.sidebar.selectbox("Gender", ["M", "F"], index=0)
age = st.sidebar.slider("Age", 18, 80, 32)
state = st.sidebar.selectbox("State", ALL_STATES, index=10)  # Karnataka default
num_dependents = st.sidebar.slider("Number of Dependents", 0, 5, 1)
estimated_salary = st.sidebar.number_input(
    "Estimated Annual Salary (₹)",
    min_value=20000,
    max_value=500000,
    value=75000,
    step=5000,
)
calls_made = st.sidebar.number_input("Monthly Calls Made", min_value=0, max_value=300, value=45)
sms_sent = st.sidebar.number_input("Monthly SMS Sent", min_value=0, max_value=150, value=20)
data_used = st.sidebar.number_input(
    "Monthly Data Used (MB)", min_value=0, max_value=20000, value=5000, step=100
)

# ------------------------------------------------------------------
# Predict button
# ------------------------------------------------------------------
predict_btn = st.sidebar.button("🔍 Predict Churn & Generate Offer", type="primary", use_container_width=True)

if predict_btn:
    payload = {
        "telecom_partner": telecom_partner,
        "gender": gender,
        "age": age,
        "state": state,
        "num_dependents": num_dependents,
        "estimated_salary": estimated_salary,
        "calls_made": calls_made,
        "sms_sent": sms_sent,
        "data_used": data_used,
    }

    with st.spinner("Analyzing customer behaviour & contacting Gemini AI..."):
        try:
            response = requests.post("http://api:8000/predict", json=payload)

            if response.status_code == 200:
                result = response.json()

                churn_pred = result.get("churn_prediction")
                churn_prob = result.get("churn_probability", 0)
                risk_level = result.get("risk_level", "Unknown")
                retention_email = result.get("retention_email", "")
                normalized = result.get("normalized_partner", telecom_partner)

                st.divider()

                # Metrics row
                m1, m2, m3 = st.columns(3)
                m1.metric("Churn Probability", f"{churn_prob}%")
                m2.metric("Risk Level", risk_level)
                m3.metric("Prediction", "Will Churn ⚠️" if churn_pred == 1 else "Will Stay ✅")

                st.divider()

                col1, col2 = st.columns([1, 1.4])

                with col1:
                    st.subheader("📊 Prediction Summary")
                    if risk_level == "High":
                        st.error("**HIGH CHURN RISK** – Immediate retention action recommended.")
                    elif risk_level == "Medium":
                        st.warning("**MEDIUM RISK** – Proactive engagement advised.")
                    else:
                        st.success("**LOW RISK** – Customer appears stable.")

                    st.markdown(
                        f"""
                        | Field | Value |
                        |-------|-------|
                        | Partner | {normalized} |
                        | Age | {age} |
                        | State | {state} |
                        | Salary | ₹{estimated_salary:,} |
                        | Calls / SMS / Data | {calls_made} / {sms_sent} / {data_used} MB |
                        """
                    )

                with col2:
                    st.subheader("🤖 Gemini AI Retention Strategy")
                    if risk_level in ("High", "Medium"):
                        st.info("Personalized retention email generated by Gemini:")
                        st.markdown(retention_email)
                    else:
                        st.info(retention_email)

            else:
                try:
                    detail = response.json().get("detail", response.text)
                except Exception:
                    detail = response.text
                st.error(f"API Error ({response.status_code}): {detail}")

        except requests.exceptions.ConnectionError:
            st.error(
                "❌ Could not connect to FastAPI backend.\n\n"
                "Make sure the server is running:\n"
                "```bash\nuvicorn app.main:app --reload --host 0.0.0.0 --port 8000\n```"
            )
        except requests.exceptions.Timeout:
            st.error("Request timed out. Gemini may be slow – try again.")
        except Exception as e:
            st.error(f"Unexpected error: {str(e)}")

# ------------------------------------------------------------------
# Footer
# ------------------------------------------------------------------
st.sidebar.markdown("---")
st.sidebar.caption("Model: XGBoost · GenAI: Gemini · Dataset: Indian Telecom Churn")
st.sidebar.caption("Run `python train_model.py` if you need to retrain.")