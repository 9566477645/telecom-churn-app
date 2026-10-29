"""
Indian Telecom Churn Prediction & GenAI Retention API
FastAPI backend that serves the XGBoost model and generates
personalized retention emails via Gemini when churn risk is high.
"""

import os
from pathlib import Path

import joblib
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator
from google import genai

# ------------------------------------------------------------------
# Paths (robust relative to this file)
# ------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "saved_models" / "xgboost_churn_v1.pkl"
SCALER_PATH = BASE_DIR / "saved_models" / "scaler.pkl"
FEATURES_PATH = BASE_DIR / "saved_models" / "feature_columns.pkl"

# ------------------------------------------------------------------
# App
# ------------------------------------------------------------------
app = FastAPI(
    title="Indian Telecom Churn Prediction & GenAI Retention API",
    description="Predict churn for Jio / Airtel / Vodafone / BSNL customers and generate AI retention offers.",
    version="2.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------------------------------------------------------
# Load artifacts
# ------------------------------------------------------------------
model = None
scaler = None
feature_columns = None

try:
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    feature_columns = joblib.load(FEATURES_PATH)
    print("✅ Model, scaler, and feature columns loaded successfully!")
except Exception as e:
    print(f"⚠️  Could not load model artifacts. Run train_model.py first! Error: {e}")

# Gemini client (requires GEMINI_API_KEY env var)
client = None
try:
    client = genai.Client()
except Exception as e:
    print(f"⚠️  Gemini client init failed (API key missing?): {e}")

# ------------------------------------------------------------------
# Partner name normalization (UI ↔ training data)
# ------------------------------------------------------------------
PARTNER_MAP = {
    "reliance jio": "Reliance Jio",
    "jio": "Reliance Jio",
    "bharti airtel": "Airtel",
    "airtel": "Airtel",
    "vodafone idea (vi)": "Vodafone",
    "vodafone idea": "Vodafone",
    "vi": "Vodafone",
    "vodafone": "Vodafone",
    "bsnl": "BSNL",
}

# ------------------------------------------------------------------
# Request schema
# ------------------------------------------------------------------
class CustomerData(BaseModel):
    telecom_partner: str = Field(..., example="Reliance Jio")
    gender: str = Field(..., example="M")
    age: int = Field(..., ge=18, le=100, example=35)
    state: str = Field(..., example="Karnataka")
    num_dependents: int = Field(..., ge=0, le=10, example=2)
    estimated_salary: int = Field(..., ge=10000, le=1000000, example=75000)
    calls_made: int = Field(..., ge=0, le=1000, example=45)
    sms_sent: int = Field(..., ge=0, le=500, example=20)
    data_used: int = Field(..., ge=0, le=100000, example=5000)

    @field_validator("telecom_partner")
    @classmethod
    def normalize_partner(cls, v: str) -> str:
        key = v.strip().lower()
        return PARTNER_MAP.get(key, v.strip())

    @field_validator("gender")
    @classmethod
    def normalize_gender(cls, v: str) -> str:
        v = v.strip().upper()
        if v not in {"M", "F"}:
            raise ValueError("gender must be 'M' or 'F'")
        return v


# ------------------------------------------------------------------
# Endpoints
# ------------------------------------------------------------------
@app.get("/")
def home():
    return {
        "message": "Indian Telecom Churn Prediction API is running successfully!",
        "version": "2.1.0",
        "endpoints": {"predict": "/predict", "health": "/health"},
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "scaler_loaded": scaler is not None,
        "features_loaded": feature_columns is not None,
        "gemini_ready": client is not None,
    }


@app.post("/predict")
def predict_churn(data: CustomerData):
    if model is None or scaler is None or feature_columns is None:
        raise HTTPException(
            status_code=500,
            detail="Model artifacts not loaded. Please run train_model.py first.",
        )

    try:
        # Build input DataFrame
        input_dict = data.model_dump()
        df_input = pd.DataFrame([input_dict])

        # One-hot encode (same as training)
        df_input = pd.get_dummies(df_input)

        # Align columns exactly to training feature set
        for col in feature_columns:
            if col not in df_input.columns:
                df_input[col] = 0
        df_input = df_input[feature_columns]

        # Scale numerical columns
        numerical_cols = [
            "age",
            "num_dependents",
            "estimated_salary",
            "calls_made",
            "sms_sent",
            "data_used",
        ]
        numerical_cols = [c for c in numerical_cols if c in df_input.columns]
        df_input[numerical_cols] = scaler.transform(df_input[numerical_cols])

        # Predict
        prediction = int(model.predict(df_input)[0])
        probability = float(model.predict_proba(df_input)[0][1])

        if probability >= 0.65:
            risk_level = "High"
        elif probability >= 0.40:
            risk_level = "Medium"
        else:
            risk_level = "Low"

        response = {
            "churn_prediction": prediction,
            "churn_probability": round(probability * 100, 2),
            "risk_level": risk_level,
            "normalized_partner": data.telecom_partner,
        }

        # GenAI retention offer for High / Medium risk
        if risk_level in ("High", "Medium") and client is not None:
            prompt = (
                f"You are a senior customer retention specialist at an Indian telecom company. "
                f"A customer using {data.telecom_partner} is at {risk_level.upper()} risk of churning "
                f"(probability ≈ {response['churn_probability']}%).\n\n"
                f"Customer profile:\n"
                f"- Age: {data.age}\n"
                f"- State: {data.state}\n"
                f"- Estimated annual salary: ₹{data.estimated_salary:,}\n"
                f"- Monthly calls: {data.calls_made}\n"
                f"- Monthly SMS: {data.sms_sent}\n"
                f"- Monthly data used: {data.data_used} MB\n"
                f"- Dependents: {data.num_dependents}\n\n"
                f"Write a short, warm, personalized retention email in English. "
                f"Offer a realistic special plan or discount that matches their usage pattern. "
                f"Keep it under 180 words, professional yet friendly, and end with a clear call-to-action."
            )
            try:
                ai_response = client.models.generate_content(
                    model="gemini-3.8-flash",
                    contents=prompt,
                )
                response["retention_email"] = ai_response.text.strip()
            except Exception as gemini_err:
                response["retention_email"] = (
                    f"Could not generate AI retention plan (Gemini error): {str(gemini_err)}"
                )
        else:
            if risk_level == "Low":
                response["retention_email"] = (
                    "Customer looks stable. No aggressive retention offer needed. "
                    "Consider a light loyalty message if desired."
                )
            else:
                response["retention_email"] = (
                    "Gemini client not available. Set GEMINI_API_KEY environment variable."
                )

        return response

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))