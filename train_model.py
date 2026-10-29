"""
Indian Telecom Customer Churn Prediction - Model Training Script
Trains an XGBoost classifier on the telecom_churn dataset and saves artifacts.
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score, f1_score, confusion_matrix
import xgboost as xgb
import joblib

# Paths
DATA_PATH = "data/telecom_churn.csv"
MODEL_DIR = "saved_models"
os.makedirs(MODEL_DIR, exist_ok=True)


def load_and_preprocess_data(file_path: str):
    print("=" * 60)
    print("1. Loading dataset...")
    df = pd.read_csv(file_path)
    print(f"   Shape: {df.shape}")

    print("\n2. Cleaning data and handling features...")
    # Drop non-predictive / high-cardinality columns
    cols_to_drop = ["customer_id", "date_of_registration", "pincode", "city"]
    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

    # Clip negative usage values (data quality issues)
    for col in ["data_used", "calls_made", "sms_sent"]:
        if col in df.columns:
            neg_count = (df[col] < 0).sum()
            if neg_count > 0:
                print(f"   Clipping {neg_count} negative values in '{col}'")
            df[col] = df[col].clip(lower=0)

    print("\n3. Encoding categorical variables (one-hot, drop_first=True)...")
    df = pd.get_dummies(df, drop_first=True)

    print("\n4. Splitting features and target...")
    X = df.drop(columns=["churn"])
    y = df["churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"   Train: {X_train.shape}, Test: {X_test.shape}")
    print(f"   Churn rate (train): {y_train.mean():.3f}")

    print("\n5. Scaling numerical features...")
    numerical_cols = [
        "age",
        "num_dependents",
        "estimated_salary",
        "calls_made",
        "sms_sent",
        "data_used",
    ]
    numerical_cols = [c for c in numerical_cols if c in X_train.columns]

    scaler = StandardScaler()
    X_train = X_train.copy()
    X_test = X_test.copy()
    X_train[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
    X_test[numerical_cols] = scaler.transform(X_test[numerical_cols])

    # Persist artifacts
    joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.pkl"))
    joblib.dump(X_train.columns.tolist(), os.path.join(MODEL_DIR, "feature_columns.pkl"))
    print(f"   Saved scaler and {len(X_train.columns)} feature columns")

    return X_train, X_test, y_train, y_test


def train_and_save_model():
    X_train, X_test, y_train, y_test = load_and_preprocess_data(DATA_PATH)

    print("\n6. Training XGBoost Model...")
    neg_count = (y_train == 0).sum()
    pos_count = (y_train == 1).sum()
    scale_weight = neg_count / pos_count if pos_count > 0 else 1.0
    print(f"   scale_pos_weight = {scale_weight:.3f}")

    model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        gamma=0.1,
        scale_pos_weight=scale_weight,
        random_state=42,
        eval_metric="logloss",
        n_jobs=-1,
    )

    model.fit(X_train, y_train)

    print("\n7. Evaluating Model on Test Set:")
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    print(classification_report(y_test, y_pred, digits=4))
    print(f"ROC-AUC Score : {roc_auc_score(y_test, y_proba):.4f}")
    print(f"F1-Score      : {f1_score(y_test, y_pred):.4f}")
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    model_path = os.path.join(MODEL_DIR, "xgboost_churn_v1.pkl")
    joblib.dump(model, model_path)
    print(f"\n✅ Model successfully saved to {model_path}")


if __name__ == "__main__":
    train_and_save_model()