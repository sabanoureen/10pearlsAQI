from fastapi import FastAPI
from datetime import datetime, timedelta
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import shap
import json

app = FastAPI(title="AQI Forecast API")


# =====================================================
# Utilities
# =====================================================

MODEL_REGISTRY_PATH = Path("model_registry.json")


def load_production_model():
    if not MODEL_REGISTRY_PATH.exists():
        raise RuntimeError("model_registry.json not found")

    with open(MODEL_REGISTRY_PATH, "r") as f:
        registry = json.load(f)

    production_model = None

    for model in registry:
        if model.get("status") == "production":
            production_model = model
            break

    if not production_model:
        raise RuntimeError("No production model found")

    model_path = production_model["model_path"]
    features = production_model["features"]

    model = joblib.load(model_path)

    return model, features, production_model


from pymongo import MongoClient
import os

MONGO_URI = os.getenv("MONGO_URI")


def load_latest_features(features):

    if not MONGO_URI:
        raise RuntimeError("MONGO_URI not set in environment")

    client = MongoClient(MONGO_URI)
    db = client.get_default_database()

    collection = db["historical_hourly_data"]

    # Load last 200 rows (for lag + rolling safety)
    data = list(
        collection.find({}, {"_id": 0})
        .sort("datetime", -1)
        .limit(200)
    )

    if not data:
        raise RuntimeError("No historical data found")

    df = pd.DataFrame(data)

    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values("datetime")

    # Same preprocessing as training
    df = df.dropna(subset=["pm2_5"])
    df["pm2_5"] = np.log1p(df["pm2_5"])

    from app.pipelines.feature_engineering import (
        add_time_features,
        add_lag_features,
        add_rolling_features,
    )

    df = add_time_features(df)
    df = add_lag_features(df)
    df = add_rolling_features(df)

    df = df.dropna().reset_index(drop=True)

    latest_row = df.tail(1)

    return latest_row[features]



# =====================================================
# Health Check
# =====================================================

@app.get("/")
def health():
    return {"status": "API running"}


# =====================================================
# Multi-Day Forecast
# =====================================================

@app.get("/forecast/multi")
def multi_forecast(horizon: int = 1):

    try:
        model, features, meta = load_production_model()
        X = load_latest_features(features)

        # Predict in log scale
        preds_log = model.predict(X)

        # Convert back to real AQI
        preds = np.expm1(preds_log)

        results = []

        for i in range(horizon):
            future_time = datetime.utcnow() + timedelta(days=i + 1)

            results.append({
                "datetime": future_time,
                "predicted_aqi": float(preds[0])
            })

        return {
            "status": "success",
            "horizon": horizon,
            "generated_at": datetime.utcnow(),
            "model_version": meta.get("model_name"),
            "predictions": results
        }

    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }


# =====================================================
# Latest Forecast (single day)
# =====================================================

@app.get("/forecast/latest")
def latest_forecast():

    try:
        model, features, meta = load_production_model()
        X = load_latest_features(features)

        preds_log = model.predict(X)
        preds = np.expm1(preds_log)

        return {
            "status": "success",
            "prediction": float(preds[0]),
            "generated_at": datetime.utcnow(),
            "model_name": meta.get("model_name")
        }

    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }


# =====================================================
# SHAP Explainability
# =====================================================

@app.get("/forecast/shap")
def shap_explain():

    try:
        model, features, meta = load_production_model()
        X = load_latest_features(features)

        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)

        contributions = []

        for feature, value in zip(features, shap_values[0]):
            contributions.append({
                "feature": feature,
                "shap_value": float(value)
            })

        preds_log = model.predict(X)
        preds = np.expm1(preds_log)

        return {
            "status": "success",
            "model_name": meta.get("model_name"),
            "prediction": float(preds[0]),
            "generated_at": datetime.utcnow(),
            "contributions": contributions
        }

    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }


# =====================================================
# Model Metrics
# =====================================================

@app.get("/models/metrics")
def model_metrics():

    try:
        if not MODEL_REGISTRY_PATH.exists():
            raise RuntimeError("model_registry.json missing")

        with open(MODEL_REGISTRY_PATH, "r") as f:
            registry = json.load(f)

        return {
            "status": "success",
            "models": registry
        }

    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }
