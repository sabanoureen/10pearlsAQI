from fastapi import FastAPI, HTTPException
from datetime import datetime, timedelta
from pathlib import Path
import numpy as np
import joblib
import json

from app.db.mongo import get_db

app = FastAPI(title="AQI Forecast API")


# =====================================================
# CONFIG
# =====================================================
BASE_DIR = Path(__file__).resolve().parents[2]
MODELS_DIR = BASE_DIR / "models"


# =====================================================
# HEALTH CHECK
# =====================================================
@app.get("/")
def health_check():
    return {"status": "ok", "message": "AQI API running"}


# =====================================================
# FIND LATEST MODEL FILE
# =====================================================
def get_latest_model_file(horizon: int) -> Path:
    """
    Finds latest .joblib file inside models/*_h{horizon}/
    Example: models/rf_h1/model_*.joblib
    """
    candidates = list(MODELS_DIR.glob(f"*h{horizon}/*.joblib"))

    if not candidates:
        raise HTTPException(
            status_code=500,
            detail=f"No model found for horizon {horizon} in {MODELS_DIR}"
        )

    latest = max(candidates, key=lambda p: p.stat().st_mtime)
    return latest


# =====================================================
# LOAD MODEL + FEATURES
# =====================================================
def load_production_model(horizon: int):
    model_path = get_latest_model_file(horizon)
    features_path = model_path.parent / "features.json"

    if not features_path.exists():
        raise HTTPException(
            status_code=500,
            detail=f"features.json missing in {model_path.parent}"
        )

    with open(features_path) as f:
        data = json.load(f)

    if isinstance(data, dict):
        features = data.get("features") or data.get("feature_columns")
    else:
        features = data

    if not features:
        raise HTTPException(500, "Invalid features.json")

    model = joblib.load(model_path)

    return model, features


# =====================================================
# GET LATEST FEATURES FROM MONGO
# =====================================================
def get_latest_features(feature_columns):
    db = get_db()
    collection = db["historical_hourly_data"]

    latest_doc = collection.find_one({}, sort=[("datetime", -1)])

    if not latest_doc:
        raise HTTPException(500, "No historical data in Mongo")

    X = [latest_doc.get(col, 0) for col in feature_columns]
    return np.array(X).reshape(1, -1)


# =====================================================
# FORECAST ENDPOINT
# =====================================================
@app.get("/forecast/multi")
def multi_forecast(horizon: int = 1):
    try:
        model, features = load_production_model(horizon)
        X_latest = get_latest_features(features)

        log_pred = model.predict(X_latest)[0]
        prediction = float(np.expm1(log_pred))

        forecast_time = datetime.utcnow() + timedelta(days=horizon)

        return {
            "status": "success",
            "horizon": horizon,
            "prediction": prediction,
            "forecast_for": forecast_time,
            "generated_at": datetime.utcnow()
        }

    except Exception as e:
        raise HTTPException(500, str(e))


# =====================================================
# SHAP ENDPOINT
# =====================================================
@app.get("/forecast/shap")
def shap_explain(horizon: int = 1):
    try:
        import shap

        model, features = load_production_model(horizon)
        X_latest = get_latest_features(features)

        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_latest)

        contributions = [
            {"feature": f, "shap_value": float(shap_values[0][i])}
            for i, f in enumerate(features)
        ]

        log_pred = model.predict(X_latest)[0]
        prediction = float(np.expm1(log_pred))

        return {
            "status": "success",
            "prediction": prediction,
            "contributions": contributions,
            "generated_at": datetime.utcnow()
        }

    except Exception as e:
        raise HTTPException(500, str(e))
