from fastapi import FastAPI, HTTPException
from datetime import datetime, timedelta
from pathlib import Path
import numpy as np
import joblib

from app.db.mongo import get_db

app = FastAPI(title="AQI Forecast API")


# ---------------------------------------------------
# Health Check
# ---------------------------------------------------
@app.get("/")
def health_check():
    return {"status": "success", "message": "AQI API running"}


# ---------------------------------------------------
# Find latest model file automatically
# ---------------------------------------------------
def get_latest_model_file(horizon: int):

    BASE_DIR = Path(__file__).resolve().parents[2]
    model_dir = BASE_DIR / f"models/rf_h{horizon}"

    if not model_dir.exists():
        raise HTTPException(500, f"Model folder not found: {model_dir}")

    model_files = list(model_dir.glob("*.joblib"))

    if not model_files:
        raise HTTPException(500, f"No model files in {model_dir}")

    latest_model = max(model_files, key=lambda f: f.stat().st_mtime)

    return latest_model



# ---------------------------------------------------
# Load model + features
# ---------------------------------------------------
def load_production_model(horizon: int):

    model_path = get_latest_model_file(horizon)

    # features.json stored in same folder
    features_path = model_path.parent / "features.json"

    if not features_path.exists():
        raise HTTPException(500, f"features.json missing in {model_path.parent}")

    import json

    with open(features_path) as f:
        data = json.load(f)

    if isinstance(data, dict):
        features = data.get("features") or data.get("feature_columns")
    else:
        features = data

    if not features:
        raise HTTPException(500, "Invalid features.json format")


    model = joblib.load(model_path)

    return model, features


# ---------------------------------------------------
# Get latest feature row from Mongo
# ---------------------------------------------------
def get_latest_features(feature_columns):

    db = get_db()
    collection = db["historical_hourly_data"]

    latest_doc = collection.find_one({}, sort=[("datetime", -1)])

    if not latest_doc:
        raise HTTPException(404, "No data found")

    X = [latest_doc.get(col, 0) for col in feature_columns]

    return np.array(X).reshape(1, -1)


# ---------------------------------------------------
# Forecast Endpoint
# ---------------------------------------------------
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
        return {"status": "error", "message": str(e)}


# ---------------------------------------------------
# SHAP Endpoint
# ---------------------------------------------------
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
        return {"status": "error", "message": str(e)}
