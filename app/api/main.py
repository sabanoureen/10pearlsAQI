from fastapi import FastAPI, HTTPException
from datetime import datetime, timedelta
from pathlib import Path
import numpy as np
import joblib
import json
import pandas as pd

from app.db.mongo import get_db

app = FastAPI(title="AQI Forecast API")


# =========================================================
# HEALTH
# =========================================================
@app.get("/")
def health():
    return {"status": "ok", "message": "AQI API running"}


# =========================================================
# MODEL PATH RESOLVER (SAFE)
# =========================================================
BASE_DIR = Path(__file__).resolve().parents[2]
MODELS_DIR = BASE_DIR / "models"


def get_latest_model_file(horizon: int) -> Path:
    """
    Returns latest joblib file inside models/rf_h{horizon}
    """
    folder = MODELS_DIR / f"rf_h{horizon}"

    if not folder.exists():
        raise HTTPException(500, f"Model folder missing: {folder}")

    files = sorted(folder.glob("*.joblib"))

    if not files:
        raise HTTPException(500, f"No model files in {folder}")

    return files[-1]


# =========================================================
# LOAD MODEL + FEATURES
# =========================================================
def load_production_model(horizon: int):
    model_path = get_latest_model_file(horizon)

    features_path = model_path.parent / "features.json"

    if not features_path.exists():
        raise HTTPException(500, f"features.json missing in {model_path.parent}")

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


# =========================================================
# GET LATEST FEATURES FROM MONGO
# =========================================================
def get_latest_features(feature_columns):
    db = get_db()
    col = db["historical_hourly_data"]

    doc = col.find_one({}, sort=[("datetime", -1)])

    if not doc:
        raise HTTPException(404, "No historical data found")

    X = [doc.get(c, 0) for c in feature_columns]

    return np.array(X).reshape(1, -1)


# =========================================================
# FORECAST
# =========================================================
@app.get("/forecast/multi")
def forecast_multi(horizon: int = 1):
    model, features = load_production_model(horizon)
    X_latest = get_latest_features(features)

# FIX: align columns with training
    X_latest = pd.DataFrame(X_latest, columns=features)

    log_pred = model.predict(X_latest)[0]

    pred = float(np.expm1(log_pred))

    return {
        "status": "success",
        "horizon": horizon,
        "prediction": pred,
        "forecast_for": datetime.utcnow() + timedelta(days=horizon),
        "generated_at": datetime.utcnow()
    }


# =========================================================
# SHAP
# =========================================================
@app.get("/forecast/shap")
def shap_explain(horizon: int = 1):
    import shap

    model, features = load_production_model(horizon)
    X_latest= get_latest_features(features)
    X_latest = pd.DataFrame(X_latest, columns=features)


    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    log_pred = model.predict(X)[0]
    pred = float(np.expm1(log_pred))

    contrib = [
        {"feature": f, "shap": float(shap_values[0][i])}
        for i, f in enumerate(features)
    ]

    return {
        "prediction": pred,
        "contributions": contrib
    }
