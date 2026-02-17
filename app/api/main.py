from fastapi import FastAPI, HTTPException
from datetime import datetime, timedelta
from pathlib import Path
import numpy as np
import joblib
import json
import pandas as pd

from app.db.mongo import get_db
from app.db.mongo import get_model_registry
from app.pipelines.training_pipeline import run_training_pipeline

app = FastAPI(title="AQI Forecast API")


# =====================================================
# HEALTH
# =====================================================
@app.get("/")
def health():
    return {"status": "success", "message": "AQI API running"}


# =====================================================
# LOAD PRODUCTION MODEL
# =====================================================
def load_production_model(horizon: int):
    registry = get_model_registry()

    doc = registry.find_one({"horizon": horizon, "is_best": True})

    if not doc:
        raise HTTPException(404, "No production model")

    model_path = doc["model_path"]

    if not Path(model_path).exists():
        raise HTTPException(500, "Model file missing")

    model = joblib.load(model_path)
    return model, doc["features"]


# =====================================================
# LATEST FEATURES
# =====================================================
def get_latest_features(columns):
    db = get_db()
    col = db["historical_hourly_data"]

    doc = col.find_one({}, sort=[("datetime", -1)])

    if not doc:
        raise HTTPException(404, "No data")

    X = [doc.get(c, 0) for c in columns]
    return np.array(X).reshape(1, -1)


# =====================================================
# FORECAST
# =====================================================
@app.get("/forecast/multi")
def forecast_multi(horizon: int = 1):
    model, features = load_production_model(horizon)
    X_latest = get_latest_features(features)

    preds = []
    base = datetime.utcnow()

    for d in range(1, horizon + 1):
        log_p = model.predict(X_latest)[0]
        p = float(np.expm1(log_p))

        preds.append({
            "datetime": base + timedelta(days=d),
            "predicted_aqi": p
        })

    return {
        "status": "success",
        "horizon": horizon,
        "generated_at": datetime.utcnow(),
        "predictions": preds
    }


# =====================================================
# MODEL METRICS
# =====================================================
@app.get("/models/metrics")
def model_metrics():
    reg = get_model_registry()
    docs = list(reg.find({}, {"_id": 0}))

    if not docs:
        return {"status": "error", "detail": "No models"}

    return {"status": "success", "models": docs}


# =====================================================
# BEST MODEL
# =====================================================
@app.get("/models/best")
def best_model():
    reg = get_model_registry()
    doc = reg.find_one({"is_best": True}, {"_id": 0})

    if not doc:
        return {"status": "error", "detail": "No best model"}

    return {"status": "success", "model": doc}


# =====================================================
# FEATURE IMPORTANCE
# =====================================================
@app.get("/features/importance")
def feature_importance(horizon: int = 1):
    model, features = load_production_model(horizon)

    if not hasattr(model, "feature_importances_"):
        return {"status": "error", "detail": "No importance"}

    data = [
        {"feature": f, "importance": float(i)}
        for f, i in zip(features, model.feature_importances_)
    ]

    return {"status": "success", "features": data}


# =====================================================
# SHAP
# =====================================================
@app.get("/forecast/shap")
def shap_explain(horizon: int = 1):
    import shap

    model, features = load_production_model(horizon)
    X_latest = get_latest_features(features)

    explainer = shap.TreeExplainer(model)
    shap_vals = explainer.shap_values(X_latest)

    contrib = [
        {"feature": f, "shap_value": float(v)}
        for f, v in zip(features, shap_vals[0])
    ]

    pred = float(np.expm1(model.predict(X_latest)[0]))

    return {
        "status": "success",
        "prediction": pred,
        "contributions": contrib
    }