from fastapi import FastAPI, HTTPException
from datetime import datetime, timedelta
from pathlib import Path
import numpy as np
import joblib
import shap

from app.db.mongo import get_db, get_model_registry

app = FastAPI(title="AQI Forecast API")


# ---------------------------------------------------
# Load production model
# ---------------------------------------------------
def load_production_model(horizon: int):

    registry = get_model_registry()

    doc = registry.find_one({
        "horizon": horizon,
        "is_best": True
    })

    if not doc:
        raise HTTPException(status_code=404, detail="No production model")

    model_path = doc["model_path"]

    if not Path(model_path).exists():
        raise HTTPException(status_code=500, detail="Model file missing")

    model = joblib.load(model_path)

    return model, doc["features"], doc


# ---------------------------------------------------
# Latest features
# ---------------------------------------------------
def get_latest_features(feature_columns):

    db = get_db()
    latest = db["historical_hourly_data"].find_one(
        {}, sort=[("datetime", -1)]
    )

    if not latest:
        raise HTTPException(status_code=404, detail="No data")

    X = [latest.get(f, 0) for f in feature_columns]
    return np.array(X).reshape(1, -1)


# ---------------------------------------------------
# Forecast
# ---------------------------------------------------
@app.get("/forecast/multi")
def multi_forecast(horizon: int = 1):

    model, features, _ = load_production_model(horizon)
    X_latest = get_latest_features(features)

    preds = []
    base = datetime.utcnow()

    for d in range(1, horizon + 1):
        log_pred = model.predict(X_latest)[0]
        pred = float(np.expm1(log_pred))

        preds.append({
            "datetime": base + timedelta(days=d),
            "predicted_aqi": pred
        })

    return {
        "status": "success",
        "horizon": horizon,
        "predictions": preds
    }


# ---------------------------------------------------
# SHAP
# ---------------------------------------------------
@app.get("/forecast/shap")
def shap_explain(horizon: int = 1):

    model, features, _ = load_production_model(horizon)
    X_latest = get_latest_features(features)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_latest)

    contrib = [
        {"feature": f, "shap_value": float(v)}
        for f, v in zip(features, shap_values[0])
    ]

    pred = float(np.expm1(model.predict(X_latest)[0]))

    return {
        "status": "success",
        "prediction": pred,
        "contributions": contrib
    }


# ---------------------------------------------------
# Metrics
# ---------------------------------------------------
@app.get("/models/metrics")
def model_metrics():

    registry = get_model_registry()
    docs = list(registry.find({}, {"_id": 0}))

    return {
        "status": "success",
        "models": docs
    }


# ---------------------------------------------------
# Best model
# ---------------------------------------------------
@app.get("/models/best")
def best_model():

    registry = get_model_registry()
    doc = registry.find_one({"is_best": True}, {"_id": 0})

    if not doc:
        return {"status": "error", "detail": "No best model"}

    return {"status": "success", "model": doc}


# ---------------------------------------------------
# Feature importance
# ---------------------------------------------------
@app.get("/features/importance")
def feature_importance(horizon: int = 1):

    model, features, _ = load_production_model(horizon)

    if not hasattr(model, "feature_importances_"):
        return {"status": "error", "detail": "No importance"}

    data = [
        {"feature": f, "importance": float(i)}
        for f, i in zip(features, model.feature_importances_)
    ]

    return {"status": "success", "features": data}
