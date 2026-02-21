from fastapi import FastAPI, HTTPException
from datetime import datetime, timedelta
import numpy as np
import joblib
import io
from gridfs import GridFS

from app.db.mongo import get_database, get_model_registry


app = FastAPI(title="AQI Forecast API")


# ---------------------------------------------------
# Health
# ---------------------------------------------------
@app.get("/")
def health():
    return {"status": "success", "message": "AQI API running"}


# ---------------------------------------------------
# Load Production Model (GridFS Version)
# ---------------------------------------------------
def load_production_model(horizon: int):

    registry = get_model_registry()
    db = get_database()
    fs = GridFS(db)

    doc = registry.find_one({
        "horizon": horizon,
        "is_best": True
    })

    if not doc:
        raise HTTPException(status_code=404, detail="No production model found")

    gridfs_id = doc.get("gridfs_id")

    if not gridfs_id:
        raise HTTPException(status_code=500, detail="Model missing GridFS ID")

    try:
        model_bytes = fs.get(gridfs_id).read()
        buffer = io.BytesIO(model_bytes)
        model = joblib.load(buffer)
    except Exception:
        raise HTTPException(status_code=500, detail="Failed to load model from GridFS")

    return model, doc["features"]


# ---------------------------------------------------
# Latest Feature Row
# ---------------------------------------------------
def get_latest_features(columns):

    db = get_database()
    col = db["historical_hourly_data"]

    doc = col.find_one({}, sort=[("datetime", -1)])

    if not doc:
        raise HTTPException(status_code=404, detail="No historical data found")

    X = [doc.get(c, 0) for c in columns]

    return np.array(X).reshape(1, -1)


# ---------------------------------------------------
# Multi Forecast
# ---------------------------------------------------
@app.get("/forecast/multi")
def forecast_multi(horizon: int = 1):

    model, features = load_production_model(horizon)
    X = get_latest_features(features)

    preds = []
    base = datetime.utcnow()

    for d in range(1, horizon + 1):
        log_pred = model.predict(X)[0]
        pred = float(np.expm1(log_pred))

        preds.append({
            "datetime": base + timedelta(days=d),
            "predicted_aqi": pred
        })

    return {
        "status": "success",
        "horizon": horizon,
        "generated_at": datetime.utcnow(),
        "predictions": preds
    }


# ---------------------------------------------------
# SHAP
# ---------------------------------------------------
@app.get("/forecast/shap")
def shap_explain(horizon: int = 1):

    import shap

    model, features = load_production_model(horizon)
    X = get_latest_features(features)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    contrib = [
        {"feature": f, "shap_value": float(v)}
        for f, v in zip(features, shap_values[0])
    ]

    pred = float(np.expm1(model.predict(X)[0]))

    return {
        "status": "success",
        "prediction": pred,
        "contributions": contrib
    }


# ---------------------------------------------------
# Model Metrics
# ---------------------------------------------------
@app.get("/models/metrics")
def metrics():

    registry = get_model_registry()
    docs = list(registry.find({}, {"_id": 0}))

    return {"status": "success", "models": docs}


# ---------------------------------------------------
# Best Model
# ---------------------------------------------------
@app.get("/models/best")
def best_model():

    registry = get_model_registry()
    doc = registry.find_one({"is_best": True}, {"_id": 0})

    if not doc:
        return {"status": "error", "detail": "No best model found"}

    return {"status": "success", "model": doc}


# ---------------------------------------------------
# Feature Importance
# ---------------------------------------------------
@app.get("/features/importance")
def importance(horizon: int = 1):

    model, features = load_production_model(horizon)

    if not hasattr(model, "feature_importances_"):
        return {"status": "error", "detail": "Model does not support feature importance"}

    data = [
        {"feature": f, "importance": float(i)}
        for f, i in zip(features, model.feature_importances_)
    ]

    return {"status": "success", "features": data}


# ---------------------------------------------------
# Simple 3-Day Forecast (for Streamlit)
# ---------------------------------------------------
@app.get("/forecast")
def forecast():

    results = {}

    for horizon in [1, 2, 3]:

        model, features = load_production_model(horizon)
        X = get_latest_features(features)

        log_pred = model.predict(X)[0]
        pred = float(np.expm1(log_pred))

        future_date = (
            datetime.utcnow() + timedelta(days=horizon)
        ).strftime("%Y-%m-%d")

        results[f"{horizon}_day"] = {
            "value": round(pred, 2),
            "date": future_date,
            "model": "random_forest"
        }

    return results