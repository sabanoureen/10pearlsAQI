from fastapi import FastAPI, HTTPException
from datetime import datetime, timedelta
import numpy as np
import joblib
import io
from gridfs import GridFS

from app.db.mongo import get_database, get_model_registry

app = FastAPI(title="AQI Forecast API")


# =====================================================
# Health Check
# =====================================================
@app.get("/")
def health():
    return {"message": "AQI Backend Running"}


# =====================================================
# Load Production Model From GridFS
# =====================================================
def load_production_model(horizon: int):
    registry = get_model_registry()

    doc = registry.find_one({
        "horizon": horizon,
        "status": "production",
        "is_best": True
    })

    if not doc:
        raise HTTPException(status_code=404, detail=f"No production model for horizon {horizon}")

    gridfs_id = doc.get("gridfs_id")
    if not gridfs_id:
        raise HTTPException(status_code=500, detail="Model missing GridFS ID")

    db = get_database()
    fs = GridFS(db)

    model_bytes = fs.get(gridfs_id).read()
    model = joblib.load(io.BytesIO(model_bytes))

    return model, doc["features"], doc["model_name"]


# =====================================================
# Get Latest Features From Mongo
# =====================================================
def get_latest_features(columns):
    db = get_database()
    col = db["historical_hourly_data"]

    doc = col.find_one({}, sort=[("datetime", -1)])

    if not doc:
        raise HTTPException(status_code=404, detail="No historical data found")

    X = [doc.get(col_name, 0) for col_name in columns]

    return np.array(X).reshape(1, -1)


# =====================================================
# Simple 3-Day Forecast (Streamlit)
# =====================================================
@app.get("/forecast")
def forecast():
    results = {}

    for horizon in [1, 2, 3]:
        model, features, model_name = load_production_model(horizon)

        X = get_latest_features(features)

        pred = float(model.predict(X)[0])

        future_date = (
            datetime.utcnow() + timedelta(days=horizon)
        ).strftime("%Y-%m-%d")

        results[f"{horizon}_day"] = {
            "value": round(pred, 2),
            "date": future_date,
            "model": model_name
        }

    return results


# =====================================================
# Multi-Day Forecast
# =====================================================
@app.get("/forecast/multi")
def forecast_multi(horizon: int = 1):

    model, features, model_name = load_production_model(horizon)
    X = get_latest_features(features)

    preds = []
    base = datetime.utcnow()

    for d in range(1, horizon + 1):
        pred = float(model.predict(X)[0])

        preds.append({
            "datetime": base + timedelta(days=d),
            "predicted_aqi": round(pred, 2)
        })

    return {
        "status": "success",
        "horizon": horizon,
        "model": model_name,
        "generated_at": datetime.utcnow(),
        "predictions": preds
    }


# =====================================================
# List All Models
# =====================================================
@app.get("/models")
def list_models():
    registry = get_model_registry()
    docs = list(registry.find({}, {"_id": 0}))

    return {"status": "success", "models": docs}


# =====================================================
# Temporary Training Endpoint
# =====================================================
@app.get("/train/{horizon}")
def train_model_endpoint(horizon: int):
    from app.pipelines.training_pipeline import run_training

    run_training(horizon)

    return {
        "status": "training completed",
        "horizon": horizon
    }