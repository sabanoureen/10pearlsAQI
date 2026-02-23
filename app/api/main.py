from fastapi import FastAPI, HTTPException
from datetime import datetime, timedelta
import numpy as np
import joblib
import io
from gridfs import GridFS

from app.db.mongo import (
    get_database,
    get_model_registry,
    get_feature_store
)


app = FastAPI(title="Karachi AQI Backend")


# ---------------------------------------------------
# HEALTH
# ---------------------------------------------------
@app.get("/")
def health():
    return {
        "status": "success",
        "message": "AQI Backend Running"
    }


# ---------------------------------------------------
# LOAD PRODUCTION MODEL (GridFS)
# ---------------------------------------------------
import os
import joblib

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_DIR = os.path.join(BASE_DIR, "models")

def load_production_model(horizon: int):

    model_path = os.path.join(MODEL_DIR, f"rf_h{horizon}.pkl")

    if not os.path.exists(model_path):
        raise HTTPException(
            status_code=500,
            detail=f"Model file not found at {model_path}"
        )

    model = joblib.load(model_path)

    # Since features are same for all models, hardcode or load from registry if needed
    feature_columns = [
        "hour",
        "day",
        "month",
        "lag_1",
        "lag_3",
        "lag_6",
        "roll_mean_6",
        "roll_mean_12",
    ]

    model_name = f"RandomForest_h{horizon}"

    return model, feature_columns, model_name

# ---------------------------------------------------
# GET LATEST FEATURE ROW (FROM FEATURE STORE)
# ---------------------------------------------------
def get_latest_feature_row(feature_columns):

    feature_store = get_feature_store()

    latest_doc = feature_store.find_one(
        sort=[("datetime", -1)]
    )

    if not latest_doc:
        raise HTTPException(
            status_code=500,
            detail="Feature store empty. Run training first."
        )

    try:
        row = [latest_doc[col] for col in feature_columns]
    except KeyError as e:
        raise HTTPException(
            status_code=500,
            detail=f"Missing feature column in feature_store: {e}"
        )

    return np.array(row).reshape(1, -1)


# ---------------------------------------------------
# SIMPLE 3-DAY FORECAST (for Streamlit)
# ---------------------------------------------------
@app.get("/forecast")
def forecast():

    results = {}

    for horizon in [1, 2, 3]:

        model, features, model_name = load_production_model(horizon)
        X = get_latest_feature_row(features)

        prediction = float(model.predict(X)[0])

        future_date = (
            datetime.utcnow() + timedelta(days=horizon)
        ).strftime("%Y-%m-%d")

        results[f"{horizon}_day"] = {
            "value": round(prediction, 2),
            "date": future_date,
            "model": model_name
        }

    return results


# ---------------------------------------------------
# MULTI-HORIZON CUSTOM FORECAST
# ---------------------------------------------------
@app.get("/forecast/multi")
def forecast_multi(days: int = 3):

    if days < 1 or days > 7:
        raise HTTPException(
            status_code=400,
            detail="Days must be between 1 and 7"
        )

    results = []
    base_time = datetime.utcnow()

    for horizon in range(1, days + 1):

        model, features, model_name = load_production_model(horizon)
        X = get_latest_feature_row(features)

        prediction = float(model.predict(X)[0])

        results.append({
            "date": (base_time + timedelta(days=horizon)).strftime("%Y-%m-%d"),
            "value": round(prediction, 2),
            "model": model_name
        })

    return {
        "status": "success",
        "generated_at": base_time,
        "predictions": results
    }


# ---------------------------------------------------
# MODEL METRICS
# ---------------------------------------------------
@app.get("/models/metrics")
def metrics():

    registry = get_model_registry()
    docs = list(registry.find({}, {"_id": 0}))

    return {
        "status": "success",
        "models": docs
    }


# ---------------------------------------------------
# BEST MODEL
# ---------------------------------------------------
@app.get("/models/best")
def best_model():

    registry = get_model_registry()

    doc = registry.find_one({
        "status": "production",
        "is_best": True
    }, {"_id": 0})

    if not doc:
        raise HTTPException(
            status_code=404,
            detail="No production model found"
        )

    return {
        "status": "success",
        "model": doc
    }


# ---------------------------------------------------
# FEATURE IMPORTANCE
# ---------------------------------------------------
@app.get("/features/importance")
def feature_importance(horizon: int = 1):

    model, features, _ = load_production_model(horizon)

    if not hasattr(model, "feature_importances_"):
        raise HTTPException(
            status_code=400,
            detail="Model does not support feature importance"
        )

    data = [
        {
            "feature": f,
            "importance": float(i)
        }
        for f, i in zip(features, model.feature_importances_)
    ]

    return {
        "status": "success",
        "features": data
    }


# ---------------------------------------------------
# TRAIN ENDPOINT
# ---------------------------------------------------
# ---------------------------------------------------
# TRAIN ENDPOINT (Railway Safe)
# ---------------------------------------------------
@app.get("/train/{horizon}")
def train_endpoint(horizon: int):

    if horizon not in [1, 2, 3]:
        raise HTTPException(status_code=400, detail="Horizon must be 1, 2 or 3")

    from app.pipelines.training_pipeline import run_training

    run_training(horizon)

    return {
        "status": "success",
        "message": f"Model trained inside Railway for horizon {horizon}"
    }