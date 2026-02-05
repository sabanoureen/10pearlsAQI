from fastapi import FastAPI, HTTPException
from datetime import datetime
from pymongo.errors import PyMongoError
from pydantic import BaseModel
from typing import List

from app.db.mongo import get_model_registry
from app.pipelines.inference import predict_aqi, predict_multi_aqi

app = FastAPI(title="AQI Prediction API")


# -------------------
# SCHEMAS
# -------------------
class MultiPredictRequest(BaseModel):
    horizons: List[int]


# -------------------
# HEALTH
# -------------------
@app.get("/health")
def health():
    try:
        get_model_registry().find_one()
        return {
            "status": "ok",
            "time": datetime.utcnow().isoformat()
        }
    except PyMongoError as e:
        raise HTTPException(status_code=500, detail=str(e))


# -------------------
# BEST MODEL
# -------------------
@app.get("/models/best")
def best_model(horizon: int = 1):
    model = get_model_registry().find_one(
        {"horizon": horizon},
        sort=[("rmse", 1)]
    )
    if not model:
        raise HTTPException(status_code=404, detail="No production model found")

    model["_id"] = str(model["_id"])

    return {
        "status": "success",
        "model": model
    }


# -------------------
# SINGLE PREDICT
# -------------------
@app.get("/predict")
def predict(horizon: int = 1):
    result = predict_aqi(horizon)
    return {
        "status": "success",
        **result
    }


# -------------------
# MULTI PREDICT
# -------------------
@app.post("/predict/multi")
def predict_multi(payload: MultiPredictRequest):
    predictions = predict_multi_aqi(payload.horizons)
    return {
        "status": "success",
        "predictions": predictions
    }
