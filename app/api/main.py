from fastapi import FastAPI, HTTPException
from datetime import datetime
from pydantic import BaseModel
from typing import List

from app.db.mongo import get_model_registry
from app.pipelines.inference import predict_aqi, predict_multi_aqi

app = FastAPI(title="AQI Prediction API")


@app.get("/")
def root():
    return {"message": "AQI API is running"}


class MultiPredictRequest(BaseModel):
    horizons: List[int]


@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "AQI API",
        "time": datetime.utcnow().isoformat()
    }


@app.get("/models/best")
def best_model(horizon: int = 1):
    model = get_model_registry().find_one(
        {"horizon": horizon, "is_best": True},
        {"_id": 0}
    )

    if not model:
        raise HTTPException(status_code=404, detail="No production model found")

    return {
        "status": "success",
        "model": model
    }


@app.get("/predict")
def predict(horizon: int = 1):
    return predict_aqi(horizon)


@app.post("/predict/multi")
def predict_multi(payload: MultiPredictRequest):
    return predict_multi_aqi(payload.horizons)
