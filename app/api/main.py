from fastapi import FastAPI
from datetime import datetime, timezone
from typing import List

from pymongo.errors import PyMongoError

from app.pipelines.inference import predict_aqi, predict_multi_aqi
from app.db.mongo import client, model_registry

app = FastAPI(title="AQI Prediction API")

# -----------------------------------
# HEALTH CHECK
# -----------------------------------
@app.get("/health")
def health_check():
    try:
        client.admin.command("ping")
        return {
            "status": "healthy",
            "mongodb": "connected",
            "service": "AQI Prediction API",
            "timestamp": datetime.utcnow().isoformat()
        }
    except PyMongoError as e:
        return {
            "status": "unhealthy",
            "error": str(e)
        }

# -----------------------------------
# SINGLE AQI PREDICTION
# -----------------------------------
@app.get("/predict")
def predict(horizon: int = 1):
    result = predict_aqi(horizon)

    if result["status"] != "success":
        return result

    return {
        "status": "success",
        "horizon_hours": horizon,
        "predicted_aqi": result["predicted_aqi"],
        "model_name": result["model_name"],
        "version": result.get("version"),
        "timestamp": datetime.utcnow().isoformat()
    }

# -----------------------------------
# MULTI-HORIZON PREDICTION
# -----------------------------------
@app.get("/predict/multi")
def predict_multi(horizons: List[int]):
    return predict_multi_aqi(horizons)

# -----------------------------------
# BEST MODEL
# -----------------------------------
@app.get("/models/best")
def get_best_model(horizon: int = 1):
    model = model_registry.find_one(
        {"horizon": horizon, "is_best": True},
        {"_id": 0}
    )

    if not model:
        return {"status": "not_found"}

    return {"status": "ok", "best_model": model}
