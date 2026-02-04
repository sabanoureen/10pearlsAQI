from fastapi import FastAPI
from datetime import datetime, timezone
from typing import List
from pymongo.errors import PyMongoError

from app.api.routes import router
from app.pipelines.inference import predict_aqi, predict_multi_aqi
from app.db.mongo import client, model_registry, feature_store

app = FastAPI(title="AQI Prediction API")

# ----------------------------
# ROUTERS
# ----------------------------
app.include_router(router)

# ----------------------------
# HEALTH CHECK
# ----------------------------
@app.get("/health")
def health():
    try:
        client.admin.command("ping")
        return {
            "status": "healthy",
            "service": "AQI API",
            "timestamp": datetime.utcnow().isoformat()
        }
    except PyMongoError as e:
        return {
            "status": "unhealthy",
            "error": str(e)
        }

# ----------------------------
# SINGLE PREDICTION
# ----------------------------
@app.get("/predict")
def predict(horizon: int = 1, version: str | None = None):
    return predict_aqi(horizon, version)

# ----------------------------
# MULTI PREDICTION
# ----------------------------
@app.get("/predict/multi")
def predict_multi(horizons: List[int]):
    return predict_multi_aqi(horizons)

# ----------------------------
# BEST MODEL
# ----------------------------
@app.get("/models/best")
def best_model(horizon: int = 1):
    model = model_registry.find_one(
        {"horizon": horizon, "is_best": True},
        {"_id": 0}
    )
    if not model:
        return {"status": "not_found"}
    return {"status": "ok", "best_model": model}
