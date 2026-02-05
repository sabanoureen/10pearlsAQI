from fastapi import FastAPI, HTTPException
from datetime import datetime, timezone
from typing import List

from pymongo.errors import PyMongoError

from app.db.mongo import get_model_registry, get_feature_store
from app.pipelines.inference import predict_aqi, predict_multi_aqi

app = FastAPI(title="AQI Prediction API")

# -------------------------------------------------
# HEALTH CHECK
# -------------------------------------------------
@app.get("/health")
def health():
    return {
        "status": "ok",
        "timestamp": datetime.now(timezone.utc)
    }


# -------------------------------------------------
# GET BEST MODEL
# -------------------------------------------------
@app.get("/models/best")
def get_best_model(horizon: int = 1):
    try:
        registry = get_model_registry()

        model = registry.find_one(
            {"horizon": horizon},
            sort=[("rmse", 1)]
        )

        if not model:
            return {"message": "No production model found"}

        model["_id"] = str(model["_id"])
        return model

    except PyMongoError as e:
        raise HTTPException(status_code=500, detail=str(e))


# -------------------------------------------------
# SINGLE AQI PREDICTION
# -------------------------------------------------
@app.get("/predict")
def predict(horizon: int = 1):
    try:
        registry = get_model_registry()
        feature_store = get_feature_store()

        model = registry.find_one(
            {"horizon": horizon},
            sort=[("rmse", 1)]
        )

        if not model:
            raise HTTPException(status_code=404, detail="No model available")

        result = predict_aqi(
            horizon=horizon,
            model_doc=model,
            feature_store=feature_store
        )

        return result

    except PyMongoError as e:
        raise HTTPException(status_code=500, detail=str(e))


# -------------------------------------------------
# MULTI AQI PREDICTION
# -------------------------------------------------
@app.post("/predict/multi")
def predict_multi(horizons: List[int]):
    try:
        registry = get_model_registry()
        feature_store = get_feature_store()

        results = {}

        for h in horizons:
            model = registry.find_one(
                {"horizon": h},
                sort=[("rmse", 1)]
            )

            if not model:
                results[h] = {"error": "No model found"}
                continue

            results[h] = predict_multi_aqi(
                horizon=h,
                model_doc=model,
                feature_store=feature_store
            )

        return results

    except PyMongoError as e:
        raise HTTPException(status_code=500, detail=str(e))
