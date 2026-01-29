from fastapi import FastAPI
from datetime import datetime, timezone
from typing import List

from pipelines.inference import predict_aqi, predict_multi_aqi
from db.mongo import feature_store

# -----------------------------------
# FastAPI App
# -----------------------------------
app = FastAPI(title="AQI Prediction API")

# -----------------------------------
# SINGLE AQI PREDICTION
# -----------------------------------
@app.get("/predict")
def predict(city: str = "Karachi", horizon: int = 24):
    """
    Predict AQI for a single horizon (hours)
    """
    result = predict_aqi(horizon)

    if result["status"] != "success":
        return {
            "status": "error",
            "message": result["message"]
        }

    return {
        "status": "ok",
        "city": city,
        "horizon_hours": horizon,
        "predicted_aqi": result["predicted_aqi"],
        "model": result["model"],
        "timestamp": datetime.utcnow().isoformat()
    }


# -----------------------------------
# MULTI-HORIZON AQI PREDICTION
# -----------------------------------
@app.get("/predict/multi")
def predict_multi(
    city: str = "Karachi",
    horizons: List[int] = [1, 6, 24]
):
    """
    Example:
    /predict/multi?horizons=1&horizons=6&horizons=24
    """
    result = predict_multi_aqi(horizons)

    if result["status"] != "success":
        return {
            "status": "error",
            "message": "Multi-horizon prediction failed"
        }

    return {
        **result,
        "timestamp": datetime.utcnow().isoformat()
    }


# -----------------------------------
# FEATURE FRESHNESS API
# -----------------------------------
@app.get("/features/freshness")
def feature_freshness(city: str = "Karachi"):
    """
    Returns feature store freshness for a city
    """
    doc = feature_store.find_one(
        {"city": city},
        {"_id": 0, "city": 1, "updated_at": 1}
    )

    if not doc or "updated_at" not in doc:
        return {
            "status": "no_data",
            "message": f"No feature timestamp found for city={city}"
        }

    updated_at = doc["updated_at"]

    # Ensure timezone safety
    if updated_at.tzinfo is None:
        updated_at = updated_at.replace(tzinfo=timezone.utc)

    now = datetime.now(timezone.utc)
    age_minutes = round((now - updated_at).total_seconds() / 60, 2)

    return {
        "status": "ok",
        "city": city,
        "updated_at": updated_at.isoformat(),
        "age_minutes": age_minutes
    }