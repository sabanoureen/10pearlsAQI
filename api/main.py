from fastapi import FastAPI
from datetime import datetime, timezone
from db.mongo import feature_store

# ✅ App MUST be defined first
app = FastAPI()
from pipelines.inference import predict_multi_aqi

@app.get("/predict/multi")
def predict_multi(horizons: str = "1,6,24"):
    h_list = [int(h) for h in horizons.split(",")]

    return predict_multi_aqi(h_list)

# -----------------------------------
# Feature Freshness API
# -----------------------------------
@app.get("/features/freshness")
def feature_freshness(city: str = "Karachi"):
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
    from fastapi import FastAPI
from datetime import datetime
from pipelines.inference import predict_aqi

# app is already defined above
# app = FastAPI()

# -----------------------------------
# AQI PREDICTION (SINGLE)
# -----------------------------------
@app.get("/predict")
def predict(city: str = "Karachi", horizon: int = 24):
    """
    Simple AQI prediction endpoint
    """
    result = predict_aqi(horizon)

    if result.get("status") != "success":
        return {
            "status": "error",
            "message": result.get("message", "Prediction failed")
        }

    return {
        "status": "ok",
        "city": city,
        "horizon_hours": horizon,
        "predicted_aqi": result["predicted_aqi"],
        "model": result["model"],
        "timestamp": datetime.utcnow().isoformat()
    }

    # Ensure timezone-safe comparison
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