from fastapi import FastAPI
from datetime import datetime
from fastapi.encoders import jsonable_encoder

from app.pipelines.predict_3day_forecast import predict_next_3_days
from app.db.mongo import get_db

app = FastAPI(
    title="AQI Forecast API",
    description="Multi-day AQI forecasting service",
    version="2.0"
)


# -----------------------------------------
# HEALTH CHECK
# -----------------------------------------
@app.get("/")
def health_check():
    return {
        "status": "ok",
        "message": "AQI Forecast API running",
        "timestamp": datetime.utcnow()
    }


# -----------------------------------------
# GENERATE + STORE FORECAST
# -----------------------------------------
from app.pipelines.predict_next_days import generate_forecast

@app.get("/forecast/generate")
def generate(horizon: int = 1):

    try:
        result = generate_forecast(horizon)
        return {
            "status": "success",
            "generated_at": result["generated_at"]
        }
    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }



# -----------------------------------------
# GET LATEST STORED FORECAST
# -----------------------------------------
@app.get("/forecast/latest")
def get_latest_forecast(horizon: int = 3):

    db = get_db()

    doc = db["daily_forecast"].find_one(
        {"horizon": horizon},
        sort=[("generated_at", -1)]
    )

    if not doc:
        return {"status": "error", "message": "No forecast found"}

    return jsonable_encoder({
        "status": "success",
        "horizon": doc["horizon"],
        "generated_at": doc["generated_at"],
        "predictions": doc["predictions"]
    })
