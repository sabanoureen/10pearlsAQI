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
@app.get("/forecast/generate")
def generate_forecast(horizon: int = 3):

    df = predict_next_3_days(horizon=horizon)

    predictions = df.to_dict(orient="records")

    db = get_db()

    db["daily_forecast"].insert_one({
        "horizon": horizon,
        "generated_at": datetime.utcnow(),
        "predictions": predictions
    })

    return {
        "status": "success",
        "horizon": horizon,
        "generated_at": datetime.utcnow(),
        "count": len(predictions)
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
