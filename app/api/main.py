from fastapi import FastAPI, HTTPException
from datetime import datetime
from fastapi.encoders import jsonable_encoder

from app.db.mongo import get_db
from app.pipelines.predict_multi_day import generate_multi_day_forecast

app = FastAPI(
    title="AQI Forecast API",
    description="Multi-day AQI forecasting service",
    version="3.1"
)


@app.get("/")
def health_check():
    return {
        "status": "ok",
        "message": "AQI Forecast API running",
        "timestamp": datetime.utcnow()
    }


@app.get("/forecast/multi")
def multi_forecast(days: int = 3):

    try:
        result = generate_multi_day_forecast(days)

        return {
            "status": "success",
            "horizon": days,
            "generated_at": result["generated_at"],
            "model_version": result["model_version"],
            "predictions": result["predictions"]
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/forecast/latest")
def get_latest_forecast(horizon: int = 3):

    db = get_db()

    doc = db["daily_forecast"].find_one(
        {"horizon": horizon},
        sort=[("generated_at", -1)]
    )

    if not doc:
        raise HTTPException(status_code=404, detail="No forecast found")

    return jsonable_encoder({
        "status": "success",
        "horizon": doc["horizon"],
        "generated_at": doc["generated_at"],
        "model_version": doc.get("model_version", "unknown"),
        "predictions": doc["predictions"]
    })


