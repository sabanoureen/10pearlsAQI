from fastapi import FastAPI, Query
from fastapi.responses import JSONResponse
from datetime import datetime

from app.pipelines.predict_3day_forecast import predict_next_3_days


app = FastAPI(
    title="AQI Forecast API",
    description="Multi-horizon AQI forecasting service",
    version="1.0"
)


@app.get("/")
def health_check():
    return {
        "status": "ok",
        "message": "AQI Forecast API running",
        "timestamp": datetime.utcnow()
    }


@app.get("/forecast")
def forecast(
    horizon: int = Query(3, description="Forecast horizon in days (1,3,7)")
):
    try:
        df = predict_next_3_days(horizon=horizon)

        # 🔥 Convert datetime to string
        df["datetime"] = df["datetime"].astype(str)

        results = df.to_dict("records")

        return JSONResponse(content={
            "horizon_days": horizon,
            "generated_at": datetime.utcnow().isoformat(),
            "predictions": results
        })

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )
