from fastapi import FastAPI, HTTPException
from datetime import datetime
from fastapi.encoders import jsonable_encoder

from app.db.mongo import get_db
from app.pipelines.predict_multi_day import generate_multi_day_forecast
from app.pipelines.shap_analysis import generate_shap_analysis

app = FastAPI(
    title="AQI Forecast API",
    description="Multi-day AQI forecasting service",
    version="4.0"
)


# ==============================
# ROOT HEALTH
# ==============================
@app.get("/")
def health_check():
    return {
        "status": "ok",
        "message": "AQI Forecast API running",
        "timestamp": datetime.utcnow().isoformat()
    }


# ==============================
# MULTI-DAY FORECAST
# ==============================
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


# ==============================
# LATEST FORECAST
# ==============================
@app.get("/forecast/latest")
def get_latest_forecast(horizon: int = 3):

    db = get_db()

    doc = db["daily_forecast"].find_one(
        {"horizon": horizon},
        sort=[("generated_at", -1)]
    )

    if not doc:
        raise HTTPException(status_code=404, detail="No forecast found")

    return {
        "status": "success",
        "horizon": int(doc["horizon"]),
        "generated_at": str(doc["generated_at"]),
        "model_version": doc.get("model_version", "unknown"),
        "predictions": doc["predictions"]
    }


# ==============================
# SHAP ANALYSIS
# ==============================

@app.get("/forecast/shap")
def shap_endpoint():
    try:
        return generate_shap_analysis()
    except Exception as e:
        return {"status": "error", "message": str(e)}



# ==============================
# MODEL METRICS
# ==============================
@app.get("/models/metrics")
def get_model_metrics():
    from app.db.mongo import get_model_registry
    from fastapi.encoders import jsonable_encoder

    collection = get_model_registry()

    # Fetch production models
    models = list(collection.find({"status": "production"}))

    formatted = []
    for m in models:
        formatted.append({
            "model_name": m.get("model_name", "unknown"),
            "rmse": float(m.get("rmse", 0)),
            "r2": float(m.get("r2", 0)),
            "horizon": m.get("horizon", 1)
        })

    return jsonable_encoder({
        "status": "success",
        "models": formatted
    })
