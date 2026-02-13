from fastapi import FastAPI, HTTPException
from datetime import datetime
from fastapi.encoders import jsonable_encoder

from app.db.mongo import get_db
from app.pipelines.predict_multi_day import generate_multi_day_forecast
from app.pipelines.shap_analysis import generate_shap_analysis

app = FastAPI(
    title="AQI Forecast API",
    version="3.2"
)

@app.get("/")
def health():
    return {
        "status": "ok",
        "timestamp": datetime.utcnow()
    }

@app.get("/forecast/multi")
def multi_forecast(days: int = 3):
    try:
        result = generate_multi_day_forecast(days)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/forecast/shap")
def shap_endpoint():
    try:
        return generate_shap_analysis()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/models/metrics")
def get_model_metrics():
    from app.db.mongo import get_model_registry

    collection = get_model_registry()
    models = list(collection.find({"status": "registered"}))

    formatted = []

    for m in models:
        formatted.append({
            "model_name": m["model_name"],
            "rmse": m["rmse"],
            "r2": m["r2"]
        })

    return {
        "status": "success",
        "models": formatted
    }
