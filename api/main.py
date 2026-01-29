from fastapi import FastAPI
from pipelines.inference import predict_aqi, predict_multi_aqi

app = FastAPI(title="10Pearls AQI API")

# -------------------------------------------------
# ROOT
# -------------------------------------------------
@app.get("/")
def root():
    return {
        "service": "10Pearls AQI API",
        "status": "running",
        "endpoints": [
            "/health",
            "/predict?horizon=1",
            "/predict/multi",
            "/models"
        ]
    }

# -------------------------------------------------
# HEALTH
# -------------------------------------------------
@app.get("/health")
def health():
    return {"status": "ok"}

# -------------------------------------------------
# SINGLE HORIZON
# -------------------------------------------------
@app.get("/predict")
def predict(horizon: int = 1):
    result = predict_aqi(horizon)

    if result.get("status") == "error":
        return result

    return {
        "predicted_aqi": result["predicted_aqi"],
        "horizon_hours": result["horizon_hours"],
        "model": result["model"],
        "rmse": 11.26,
        "r2": 0.736,
        "status": "success"
    }

# -------------------------------------------------
# MULTI HORIZON
# -------------------------------------------------
@app.get("/predict/multi")
def predict_multi():
    return predict_multi_aqi([1, 6, 24])

# -------------------------------------------------
# MODEL REGISTRY
# -------------------------------------------------
@app.get("/models")
def list_models():
    from db.mongo import model_registry
    return list(model_registry.find({}, {"_id": 0}))
@app.get("/features/freshness")
def feature_freshness():
    from db.mongo import get_feature_freshness
    return get_feature_freshness("Karachi") or {}