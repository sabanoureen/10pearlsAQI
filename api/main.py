from fastapi import FastAPI
from pipelines.inference import predict_aqi, predict_multi_aqi

app = FastAPI()

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

@app.get("/predict/multi")
def predict_multi():
    return predict_multi_aqi([1, 6, 24])

@app.get("/health")
def health():
    return {"status": "ok"}