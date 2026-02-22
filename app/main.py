from fastapi import FastAPI
from fastapi.responses import JSONResponse
import threading
import traceback

app = FastAPI(title="Karachi AQI Backend")

# ------------------------------------------------
# Root
# ------------------------------------------------
@app.get("/")
def root():
    return {"message": "AQI Backend Running"}

# ------------------------------------------------
# Forecast Endpoint
# ------------------------------------------------
@app.get("/forecast")
def forecast():
    try:
        from app.pipelines.inference_multi import predict_next_3_days
        results = predict_next_3_days()
        return results
    except Exception as e:
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )

# ------------------------------------------------
# Background Training Runner
# ------------------------------------------------
def background_training(horizon: int):
    try:
        from app.pipelines.training_pipeline import run_training
        run_training(horizon)
        print(f"✅ Training completed for horizon {horizon}")
    except Exception as e:
        print("❌ Training failed")
        traceback.print_exc()

# ------------------------------------------------
# Train Endpoint (NON-BLOCKING)
# ------------------------------------------------
@app.get("/train/{horizon}")
def train_model(horizon: int):
    thread = threading.Thread(
        target=background_training,
        args=(horizon,)
    )
    thread.start()

    return {
        "status": "training started",
        "horizon": horizon
    }