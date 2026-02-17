from fastapi import FastAPI, HTTPException
from pathlib import Path
import numpy as np
import joblib
from datetime import datetime

app = FastAPI(title="AQI Forecast API")


# ---------------------------------------------------
# Health Check
# ---------------------------------------------------
@app.get("/")
def health():
    return {"status": "ok", "message": "AQI API running"}


# ---------------------------------------------------
# Get latest model file for a horizon
# ---------------------------------------------------
def get_latest_model(horizon: int) -> Path:
    """
    Locate newest .joblib model inside:
    repo_root/models/rf_h{horizon}
    """

    # repo root (Railway safe)
    BASE_DIR = Path(__file__).resolve().parents[2]

    model_dir = BASE_DIR / f"models/rf_h{horizon}"

    if not model_dir.exists():
        raise HTTPException(
            status_code=500,
            detail=f"Model folder not found: {model_dir}"
        )

    model_files = list(model_dir.glob("*.joblib"))

    if not model_files:
        raise HTTPException(
            status_code=500,
            detail=f"No model files in {model_dir}"
        )

    # newest file
    latest_model = max(model_files, key=lambda f: f.stat().st_mtime)

    return latest_model


# ---------------------------------------------------
# Forecast Endpoint
# ---------------------------------------------------
@app.get("/forecast/multi")
def forecast_multi(horizon: int = 1):
    """
    Returns AQI forecast using newest model for given horizon.
    """

    model_path = get_latest_model(horizon)

    # load model
    model = joblib.load(model_path)

    # create dummy feature vector matching model input
    if hasattr(model, "n_features_in_"):
        n_features = model.n_features_in_
    else:
        # fallback
        n_features = 10

    X = np.zeros((1, n_features))

    # predict (model trained on log(AQI+1))
    log_pred = model.predict(X)[0]
    prediction = float(np.expm1(log_pred))

    return {
        "status": "success",
        "horizon": horizon,
        "prediction": prediction,
        "model_used": model_path.name,
        "generated_at": datetime.utcnow()
    }
