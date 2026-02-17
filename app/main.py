from fastapi import FastAPI, HTTPException
from datetime import datetime, timedelta
from pathlib import Path
import numpy as np
import joblib
import os

from app.db.mongo import get_db, get_model_registry

app = FastAPI(title="AQI Forecast API")


# ---------------------------------------------------
# Health Check
# ---------------------------------------------------
@app.get("/")
def health_check():
    return {"status": "success", "message": "AQI API running"}


# ---------------------------------------------------
# Get Production Model
# ---------------------------------------------------
def load_production_model(horizon: int):

    registry = get_model_registry()

    model_doc = registry.find_one({
        "horizon": horizon,
        "is_best": True
    })

    if not model_doc:
        raise HTTPException(status_code=404, detail="No production model found")

    model_path = model_doc["model_path"]

    if not Path(model_path).exists():
        raise HTTPException(status_code=500, detail="Model file not found")

    model = joblib.load(model_path)

    return model, model_doc["features"]


# ---------------------------------------------------
# Get Latest Feature Row from Mongo
# ---------------------------------------------------
def get_latest_features(feature_columns):

    db = get_db()
    collection = db["historical_hourly_data"]

    latest_doc = collection.find_one(
        {},
        sort=[("datetime", -1)]
    )

    if not latest_doc:
        raise HTTPException(status_code=404, detail="No data found")

    # Convert to feature vector
    X = []

    for col in feature_columns:
        X.append(latest_doc.get(col, 0))

    return np.array(X).reshape(1, -1)


# ---------------------------------------------------
# Multi Forecast Endpoint
# ---------------------------------------------------
@app.get("/forecast/multi")
def multi_forecast(horizon: int = 1):

    try:
        model, features = load_production_model(horizon)

        X_latest = get_latest_features(features)

        predictions = []

        base_time = datetime.utcnow()

        for day in range(1, horizon + 1):

            log_pred = model.predict(X_latest)[0]
            pred = float(np.expm1(log_pred))

            predictions.append({
                "datetime": base_time + timedelta(days=day),
                "predicted_aqi": pred
            })

        return {
            "status": "success",
            "horizon": horizon,
            "generated_at": datetime.utcnow(),
            "predictions": predictions
        }

    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }

# ---------------------------------------------------
# SHAP Endpoint
# ---------------------------------------------------
@app.get("/forecast/shap")
def shap_explain(horizon: int = 1):

    try:
        import shap

        model, features = load_production_model(horizon)
        X_latest = get_latest_features(features)

        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_latest)

        contributions = []

        for i, feature in enumerate(features):
            contributions.append({
                "feature": feature,
                "shap_value": float(shap_values[0][i])
            })

        log_pred = model.predict(X_latest)[0]
        prediction = float(np.expm1(log_pred))

        return {
            "status": "success",
            "model_name": "production",
            "prediction": prediction,
            "generated_at": datetime.utcnow(),
            "contributions": contributions
        }

    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }


# ---------------------------------------------------
# Model Metrics
# ---------------------------------------------------
@app.get("/models/metrics")
def get_model_metrics():

    registry = get_model_registry()

    models = list(registry.find(
        {"status": "production"},
        {"_id": 0}
    ))

    return {
        "status": "success",
        "models": models
    }
@app.get("/models/metrics")
def get_model_metrics():
    registry = get_model_registry()
    docs = list(registry.find({}, {"_id": 0}))

    if not docs:
        return {"status": "error", "detail": "No models found"}

    return {"status": "success", "models": docs}
@app.get("/models/best")
def get_best_model():
    registry = get_model_registry()
    doc = registry.find_one({"is_best": True}, {"_id": 0})

    if not doc:
        return {"status": "error", "detail": "No best model"}

    return {"status": "success", "model": doc}
@app.get("/features/importance")
def feature_importance(horizon: int = 1):
    model, features = load_production_model(horizon)

    if not hasattr(model, "feature_importances_"):
        return {"status": "error", "detail": "Model has no importance"}

    importance = model.feature_importances_

    data = [
        {"feature": f, "importance": float(i)}
        for f, i in zip(features, importance)
    ]

    return {"status": "success", "features": data}
import shap
import numpy as np

@app.get("/forecast/shap")
def shap_explain(horizon: int = 1):
    model, features = load_production_model(horizon)

    X_latest = get_latest_features(features)  # you already use this in forecast

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_latest)

    contributions = [
        {"feature": f, "shap_value": float(v)}
        for f, v in zip(features, shap_values[0])
    ]

    pred = float(model.predict(X_latest)[0])

    return {
        "status": "success",
        "prediction": pred,
        "contributions": contributions
    }
