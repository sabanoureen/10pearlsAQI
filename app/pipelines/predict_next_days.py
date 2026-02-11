"""
Production Forecast Generator
Generates N-day forecast using best production model
"""

import joblib
import pandas as pd
from datetime import datetime, timedelta

from app.db.mongo import get_db
from app.pipelines.select_best_model import select_best_model


def generate_forecast(horizon: int = 1):

    db = get_db()

    # ----------------------------------
    # 1️⃣ Get best model
    # ----------------------------------
    best_model = select_best_model(horizon)

    if not best_model:
        raise RuntimeError("No production model found")

    model_path = best_model["model_path"]

    model = joblib.load(model_path)

    # ----------------------------------
    # 2️⃣ Load latest features
    # ----------------------------------
    feature_store = db["feature_store"]

    latest_doc = (
        feature_store
        .find()
        .sort("datetime", -1)
        .limit(1)
    )

    latest_doc = list(latest_doc)

    if not latest_doc:
        raise RuntimeError("No feature data found")

    latest_doc = latest_doc[0]

    feature_columns = best_model["features"]

    X = pd.DataFrame([{
        col: latest_doc.get(col)
        for col in feature_columns
    }])

    # ----------------------------------
    # 3️⃣ Predict
    # ----------------------------------
    prediction = model.predict(X)[0]

    # ----------------------------------
    # 4️⃣ Save forecast
    # ----------------------------------
    forecast_collection = db["forecast_results"]

    forecast_doc = {
        "horizon": horizon,
        "generated_at": datetime.utcnow(),
        "predictions": [{
            "datetime": datetime.utcnow() + timedelta(days=horizon),
            "predicted_aqi": float(prediction)
        }]
    }

    forecast_collection.insert_one(forecast_doc)

    return forecast_doc
