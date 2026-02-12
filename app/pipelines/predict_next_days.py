"""
Production Forecast Generator
Generates forecast using PRODUCTION model only
"""

import joblib
import pandas as pd
from datetime import datetime, timedelta

from app.db.mongo import get_db
from app.pipelines.load_production_model import load_production_model


def generate_forecast(horizon: int = 1):

    db = get_db()

    # ----------------------------------
    # 1️⃣ Load PRODUCTION model
    # ----------------------------------
    model, feature_columns = load_production_model(horizon)

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
    forecast_collection = db["daily_forecast"]

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
