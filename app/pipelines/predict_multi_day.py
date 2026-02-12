"""
Multi-Day Rolling Forecast
Generates recursive N-day forecast using production model
"""

import joblib
import pandas as pd
from datetime import datetime, timedelta

from app.db.mongo import get_db
from app.pipelines.load_production_model import load_production_model


def generate_multi_day_forecast(horizon: int = 3):

    db = get_db()

    # -------------------------------------------------
    # 1️⃣ Load production model
    # -------------------------------------------------
    model, features = load_production_model(horizon=1)

    # -------------------------------------------------
    # 2️⃣ Get latest feature row
    # -------------------------------------------------
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

    current_features = {
        col: latest_doc.get(col)
        for col in features
    }

    predictions = []

    # -------------------------------------------------
    # 3️⃣ Rolling prediction loop
    # -------------------------------------------------
        # -------------------------------------------------
    # 3️⃣ Rolling prediction loop
    # -------------------------------------------------
    for step in range(1, horizon + 1):

        X = pd.DataFrame([current_features])
        pred = float(model.predict(X)[0])

        future_datetime = datetime.utcnow() + timedelta(days=step)

        predictions.append({
            "datetime": future_datetime,
            "predicted_aqi": pred
        })

        # 🔁 Shift lag features properly
        lag_cols = [col for col in features if col.startswith("lag_")]

        # Sort lag columns descending (lag_3 → lag_2 → lag_1)
        lag_cols_sorted = sorted(
            lag_cols,
            key=lambda x: int(x.split("_")[1]),
            reverse=True
        )

        for i in range(len(lag_cols_sorted) - 1):
            current_features[lag_cols_sorted[i]] = current_features[lag_cols_sorted[i + 1]]

        if "lag_1" in current_features:
            current_features["lag_1"] = pred

    # -------------------------------------------------
    # 4️⃣ Save forecast to Mongo
    # -------------------------------------------------
    forecast_doc = {
        "horizon": horizon,
        "generated_at": datetime.utcnow(),
        "predictions": predictions
    }

    db["daily_forecast"].insert_one(forecast_doc)

    return forecast_doc
