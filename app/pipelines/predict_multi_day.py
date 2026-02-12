"""
Multi-Day Rolling Forecast
Generates recursive N-day forecast using production model
"""

import pandas as pd
from datetime import datetime, timedelta

from app.db.mongo import get_db
from app.pipelines.load_production_model import load_production_model


def generate_multi_day_forecast(horizon: int = 3):

    db = get_db()

    # -------------------------------------------------
    # 1️⃣ Load production model (always horizon=1)
    # -------------------------------------------------
    model, features = load_production_model(horizon=1)

    print("====================================")
    print("MODEL FEATURES:", features)
    print("====================================")

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

    print("Initial Features Loaded:", current_features)

    predictions = []

    # -------------------------------------------------
    # 3️⃣ Rolling recursive prediction
    # -------------------------------------------------
    for step in range(1, horizon + 1):

        print(f"\n----- STEP {step} -----")
        print("Features BEFORE prediction:", current_features)

        # Predict
        X = pd.DataFrame([current_features])
        pred = float(model.predict(X)[0])

        print("Prediction:", pred)

        future_datetime = datetime.utcnow() + timedelta(days=step)

        predictions.append({
            "datetime": future_datetime,
            "predicted_aqi": pred
        })

        # -------------------------------------------------
        # 🔁 Shift lag features properly
        # -------------------------------------------------
        lag_cols = [col for col in features if "lag" in col]

        if lag_cols:
            # Sort by numeric lag (1h, 3h, 6h → 6h, 3h, 1h)
            lag_cols_sorted = sorted(
                lag_cols,
                key=lambda x: int(x.split("_")[-1].replace("h", "")),
                reverse=True
            )

            # Shift values
            for i in range(len(lag_cols_sorted) - 1):
                current_features[lag_cols_sorted[i]] = current_features[lag_cols_sorted[i + 1]]

            # Update smallest lag with prediction
            smallest_lag = sorted(
                lag_cols,
                key=lambda x: int(x.split("_")[-1].replace("h", ""))
            )[0]

            current_features[smallest_lag] = pred

        print("Features AFTER update:", current_features)

    print("\n====================================")
    print("FINAL PREDICTIONS:", predictions)
    print("====================================")

    # -------------------------------------------------
    # 4️⃣ Save forecast
    # -------------------------------------------------
    forecast_doc = {
        "horizon": horizon,
        "generated_at": datetime.utcnow(),
        "predictions": predictions
    }

    db["daily_forecast"].insert_one(forecast_doc)

    return forecast_doc
