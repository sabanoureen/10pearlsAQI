from pathlib import Path
import json
import joblib
from typing import List

from app.db.mongo import model_registry, upsert_features
from app.pipelines.final_feature_table import build_final_dataframe
from app.pipelines.horizon_feature_filter import filter_features_for_horizon


# -------------------------------------------------
# Load production model from MongoDB
# -------------------------------------------------
def _load_production_model(horizon: int):
    model_doc = model_registry.find_one(
        {"horizon": horizon, "is_best": True}
    )

    if not model_doc:
        raise RuntimeError(f"No production model found for horizon={horizon}")

    model_path = Path(model_doc["model_path"])
    features = model_doc["features"]

    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    model = joblib.load(model_path)

    return model, features, model_doc


# -------------------------------------------------
# Single-horizon prediction
# -------------------------------------------------
def predict_aqi(horizon: int):
    try:
        # 1️⃣ Load production model
        model, feature_order, model_doc = _load_production_model(horizon)

        # 2️⃣ Build features
        df = build_final_dataframe()
        if df.empty:
            raise RuntimeError("Final feature dataframe is empty")

        X = df.drop(columns=["aqi_pm25", "timestamp"], errors="ignore")
        X = filter_features_for_horizon(X, horizon)
        X = X[feature_order]

        X_last = X.dropna().tail(1)
        if X_last.empty:
            raise RuntimeError("No valid feature row available")

        # 3️⃣ Save to feature store
        upsert_features(
            city="Karachi",
            features=X_last.to_dict(orient="records")[0]
        )

        # 4️⃣ Predict
        pred = float(model.predict(X_last)[0])

        return {
            "status": "success",
            "predicted_aqi": round(pred, 2),
            "horizon_hours": horizon,
            "model_name": model_doc["model_name"],
            "version": model_doc.get("version", "legacy"),   # ✅ NOW CORRECT
        }

    except Exception as e:
        return {
            "status": "error",
            "message": str(e),
            "horizon_hours": horizon
        }


# -------------------------------------------------
# Multi-horizon prediction
# -------------------------------------------------
def predict_multi_aqi(horizons: List[int]):
    predictions = {}

    for h in horizons:
        result = predict_aqi(h)
        if result["status"] == "success":
            predictions[f"{h}h"] = result["predicted_aqi"]
        else:
            predictions[f"{h}h"] = {"error": result["message"]}

    return {
        "status": "success",
        "city": "Karachi",
        "predictions": predictions,
    }
