from pathlib import Path
import joblib
from typing import List

from app.db.mongo import get_model_registry, upsert_features
from app.pipelines.final_feature_table import build_final_dataframe
from app.pipelines.horizon_feature_filter import filter_features_for_horizon


# -------------------------------------------------
# Load production model
# -------------------------------------------------
def load_best_model(horizon: int):
    registry = get_model_registry()

    model_doc = registry.find_one(
        {"horizon": horizon, "is_best": True}
    )

    if not model_doc:
        raise RuntimeError(f"No production model found for horizon={horizon}")

    model_path = Path(model_doc["model_path"])

    if not model_path.exists():
        raise RuntimeError(f"Model file not found: {model_path}")

    model = joblib.load(model_path)

    return model, model_doc


# -------------------------------------------------
# Single-horizon prediction
# -------------------------------------------------
def predict_aqi(horizon: int, city: str = "Karachi"):
    try:
        model, model_doc = load_best_model(horizon)

        # 1️⃣ Build features
        df = build_final_dataframe(city=city)

        if df.empty:
            raise RuntimeError("Final feature dataframe is empty")

        X = df.drop(columns=["aqi_pm25", "timestamp"], errors="ignore")

        # Filter horizon-specific features
        X = filter_features_for_horizon(X, horizon)

        # Ensure exact feature order
        X = X[model_doc["features"]]

        X_last = X.dropna().tail(1)

        if X_last.empty:
            raise RuntimeError("No valid feature row available")

        # 2️⃣ Log features (MLOps tracking)
        upsert_features(
            city=city,
            features=X_last.to_dict(orient="records")[0],
        )

        # 3️⃣ Predict
        pred_log = model.predict(X_last)[0]
        pred = float(np.expm1(pred_log))

# 🔥 Convert back to real PM2.5 scale
        pred = float(np.expm1(log_pred))
        return {
            "status": "success",
            "city": city,
            "predicted_aqi": round(pred, 2),
            "horizon": horizon,
            "model_name": model_doc["model_name"],
            "version": model_doc.get("version"),
        }

    except Exception as e:
        return {
            "status": "error",
            "message": str(e),
            "horizon": horizon,
            "city": city,
        }


# -------------------------------------------------
# Multi-horizon prediction
# -------------------------------------------------
def predict_multi_aqi(horizons: List[int], city: str = "Karachi"):

    predictions = {}

    for h in horizons:
        predictions[f"{h}h"] = predict_aqi(horizon=h, city=city)

    return {
        "status": "success",
        "city": city,
        "predictions": predictions,
    }
