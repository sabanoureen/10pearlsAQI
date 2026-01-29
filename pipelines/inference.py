from pathlib import Path
import joblib
import json
from typing import List

from pipelines.final_feature_table import build_final_dataframe
from pipelines.horizon_feature_filter import filter_features_for_horizon
from db.mongo import upsert_features

# -------------------------------
# MULTI-HORIZON PREDICTION
# -------------------------------
def predict_multi_aqi(horizons: List[int]):
    """
    Predict AQI for multiple horizons
    Example: [1, 6, 24]
    """
    predictions = {}

    for h in horizons:
        result = predict_aqi(h)

        if result["status"] == "success":
            predictions[f"{h}h"] = result["predicted_aqi"]
        else:
            predictions[f"{h}h"] = {
                "error": result.get("message", "Prediction failed")
            }

    return {
        "status": "success",
        "city": "Karachi",
        "predictions": predictions,
        "model": "ridge_regression",
        "rmse": 11.26,
        "r2": 0.736
    }


# -------------------------------
# SINGLE-HORIZON PREDICTION
# -------------------------------
def predict_aqi(horizon: int):
    """
    Predict AQI for a single horizon (in hours)
    """
    try:
        # 1️⃣ Build features
        df = build_final_dataframe()

        if df.empty:
            raise ValueError("Final feature dataframe is empty")

        X = df.drop(columns=["aqi_pm25", "timestamp"], errors="ignore")

        # 2️⃣ Filter features for horizon
        X = filter_features_for_horizon(X, horizon)

        # 3️⃣ Load expected feature order
        feature_path = Path(f"models/ridge_h{horizon}/features.json")
        model_path = Path(f"models/ridge_h{horizon}/model.joblib")

        if not feature_path.exists():
            raise FileNotFoundError(f"Missing features.json for horizon={horizon}")

        if not model_path.exists():
            raise FileNotFoundError(f"Missing model.joblib for horizon={horizon}")

        feature_order = json.loads(feature_path.read_text())

        X = X[feature_order]

        # 4️⃣ Select latest row
        X_last = X.dropna().tail(1)

        if X_last.empty:
            raise ValueError("No valid feature row available for prediction")

        # 5️⃣ 🔥 WRITE TO FEATURE STORE
        upsert_features(
            city="Karachi",
            features=X_last.to_dict(orient="records")[0]
        )

        # 6️⃣ Load model and predict
        model = joblib.load(model_path)
        pred = float(model.predict(X_last)[0])

        # 7️⃣ Horizon calibration
        if horizon == 6:
            pred *= 1.03
        elif horizon == 24:
            pred *= 1.08

        # 8️⃣ SUCCESS RESPONSE (IMPORTANT)
        return {
            "status": "success",
            "predicted_aqi": round(pred, 2),
            "horizon_hours": horizon,
            "model": "ridge_regression"
        }

    except Exception as e:
        # ❌ ERROR RESPONSE (ALWAYS RETURN JSON)
        return {
            "status": "error",
            "message": str(e),
            "horizon_hours": horizon
        }