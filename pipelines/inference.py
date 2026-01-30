from pathlib import Path
from typing import List, Optional
import joblib

from pipelines.final_feature_table import build_final_dataframe
from pipelines.horizon_feature_filter import filter_features_for_horizon

from db.mongo import model_registry, upsert_features


# -------------------------------------------------
# INTERNAL: Load model from MongoDB registry
# -------------------------------------------------
def _load_model_from_registry(horizon: int, version: Optional[str] = None):
    """
    Load model metadata + artifact from MongoDB registry.
    If version is None -> load is_best model.
    """

    query = {"horizon": horizon}

    if version:
        query["version"] = version
    else:
        query["is_best"] = True

    model_doc = model_registry.find_one(query)

    if not model_doc:
        raise ValueError(f"No model found for horizon={horizon}, version={version}")

    model_path = Path(model_doc["model_path"])

    if not model_path.exists():
        raise FileNotFoundError(f"Model file missing: {model_path}")

    model = joblib.load(model_path)

    return model, model_doc


# -------------------------------------------------
# SINGLE-HORIZON AQI PREDICTION
# -------------------------------------------------
def predict_aqi(horizon: int, version: Optional[str] = None):
    """
    Predict AQI for a single horizon (hours).
    If version is None -> uses best model.
    """

    try:
        # 1️⃣ Build final feature table
        df = build_final_dataframe()

        if df.empty:
            raise ValueError("Final feature dataframe is empty")

        # Drop target columns if present
        X = df.drop(columns=["aqi_pm25", "timestamp"], errors="ignore")

        # 2️⃣ Horizon-specific feature filtering
        X = filter_features_for_horizon(X, horizon)

        # 3️⃣ Select latest valid row
        X_last = X.dropna().tail(1)

        if X_last.empty:
            raise ValueError("No valid feature row available for prediction")

        # 4️⃣ Persist features to feature store
        upsert_features(
            city="Karachi",
            features=X_last.to_dict(orient="records")[0]
        )

        # 5️⃣ Load model from registry (best or pinned)
        model, meta = _load_model_from_registry(horizon, version)

        # 6️⃣ Enforce feature order
        feature_order = meta["features"]
        X_last = X_last[feature_order]

        # 7️⃣ Predict
        pred = float(model.predict(X_last)[0])

        # 8️⃣ Optional horizon calibration (keep if you want)
        if horizon == 6:
            pred *= 1.03
        elif horizon == 24:
            pred *= 1.08

        # 9️⃣ Success response
        return {
            "status": "success",
            "predicted_aqi": round(pred, 2),
            "horizon_hours": horizon,
            "model": {
                "name": meta["model_name"],
                "version": meta["version"],
                "rmse": meta["rmse"],
                "r2": meta.get("r2")
            }
        }

    except Exception as e:
        return {
            "status": "error",
            "message": str(e),
            "horizon_hours": horizon
        }


# -------------------------------------------------
# MULTI-HORIZON AQI PREDICTION
# -------------------------------------------------
def predict_multi_aqi(horizons: List[int], version: Optional[str] = None):
    """
    Predict AQI for multiple horizons.
    Example: horizons=[1, 6, 24]
    """

    predictions = {}

    for h in horizons:
        result = predict_aqi(h, version)

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
        "model_version": version or "latest"
    }
