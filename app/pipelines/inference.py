from pathlib import Path
import joblib
from typing import List

from app.db.mongo import get_model_registry, upsert_features
from app.pipelines.final_feature_table import build_final_dataframe
from app.pipelines.horizon_feature_filter import filter_features_for_horizon


def _load_production_model(horizon: int):
    registry = get_model_registry()

    model_doc = registry.find_one(
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


def predict_aqi(horizon: int):
    model, feature_order, model_doc = _load_production_model(horizon)

    df = build_final_dataframe()
    if df.empty:
        raise RuntimeError("Final feature dataframe is empty")

    X = df.drop(columns=["aqi_pm25", "timestamp"], errors="ignore")
    X = filter_features_for_horizon(X, horizon)
    X = X[feature_order]

    X_last = X.dropna().tail(1)
    if X_last.empty:
        raise RuntimeError("No valid feature row available")

    upsert_features(
        city="Karachi",
        features=X_last.to_dict(orient="records")[0]
    )

    pred = float(model.predict(X_last)[0])

    return {
        "status": "success",
        "predicted_aqi": round(pred, 2),
        "horizon": horizon,
        "model_name": model_doc["model_name"],
        "version": model_doc.get("version"),
    }


def predict_multi_aqi(horizons: List[int]):
    results = {}
    for h in horizons:
        try:
            res = predict_aqi(h)
            results[f"{h}h"] = res["predicted_aqi"]
        except Exception as e:
            results[f"{h}h"] = {"error": str(e)}

    return {
        "status": "success",
        "city": "Karachi",
        "predictions": results,
    }
