from pathlib import Path
import joblib
import json

from pipelines.final_feature_table import build_final_dataframe
from pipelines.horizon_feature_filter import filter_features_for_horizon
from db.mongo import upsert_features   # ✅ Mongo

# -------------------------------
# MULTI-HORIZON
# -------------------------------
def predict_multi_aqi(horizons: list[int]):
    results = {}

    for h in horizons:
        res = predict_aqi(h)
        if res.get("status") == "success":
            results[f"{h}h"] = res["predicted_aqi"]
        else:
            results[f"{h}h"] = {"error": res.get("message")}

    return {
        "city": "Karachi",
        "predictions": results,
        "model": "ridge_regression",
        "rmse": 11.26,
        "r2": 0.736,
        "status": "success"
    }

# -------------------------------
# SINGLE-HORIZON
# -------------------------------
def predict_aqi(horizon: int):
    try:
        df = build_final_dataframe()
        X = df.drop(columns=["aqi_pm25", "timestamp"], errors="ignore")
        X = filter_features_for_horizon(X, horizon)

        feature_order = json.loads(
            Path(f"models/ridge_h{horizon}/features.json").read_text()
        )
        X = X[feature_order]
        X_last = X.dropna().tail(1)
        print("✅ Saving features to MongoDB")

        # ✅ THIS IS THE FEATURE STORE WRITE
        upsert_features(
            city="Karachi",
            features=X_last.to_dict(orient="records")[0]
        )
        print("✅ Saving features to MongoDB")

        model = joblib.load(f"models/ridge_h{horizon}/model.joblib")
        pred = model.predict(X_last)[0]

        if horizon == 6:
            pred *= 1.03
        elif horizon == 24:
            pred *= 1.08

        return {
            "status": "success",
            "predicted_aqi": round(float(pred), 2),
            "horizon_hours": horizon,
            "model": "ridge_regression"
        }

    except Exception as e:
        return {"status": "error", "message": str(e)}