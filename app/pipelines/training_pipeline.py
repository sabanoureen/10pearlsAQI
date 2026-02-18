from datetime import datetime
import io
import joblib

from app.db.mongo import get_model_registry, get_fs
from app.pipelines.training_dataset import build_training_dataset
from app.pipelines.train_random_forest import train_random_forest
from app.pipelines.train_xgboost import train_xgboost
from app.pipelines.train_gradient_boosting import train_gradient_boosting
from app.pipelines.train_ensemble import train_ensemble
from app.pipelines.select_best_model import select_best_model


def run_training_pipeline(horizon: int = 1):

    print("\n" + "=" * 60)
    run_id = datetime.utcnow().strftime("%Y%m%d_%H%M%S")

    print(f"🆔 Run ID: {run_id}")
    print(f"📌 Horizon: {horizon}")
    print("=" * 60)

    # 1️⃣ Build Dataset
    X, y = build_training_dataset(horizon)

    if X.empty or y.empty:
        raise RuntimeError("Training dataset is empty")

    # 2️⃣ Split
    split_idx = int(len(X) * 0.8)
    X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx:]

    # 3️⃣ Train Models
    rf_model, rf_metrics = train_random_forest(X_train, y_train, X_val, y_val, horizon, run_id)
    xgb_model, xgb_metrics = train_xgboost(X_train, y_train, X_val, y_val, horizon, run_id)
    gb_model, gb_metrics = train_gradient_boosting(X_train, y_train, X_val, y_val, horizon, run_id)
    ensemble_model, ensemble_metrics = train_ensemble(
        rf_model, xgb_model, gb_model,
        X_train, y_train, X_val, y_val,
        horizon, run_id
    )

    # 4️⃣ Select Best Model
    best_model_info = select_best_model(horizon)
    best_model = best_model_info["model"]
    best_model_name = best_model_info["model_name"]

    registry = get_model_registry()
    fs = get_fs()

    # 5️⃣ Compare With Existing Production
    existing_model = registry.find_one({
        "horizon": horizon,
        "is_best": True
    })

    if existing_model:
        if best_model_info["rmse"] >= existing_model["rmse"]:
            print("⚠ New model not better. Keeping existing production model.")
            return existing_model

    # 6️⃣ Deactivate Old Production
    registry.update_many(
        {"horizon": horizon},
        {"$set": {"is_best": False}}
    )

    # 7️⃣ Save Model to GridFS
    buffer = io.BytesIO()
    joblib.dump(best_model, buffer)
    buffer.seek(0)

    file_id = fs.put(
        buffer.read(),
        filename=f"{best_model_name}_h{horizon}.joblib",
        metadata={
            "run_id": run_id,
            "horizon": horizon,
            "model_name": best_model_name,
            "created_at": datetime.utcnow()
        }
    )

    # 8️⃣ Register Model Version
    registry.insert_one({
        "model_name": best_model_name,
        "horizon": horizon,
        "rmse": best_model_info["rmse"],
        "mae": best_model_info["mae"],
        "run_id": run_id,
        "gridfs_id": file_id,
        "features": X.columns.tolist(),
        "created_at": datetime.utcnow(),
        "is_best": True,
        "status": "production"
    })

    print("✅ Production model updated")
    print("=" * 60)

    return best_model_info


import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--horizon", type=int, required=True)
    args = parser.parse_args()

    run_training_pipeline(horizon=args.horizon)
