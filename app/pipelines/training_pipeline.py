from datetime import datetime
from app.db.mongo import get_model_registry
from app.pipelines.training_dataset import build_training_dataset
from app.pipelines.train_random_forest import train_random_forest
from app.pipelines.train_xgboost import train_xgboost
from app.pipelines.train_gradient_boosting import train_gradient_boosting
from app.pipelines.train_ensemble import train_ensemble
from app.pipelines.select_best_model import select_best_model
import joblib
import os



# ==========================================================
# MAIN TRAINING PIPELINE
# ==========================================================
def run_training_pipeline(horizon: int = 1):

    print("\n" + "=" * 60)

    run_id = datetime.utcnow().strftime("%Y%m%d_%H%M%S")

    print(f"🆔 Training run_id = {run_id}")
    print("🚀 Starting training pipeline")
    print(f"📌 Forecast horizon: {horizon} day(s)")
    print("=" * 60)

    # --------------------------------------------------
    # 1️⃣ Build Dataset
    # --------------------------------------------------
    X, y = build_training_dataset(horizon)

    if X.empty or y.empty:
        raise RuntimeError("Training dataset is empty")

    print(f"📊 Dataset size: {X.shape[0]} rows")

    # --------------------------------------------------
    # 2️⃣ Clean OLD candidate models
    # --------------------------------------------------
    registry = get_model_registry()

    delete_result = registry.delete_many({
    "horizon": horizon
})
    print("🧹 All old models deleted for this horizon")

    # --------------------------------------------------
    # 3️⃣ Train / Validation Split
    # --------------------------------------------------
    split_idx = int(len(X) * 0.8)

    X_train = X.iloc[:split_idx]
    X_val   = X.iloc[split_idx:]
    y_train = y.iloc[:split_idx]
    y_val   = y.iloc[split_idx:]

    print("🔀 Train/Validation split completed")

    # --------------------------------------------------
    # 4️⃣ Train Models
    # --------------------------------------------------
    rf_model, rf_metrics = train_random_forest(
        X_train, y_train, X_val, y_val, horizon, run_id
    )

    xgb_model, xgb_metrics = train_xgboost(
        X_train, y_train, X_val, y_val, horizon, run_id
    )

    gb_model, gb_metrics = train_gradient_boosting(
        X_train, y_train, X_val, y_val, horizon, run_id
    )

    ensemble_model, ensemble_metrics = train_ensemble(
        rf_model,
        xgb_model,
        gb_model,
        X_train,
        y_train,
        X_val,
        y_val,
        horizon,
        run_id
    )

    # --------------------------------------------------
    # 5️⃣ Select Best Model
    # --------------------------------------------------
    print("\n🏆 Selecting best model...")

    best_model_info = select_best_model(horizon)
        # --------------------------------------------------
    # 6️⃣ Export production model (for API)
    # --------------------------------------------------
    print("\n💾 Exporting production model for API...")

    best_model = best_model_info["model"]
    feature_list = X.columns.tolist()

    os.makedirs("models", exist_ok=True)

    model_path = f"models/rf_model_h{horizon}.joblib"
    features_path = f"models/features_h{horizon}.joblib"

    joblib.dump(best_model, model_path)
    joblib.dump(feature_list, features_path)

    print(f"✔ Model saved → {model_path}")
    print(f"✔ Features saved → {features_path}")


    print("\n🎯 Production Model Selected:")
    print(f"Model Name : {best_model_info['model_name']}")
    print(f"RMSE       : {best_model_info['rmse']}")
    print(f"MAE        : {best_model_info['mae']}")

    print("\n✅ Training pipeline completed successfully")
    print("=" * 60)

    return best_model_info


# ==========================================================
# Allow running directly
# ==========================================================
if __name__ == "__main__":
    run_training_pipeline(horizon=1)
