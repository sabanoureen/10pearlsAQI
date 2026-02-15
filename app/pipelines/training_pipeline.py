from datetime import datetime
from app.db.mongo import get_model_registry
from app.pipelines.training_dataset import build_training_dataset

from app.pipelines.train_random_forest import train_random_forest
from app.pipelines.train_xgboost import train_xgboost
from app.pipelines.train_gradient_boosting import train_gradient_boosting
from app.pipelines.train_ensemble import train_ensemble
from app.pipelines.select_best_model import select_best_model


def run_training_pipeline(horizon: int = 1):

    try:
        # --------------------------------------------------
        # 1️⃣ Create unique run_id
        # --------------------------------------------------
        run_id = datetime.utcnow().strftime("%Y%m%d_%H%M%S")

        print("\n" + "=" * 60)
        print(f"🆔 Training run_id = {run_id}")
        print("🚀 Starting training pipeline")
        print(f"📌 Forecast horizon: {horizon} day(s)")
        print("=" * 60)

        # --------------------------------------------------
        # 2️⃣ Build dataset
        # --------------------------------------------------
        X, y = build_training_dataset(horizon)

        if X.empty or y.empty:
            raise RuntimeError("Training dataset is empty")

        print(f"📊 Dataset size: {X.shape[0]} rows")

        # --------------------------------------------------
        # 3️⃣ Train / Validation split
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
        print("\n🌲 Training Random Forest...")
        rf_model, rf_metrics = train_random_forest(
            X_train, y_train, X_val, y_val, horizon, run_id
        )

        print("\n⚡ Training XGBoost...")
        xgb_model, xgb_metrics = train_xgboost(
            X_train, y_train, X_val, y_val, horizon, run_id
        )

        print("\n🌊 Training Gradient Boosting...")
        gb_model, gb_metrics = train_gradient_boosting(
            X_train, y_train, X_val, y_val, horizon, run_id
        )

        print("\n🤝 Training Ensemble...")
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

        print("\n🎯 Production Model Selected:")
        print(f"Model Name : {best_model_info['model_name']}")
        print(f"RMSE       : {best_model_info.get('rmse')}")
        print(f"MAE        : {best_model_info.get('mae')}")

        print("\n✅ Training pipeline completed successfully")
        print("=" * 60)

    except Exception as e:
        print("\n❌ TRAINING FAILED")
        print("Reason:", str(e))
        raise


# --------------------------------------------------
# Allow running directly
# --------------------------------------------------
if __name__ == "__main__":
    run_training_pipeline(horizon=1)
