"""
Training Pipeline
-----------------
- Builds training dataset
- Trains all ML models
- Saves models
- Registers models in MongoDB
- Selects best model automatically
"""

from datetime import datetime
import uuid
import argparse
import os

from app.db.mongo import get_model_registry   # ✅ ADD THIS
from app.pipelines.train_random_forest import train_random_forest
from app.pipelines.train_xgboost import train_xgboost
from app.pipelines.train_gradient_boosting import train_gradient_boosting
from app.pipelines.train_ensemble import train_ensemble
from app.pipelines.select_best_model import select_best_model
from app.pipelines.training_dataset import build_training_dataset




def run_training_pipeline(horizon: int = 1):

    try:
        run_id = datetime.utcnow().strftime("%Y%m%d_%H%M%S")

        print(f"\n🆔 Training run_id = {run_id}")
        print("🚀 Starting training pipeline")
        print(f"📌 Forecast horizon: {horizon} day(s)\n")

        # Build dataset
        X, y = build_training_dataset(horizon)

        if X.empty or y.empty:
            raise RuntimeError("Training dataset is empty")

        print(f"📊 Dataset size: {X.shape[0]} rows")

        # Clean old models
        registry = get_model_registry()
        registry.delete_many({"horizon": horizon})
        print("🧹 Old models deleted")

        # Split
        split_idx = int(len(X) * 0.8)

        X_train = X.iloc[:split_idx]
        X_val   = X.iloc[split_idx:]
        y_train = y.iloc[:split_idx]
        y_val   = y.iloc[split_idx:]

        # Train models
        rf_model, _ = train_random_forest(X_train, y_train, X_val, y_val, horizon)
        xgb_model, _ = train_xgboost(X_train, y_train, X_val, y_val, horizon)
        gb_model, _ = train_gradient_boosting(X_train, y_train, X_val, y_val, horizon)

        ensemble_model, _ = train_ensemble(
            rf_model=rf_model,
            xgb_model=xgb_model,
            gb_model=gb_model,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            horizon=horizon
        )

        best_model_info = select_best_model(horizon)

        print(f"\n🎯 Production Model: {best_model_info['model_name']}")
        print("✅ Training pipeline completed")

    except Exception as e:
        print("\n❌ TRAINING FAILED")
        print(str(e))
        raise
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--horizon",
        type=int,
        default=1,
        help="Forecast horizon in DAYS"
    )
    args = parser.parse_args()

    run_training_pipeline(horizon=args.horizon)
