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

from app.pipelines.train_random_forest import train_random_forest
from app.pipelines.train_xgboost import train_xgboost
from app.pipelines.train_gradient_boosting import train_gradient_boosting
from app.pipelines.train_ensemble import train_ensemble
from app.pipelines.select_best_model import select_best_model
from app.pipelines.training_dataset import build_training_dataset


def run_training_pipeline(horizon: int = 1):

    run_id = datetime.utcnow().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6]

    print(f"\n🆔 Training run_id = {run_id}")
    print("🚀 Starting training pipeline")
    print(f"📌 Forecast horizon: {horizon} day(s)")
    print(f"📌 Shift applied: {24*horizon} hours\n")

    # -----------------------------------------
    # 1️⃣ Build dataset
    # -----------------------------------------
    X, y = build_training_dataset(horizon)

    if X.empty or y.empty:
        raise RuntimeError("Training dataset is empty")

    print(f"📊 Dataset size: {df.shape[0]} rows")
    print(f"📊 Feature count: {df.shape[1]} columns")


    # -----------------------------------------
    # 2️⃣ Time-based split (NO SHUFFLE)
    # -----------------------------------------
    split_idx = int(len(X) * 0.8)

    X_train = X.iloc[:split_idx]
    X_val   = X.iloc[split_idx:]
    y_train = y.iloc[:split_idx]
    y_val   = y.iloc[split_idx:]

    print(f"🧪 Training samples: {len(X_train)}")
    print(f"🧪 Validation samples: {len(X_val)}\n")

    # Ensure models directory exists
    os.makedirs("models", exist_ok=True)

    # -----------------------------------------
    # 3️⃣ Train base models
    # -----------------------------------------
    print("🌲 Training Random Forest...")
    rf_model, rf_metrics = train_random_forest(
        X_train, y_train, X_val, y_val, horizon, run_id
    )

    print("⚡ Training XGBoost...")
    xgb_model, xgb_metrics = train_xgboost(
        X_train, y_train, X_val, y_val, horizon, run_id
    )

    print("🌊 Training Gradient Boosting...")
    gb_model, gb_metrics = train_gradient_boosting(
        X_train, y_train, X_val, y_val, horizon, run_id
    )

    # -----------------------------------------
    # 4️⃣ Train ensemble
    # -----------------------------------------
    print("🤝 Training Ensemble...")
    ensemble_model, ensemble_metrics = train_ensemble(
        rf_model=rf_model,
        xgb_model=xgb_model,
        gb_model=gb_model,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        horizon=horizon,
        run_id=run_id,
    )

    # -----------------------------------------
    # 5️⃣ Select best model
    # -----------------------------------------
    print("\n🏆 Selecting best model...")
    best_model_info = select_best_model(horizon)

    print(
        f"\n🎯 Production Model: {best_model_info['model_name']} "
        f"(RMSE={best_model_info['rmse']:.2f})"
    )

    print("\n✅ Training pipeline completed successfully")


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

print("📊 Feature count:", X.shape[1])
print("🧪 Training samples:", len(X_train))
print("🧪 Validation samples:", len(X_val))
print("📈 Best RMSE:", best_model["rmse"])
