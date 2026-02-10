"""
Training Pipeline
-----------------
- Trains all ML models
- Registers models in MongoDB
- Selects best model automatically
"""

from datetime import datetime
import uuid

from app.pipelines.train_random_forest import train_random_forest
from app.pipelines.train_xgboost import train_xgboost
from app.pipelines.train_gradient_boosting import train_gradient_boosting
from app.pipelines.train_ensemble import train_ensemble
from app.pipelines.select_best_model import select_best_model
from app.pipelines.training_dataset import build_training_dataset



run_id = datetime.utcnow().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6]
print(f"🆔 Training run_id = {run_id}")




def run_training_pipeline(horizon: int = 1):
    print("🚀 Starting training pipeline")
    print(f"📌 Forecast horizon: {horizon} hour(s)\n")

    # -----------------------------
    # 1️⃣ Build dataset
    # -----------------------------
    X, y = build_training_dataset()

    if X.empty or y.empty:
        raise RuntimeError("Training dataset is empty")

    # -----------------------------
    # 2️⃣ Time-based split
    # -----------------------------
    split_idx = int(len(X) * 0.8)

    X_train = X.iloc[:split_idx]
    X_val   = X.iloc[split_idx:]
    y_train = y.iloc[:split_idx]
    y_val   = y.iloc[split_idx:]

    # -----------------------------
    # 3️⃣ Train base models
    # -----------------------------
    rf_model, rf_metrics = train_random_forest(
        X_train, y_train, X_val, y_val, horizon, run_id
    )

    xgb_model, xgb_metrics = train_xgboost(
        X_train, y_train, X_val, y_val, horizon, run_id
    )

    gb_model, gb_metrics = train_gradient_boosting(
        X_train, y_train, X_val, y_val, horizon, run_id
    )

    # -----------------------------
    # 4️⃣ Train ensemble
    # -----------------------------
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

    # -----------------------------
    # 5️⃣ Select best model
    # -----------------------------
    select_best_model(horizon)

    print("\n✅ Training pipeline completed successfully")
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--horizon", type=int, default=1)
    args = parser.parse_args()

    run_training_pipeline(horizon=args.horizon)
