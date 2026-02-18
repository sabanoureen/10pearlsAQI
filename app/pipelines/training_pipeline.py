"""
Daily Training Pipeline
-----------------------
- Builds dataset from MongoDB
- Trains multiple models (RF, XGB, GB, etc.)
- Registers candidate models
- Automatically selects best model
- Promotes best model to production
"""

import argparse
from datetime import datetime
import pandas as pd

from app.pipelines.training_dataset import build_training_dataset
from app.pipelines.train_models import train_all_models
from app.pipelines.select_best_model import select_best_model
from app.db.mongo import get_model_registry


# ==========================================================
# MAIN TRAINING PIPELINE
# ==========================================================
def run_training_pipeline(horizon: int):

    print("\n" + "=" * 70)
    print(f"🚀 STARTING TRAINING PIPELINE | Horizon = H{horizon}")
    print("=" * 70)

    run_id = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    print(f"🆔 Run ID: {run_id}")

    # ------------------------------------------------------
    # 1️⃣ Load Training Dataset
    # ------------------------------------------------------
    print("\n📊 Building training dataset...")

    df = build_training_dataset()

    if df is None or df.empty:
        raise RuntimeError("❌ Training dataset is empty")

    print(f"✔ Dataset loaded | rows = {len(df)}")

    # ------------------------------------------------------
    # 2️⃣ Select Target Column
    # ------------------------------------------------------
    target_col = f"target_h{horizon}"

    if target_col not in df.columns:
        raise ValueError(f"❌ Target column {target_col} not found")

    X = df.drop(columns=["target_h1", "target_h2", "target_h3"])
    y = df[target_col]

    print(f"✔ Using target column: {target_col}")

    # ------------------------------------------------------
    # 3️⃣ Train / Validation Split
    # ------------------------------------------------------
    split_index = int(len(X) * 0.8)

    X_train = X.iloc[:split_index]
    X_val   = X.iloc[split_index:]
    y_train = y.iloc[:split_index]
    y_val   = y.iloc[split_index:]

    print(f"✔ Train size: {len(X_train)}")
    print(f"✔ Validation size: {len(X_val)}")

    # ------------------------------------------------------
    # 4️⃣ Train All Candidate Models
    # ------------------------------------------------------
    print("\n🤖 Training candidate models...")

    train_all_models(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        horizon=horizon,
        run_id=run_id
    )

    # ------------------------------------------------------
    # 5️⃣ Select Best Model Automatically
    # ------------------------------------------------------
    print("\n🏆 Selecting best model...")

    best_model_info = select_best_model(horizon)

    print("\n🎯 PRODUCTION MODEL SELECTED")
    print(f"Model Name : {best_model_info['model_name']}")
    print(f"RMSE       : {best_model_info['rmse']}")
    print(f"MAE        : {best_model_info['mae']}")

    print("\n✅ TRAINING PIPELINE COMPLETED SUCCESSFULLY")
    print("=" * 70)

    return best_model_info


# ==========================================================
# CLI ENTRY POINT (For GitHub Actions)
# ==========================================================
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--horizon",
        type=int,
        required=True,
        help="Forecast horizon (1, 2, or 3)"
    )

    args = parser.parse_args()

    if args.horizon not in [1, 2, 3]:
        raise ValueError("Horizon must be 1, 2, or 3")

    run_training_pipeline(args.horizon)
