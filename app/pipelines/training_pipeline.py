"""
Training Pipeline
-----------------
- Builds dataset
- Trains models for selected horizon
- Automatically selects best model
"""

import argparse
from datetime import datetime

from app.pipelines.training_dataset import build_training_dataset
from app.pipelines.train_models import train_all_models
from app.pipelines.select_best_model import select_best_model


def run_training_pipeline(horizon: int):

    print("\n" + "=" * 60)
    run_id = datetime.utcnow().strftime("%Y%m%d_%H%M%S")

    print(f"🆔 Training run_id = {run_id}")
    print(f"📌 Forecast horizon: {horizon}")
    print("=" * 60)

    # 1️⃣ Build Dataset
    df = build_training_dataset()

    X = df.drop(columns=[f"target_h{horizon}"])
    y = df[f"target_h{horizon}"]

    # 2️⃣ Train Models
    train_all_models(X, y, horizon=horizon, run_id=run_id)

    # 3️⃣ Select Best Model Automatically
    best = select_best_model(horizon)

    print("\n🏆 Best Model Selected:")
    print(best)

    print("\n✅ Training completed")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--horizon", type=int, required=True)
    args = parser.parse_args()

    run_training_pipeline(args.horizon)
