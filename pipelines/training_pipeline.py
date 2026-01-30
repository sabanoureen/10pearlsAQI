"""
Training Pipeline
-----------------
- Trains all ML models
- Evaluates them using RMSE / R²
- Registers models in MongoDB
- Selects best model automatically

Single authoritative entry point for training.
"""

import sys
import os

# -----------------------------------
# Ensure project root is on PYTHONPATH
# -----------------------------------
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
)

from pipelines.train_random_forest import train_random_forest
from pipelines.train_xgboost import train_xgboost
from pipelines.train_gradient_boosting import train_gradient_boosting
from pipelines.train_ensemble import train_ensemble
from pipelines.select_best_model import select_best_model


def run_training_pipeline(horizon: int = 1):
    print("🚀 Starting training pipeline")
    print(f"📌 Forecast horizon: {horizon} hour(s)\n")

    # -----------------------------
    # Train individual models
    # -----------------------------
    train_random_forest(horizon)
    train_xgboost(horizon)
    train_gradient_boosting(horizon)

    # -----------------------------
    # Train ensemble
    # -----------------------------
    train_ensemble(horizon)

    # -----------------------------
    # Select best model
    # -----------------------------
    select_best_model(horizon)

    print("\n✅ Training pipeline completed successfully")


if __name__ == "__main__":
    run_training_pipeline(horizon=1)
