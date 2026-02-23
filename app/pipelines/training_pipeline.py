import os
import joblib
from datetime import datetime

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from app.pipelines.training_dataset import build_training_dataset
from app.db.mongo import get_model_registry


# ---------------------------------------------------
# TRAIN SINGLE HORIZON
# ---------------------------------------------------
def train_horizon(df: pd.DataFrame, horizon: int):

    print(f"🔥 Starting training for horizon {horizon}")

    # Target column
    target_col = f"target_h{horizon}"

    if target_col not in df.columns:
        raise ValueError(f"{target_col} not found in dataset")

    X = df.drop(columns=["datetime", "target_h1", "target_h2", "target_h3"])
    y = df[target_col]

    # Train model
    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X, y)

    # Predictions for metrics
    preds = model.predict(X)

    rmse = mean_squared_error(y, preds, squared=False)
    mae = mean_absolute_error(y, preds)
    r2 = r2_score(y, preds)

    print("✅ Model trained")
    print("RMSE:", rmse)
    print("MAE:", mae)
    print("R2:", r2)

    # ---------------------------------------------------
    # SAVE MODEL TO DISK (Railway container)
    # ---------------------------------------------------
    MODEL_DIR = "models"
    os.makedirs(MODEL_DIR, exist_ok=True)

    model_path = os.path.join(MODEL_DIR, f"rf_h{horizon}.pkl")
    joblib.dump(model, model_path)

    print(f"💾 Model saved to {model_path}")

    # ---------------------------------------------------
    # REGISTER METADATA IN MONGO
    # ---------------------------------------------------
    registry = get_model_registry()

    # Remove old best model for this horizon
    registry.update_many(
        {"horizon": horizon},
        {"$set": {"is_best": False}}
    )

    registry.insert_one({
        "model_name": f"rf_h{horizon}",
        "horizon": horizon,
        "model_path": model_path,
        "features": list(X.columns),
        "rmse": float(rmse),
        "mae": float(mae),
        "r2": float(r2),
        "created_at": datetime.utcnow(),
        "is_best": True,
        "status": "production"
    })

    print("📦 Model metadata stored in Mongo")

    return {
        "horizon": horizon,
        "rmse": float(rmse),
        "mae": float(mae),
        "r2": float(r2)
    }


# ---------------------------------------------------
# RUN TRAINING PIPELINE
# ---------------------------------------------------
def run_training(horizon: int):

    if horizon not in [1, 2, 3]:
        raise ValueError("Horizon must be 1, 2, or 3")

    print("🚀 Building training dataset")

    df = build_training_dataset()

    print("Dataset shape:", df.shape)

    result = train_horizon(df, horizon)

    print("🎯 Training completed successfully")

    return result