"""
Production Training Pipeline
----------------------------
• Trains 3 forecast horizons (1, 2, 3 days)
• Saves models locally
• Registers models in MongoDB model_registry
• Automatically updates production model
"""

import os
import joblib
import numpy as np
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

from app.db.mongo import get_model_registry, get_feature_store


# =========================================================
# CONFIG
# =========================================================

MODEL_BASE_PATH = "models"


# =========================================================
# LOAD FEATURE DATA
# =========================================================

def load_feature_data():
    feature_store = get_feature_store()
    data = list(feature_store.find({}, {"_id": 0}))

    if not data:
        raise RuntimeError("❌ No data found in feature_store")

    import pandas as pd
    df = pd.DataFrame(data)

    return df


# =========================================================
# REGISTER MODEL IN MONGODB
# =========================================================

def register_model(
    model_name,
    horizon,
    rmse,
    mae,
    r2,
    model_path,
    features
):
    registry = get_model_registry()

    # Archive old production model for this horizon
    registry.update_many(
        {"horizon": horizon, "is_best": True},
        {"$set": {"is_best": False, "status": "archived"}}
    )

    # Insert new production model
    registry.insert_one({
        "model_name": model_name,
        "horizon": horizon,
        "rmse": float(rmse),
        "mae": float(mae),
        "r2": float(r2),
        "model_path": model_path,
        "features": features,
        "status": "production",
        "is_best": True,
        "registered_at": datetime.utcnow()
    })

    print(f"✅ Registered production model for horizon {horizon}")


# =========================================================
# TRAIN SINGLE HORIZON
# =========================================================

def train_horizon(df, horizon):

    print(f"\n🚀 Training Horizon {horizon}...")

    target_column = f"target_h{horizon}"

    if target_column not in df.columns:
        raise RuntimeError(f"❌ Missing target column: {target_column}")

    X = df.drop(columns=[target_column])
    y = df[target_column]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=10,
        random_state=42
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    print(f"📊 Horizon {horizon} Metrics:")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE:  {mae:.4f}")
    print(f"R2:   {r2:.4f}")

    # Create model folder
    os.makedirs(f"{MODEL_BASE_PATH}/rf_h{horizon}", exist_ok=True)

    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    model_path = f"{MODEL_BASE_PATH}/rf_h{horizon}/model_{timestamp}.joblib"

    joblib.dump(model, model_path)

    print(f"💾 Model saved to {model_path}")

    # Register model in MongoDB
    register_model(
        model_name="random_forest",
        horizon=horizon,
        rmse=rmse,
        mae=mae,
        r2=r2,
        model_path=model_path,
        features=list(X.columns)
    )

    return model_path


# =========================================================
# MAIN TRAINING ENTRY
# =========================================================

def run_training():

    print("🔥 Starting Daily Training Pipeline")

    df = load_feature_data()

    for horizon in [1, 2, 3]:
        train_horizon(df, horizon)

    print("\n🎉 All horizons trained & registered successfully")


# =========================================================
# CLI ENTRY
# =========================================================

if __name__ == "__main__":
    run_training()
