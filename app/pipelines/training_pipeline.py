import argparse
import os
from datetime import datetime

import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from app.pipelines.training_dataset import build_training_dataset
from app.db.mongo import (
    get_model_registry,
    get_feature_store
)


BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_DIR = os.path.join(BASE_DIR, "models")

os.makedirs(MODEL_DIR, exist_ok=True)


# ---------------------------------------------------
# Train One Horizon
# ---------------------------------------------------
def train_horizon(df, horizon: int):

    target_column = f"target_h{horizon}"

    if target_column not in df.columns:
        raise RuntimeError(f"Missing target column: {target_column}")

    feature_cols = [
        "hour",
        "day",
        "month",
        "lag_1",
        "lag_3",
        "lag_6",
        "roll_mean_6",
        "roll_mean_12",
    ]

    X = df[feature_cols]
    y = df[target_column]

    split_index = int(len(df) * 0.8)

    X_train, X_test = X[:split_index], X[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]

    model = RandomForestRegressor(
        n_estimators=200,
        random_state=42
    )

    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    rmse = mean_squared_error(y_test, preds, squared=False)
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)

    print(f"✅ Horizon {horizon} trained")
    print("RMSE:", rmse)
    print("MAE:", mae)
    print("R2:", r2)

    # ---------------------------------------------------
    # SAVE MODEL TO DISK (NOT MONGODB)
    # ---------------------------------------------------

    model_path = os.path.join(MODEL_DIR, f"rf_h{horizon}.pkl")

    print(f"📦 Model saved to disk: {model_path}")

    # ---------------------------------------------------
    # SAVE METADATA TO MONGODB
    # ---------------------------------------------------

    registry = get_model_registry()

    registry.delete_many({"horizon": horizon})

    registry.insert_one({
        "model_name": "random_forest",
        "horizon": horizon,
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "model_path": model_path,
        "features": feature_cols,
        "status": "production",
        "is_best": True,
        "registered_at": datetime.utcnow()
    })

    print("📦 Model metadata stored in MongoDB")


# ---------------------------------------------------
# Run Training
# ---------------------------------------------------
def run_training(horizon: int):

    print("🔥 Starting Daily Training Pipeline")

    df = build_training_dataset()

    print("Dataset shape:", df.shape)

    # ---------------------------------------------------
    # FEATURE STORE (1 ROW ONLY)
    # ---------------------------------------------------

    feature_store = get_feature_store()

    feature_columns = [
        col for col in df.columns
        if not col.startswith("target_")
    ]

    latest_row = df[feature_columns].iloc[-1].to_dict()

    if "datetime" in latest_row:
        latest_row["datetime"] = latest_row["datetime"].to_pydatetime()

    feature_store.delete_many({})
    feature_store.insert_one(latest_row)

    print("📦 Feature store updated (1 row only)")

    train_horizon(df, horizon)


# ---------------------------------------------------
# CLI
# ---------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--horizon", type=int, required=True)
    args = parser.parse_args()

    run_training(args.horizon)