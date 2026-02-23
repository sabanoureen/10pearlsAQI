import argparse
import io
from datetime import datetime

import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from app.pipelines.training_dataset import build_training_dataset
from app.db.mongo import (
    get_model_registry,
    get_feature_store
)


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
    # SIMPLE MODEL STORAGE (NO GRIDFS)
    # ---------------------------------------------------

    registry = get_model_registry()

    # Delete old model of same horizon
    registry.delete_many({"horizon": horizon})

    # Serialize model to binary
    buffer = io.BytesIO()
    joblib.dump(model, buffer)
    model_bytes = buffer.getvalue()

    registry.insert_one({
        "model_name": "random_forest",
        "horizon": horizon,
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "model_binary": model_bytes,
        "features": feature_cols,
        "status": "production",
        "is_best": True,
        "registered_at": datetime.utcnow()
    })

    print("📦 Model stored directly in MongoDB (NO GridFS)")


# ---------------------------------------------------
# Run Training
# ---------------------------------------------------
def run_training(horizon: int):

    print("🔥 Starting Daily Training Pipeline")

    df = build_training_dataset()

    print("Dataset shape:", df.shape)

    # ---------------------------------------------------
    # SAFE FEATURE STORE (ONLY LATEST ROW)
    # ---------------------------------------------------

    feature_store = get_feature_store()

    feature_columns = [
        col for col in df.columns
        if not col.startswith("target_")
    ]

    latest_row = df[feature_columns].iloc[-1].to_dict()

    # Convert pandas Timestamp to Python datetime
    if "datetime" in latest_row:
        latest_row["datetime"] = latest_row["datetime"].to_pydatetime()

    # Keep only 1 row
    feature_store.delete_many({})
    feature_store.insert_one(latest_row)

    print("📦 Feature store updated (1 row only)")

    # Train model
    train_horizon(df, horizon)


# ---------------------------------------------------
# CLI ENTRY
# ---------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--horizon", type=int, required=True)
    args = parser.parse_args()

    run_training(args.horizon)