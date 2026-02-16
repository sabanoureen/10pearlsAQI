import pandas as pd
import numpy as np
from app.db.mongo import get_db
from app.pipelines.feature_engineering_time import add_time_features
from app.pipelines.feature_engineering_lag import add_lag_features
from app.pipelines.feature_engineering_rolling import add_rolling_features


# -------------------------------------------------------
# Load Historical Data
# -------------------------------------------------------
def load_historical_df():

    collection = get_db()["historical_hourly_data"]
    data = list(collection.find({}, {"_id": 0}))

    if not data:
        raise RuntimeError("Historical data empty")

    return pd.DataFrame(data)


# -------------------------------------------------------
# Build Training Dataset
# -------------------------------------------------------
def build_training_dataset(horizon: int):

    df = load_historical_df()

    # Datetime processing
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values("datetime").reset_index(drop=True)

    # Remove missing pm2_5
    df = df.dropna(subset=["pm2_5"])

    # 🔥 Create TARGET FIRST (before log)
    shift_hours = horizon * 24
    df["target"] = df["pm2_5"].shift(-shift_hours)

    # Drop rows where future target is missing
    df = df.dropna(subset=["target"]).reset_index(drop=True)

    # 🔥 Apply log transform to BOTH feature + target
    df["pm2_5"] = np.log1p(df["pm2_5"])
    df["target"] = np.log1p(df["target"])

    # Feature engineering
    df = add_time_features(df)
    df = add_lag_features(df)
    df = add_rolling_features(df)

    # Drop rows created by lag/rolling
    df = df.dropna().reset_index(drop=True)

    # 🔥 REMOVE current pm2_5 from features (NO LEAKAGE)
    drop_cols = ["datetime", "target", "pm2_5"]

    X = df.drop(columns=drop_cols)
    y = df["target"]

    return X, y
