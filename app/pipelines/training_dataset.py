"""
Build Training Dataset
----------------------
Creates features and shifted target for multi-day forecast (hourly data)
"""

import pandas as pd
from app.db.mongo import get_db

from app.pipelines.aqi_calculation import add_aqi_column
from app.pipelines.feature_engineering_time import add_time_features
from app.pipelines.feature_engineering_lag import add_lag_features
from app.pipelines.feature_engineering_rolling import add_rolling_features


def load_historical_df():
    collection = get_db()["historical_hourly_data"]
    data = list(collection.find({}, {"_id": 0}))

    if not data:
        raise RuntimeError("Historical data collection is empty")

    return pd.DataFrame(data)


def build_training_dataset(horizon: int):

    # ----------------------------------------
    # 1️⃣ Load historical data
    # ----------------------------------------
    df = load_historical_df()
    print("Total historical rows:", len(df))

    # ----------------------------------------
    # 2️⃣ Datetime processing
    # ----------------------------------------
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values("datetime").reset_index(drop=True)

    # ----------------------------------------
    # 3️⃣ Remove missing pm2_5
    # ----------------------------------------
    df = df.dropna(subset=["pm2_5"])
    print("After dropping pm2_5 NaNs:", len(df))

    # ----------------------------------------
    # 4️⃣ AQI calculation
    # ----------------------------------------
    df = add_aqi_column(df)
    df["aqi"] = df["aqi_pm25"]

    # ----------------------------------------
    # 5️⃣ Feature engineering (ONLY ONCE)
    # ----------------------------------------
    df = add_time_features(df)
    df = add_lag_features(df)
    df = add_rolling_features(df)

    # Drop rows created by lag/rolling
    df = df.dropna().reset_index(drop=True)

    print("Rows after feature engineering:", len(df))

    # ----------------------------------------
    # 6️⃣ Apply forecast shift
    # ----------------------------------------
    shift_hours = horizon * 24
    print(f"📌 Shift applied: {shift_hours} hours")

    df["target"] = df["aqi_pm25"].shift(-shift_hours)

    df = df.dropna(subset=["target"]).reset_index(drop=True)

    if df.empty:
        raise RuntimeError("Dataset empty after shift. Not enough history.")

    print("Rows after shift:", len(df))

    # ----------------------------------------
    # 7️⃣ Prepare X and y
    # ----------------------------------------
    drop_cols = ["target", "datetime", "pm2_5", "aqi_pm25"]

    X = df.drop(columns=drop_cols, errors="ignore")
    y = df["target"]

    return X, y
