import pandas as pd
import numpy as np

def add_time_features(df):

    # ✅ Ensure datetime exists
    if "datetime" not in df.columns:
        if "updated_at" in df.columns:
            df["datetime"] = pd.to_datetime(df["updated_at"])
        else:
            # fallback: create current timestamp
            df["datetime"] = pd.Timestamp.utcnow()

    df["datetime"] = pd.to_datetime(df["datetime"])

    """
    Adds time-based features using datetime column.
    """

    if "datetime" not in df.columns:
        raise RuntimeError("datetime column missing for time feature engineering")

    df["hour"] = df["datetime"].dt.hour
    df["day_of_week"] = df["datetime"].dt.dayofweek
    df["month"] = df["datetime"].dt.month

    # Cyclical encoding
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

    return df
