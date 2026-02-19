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
def build_training_dataset(horizon):

    df = ...  # your existing code

    # Select correct target
    if horizon == 1:
        target_col = "target_h1"
    elif horizon == 2:
        target_col = "target_h2"
    elif horizon == 3:
        target_col = "target_h3"
    else:
        raise ValueError("Invalid horizon")

    feature_cols = [
        "hour",
        "day",
        "month",
        "lag_1",
        "lag_3",
        "lag_6",
        "roll_mean_6",
        "roll_mean_12"
    ]

    X = df[feature_cols]
    y = df[target_col]

    return X, y

