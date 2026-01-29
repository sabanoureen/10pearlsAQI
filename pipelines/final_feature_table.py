import pandas as pd

from pipelines.fetch_karachi_aqi import fetch_karachi_air_quality
from pipelines.aqi_calculation import add_aqi_column
from pipelines.feature_engineering_time import add_time_features
from pipelines.feature_engineering_lag import add_lag_features
from pipelines.feature_engineering_rolling import add_rolling_features


def build_final_dataframe() -> pd.DataFrame:
    """
    Builds feature dataframe for LIVE inference.
    DO NOT drop NaNs here.
    """
    df = fetch_karachi_air_quality()
    df = add_aqi_column(df)

    df = add_time_features(df)
    df = add_lag_features(df)
    df = add_rolling_features(df)

    return df


def build_training_dataset():
    """
    Used ONLY during training.
    """
    df = build_final_dataframe()

    df = df.dropna().reset_index(drop=True)

    y = df["aqi_pm25"]
    X = df.drop(columns=["aqi_pm25", "timestamp"], errors="ignore")

    return X, y