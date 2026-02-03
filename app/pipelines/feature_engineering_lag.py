import pandas as pd

def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds lag-based AQI features.
    Assumes `aqi_pm25` column already exists.
    """
    df = df.copy()

    df["aqi_lag_1h"] = df["aqi_pm25"].shift(1)
    df["aqi_lag_3h"] = df["aqi_pm25"].shift(3)
    df["aqi_lag_6h"] = df["aqi_pm25"].shift(6)

    return df