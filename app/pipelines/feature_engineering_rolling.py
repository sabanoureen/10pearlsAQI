import pandas as pd

def add_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds rolling AQI features.
    Assumes `aqi_pm25` column already exists.
    """
    df = df.copy()

    df["aqi_roll_3h"] = df["aqi_pm25"].rolling(window=3).mean()
    df["aqi_roll_6h"] = df["aqi_pm25"].rolling(window=6).mean()
    df["aqi_roll_12h"] = df["aqi_pm25"].rolling(window=12).mean()

    return df