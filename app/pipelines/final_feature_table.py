import pandas as pd
from app.db.mongo import get_db


def build_training_dataset():
    """
    Load historical data from MongoDB
    Build clean dataset for multi-horizon training
    """

    db = get_db()
    collection = db["historical_hourly_data"]

    data = list(collection.find({}, {"_id": 0}))

    if not data:
        raise RuntimeError("No historical data found in MongoDB")

    df = pd.DataFrame(data)

    # Ensure datetime exists and sort properly
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values("datetime").reset_index(drop=True)

    print("Initial shape:", df.shape)

    # Use PM2.5 as AQI target
    df["aqi_pm25"] = df["pm2_5"]

    # Time Features
    df["hour"] = df["datetime"].dt.hour
    df["day"] = df["datetime"].dt.day
    df["month"] = df["datetime"].dt.month

    # Lag Features
    df["lag_1"] = df["aqi_pm25"].shift(1)
    df["lag_3"] = df["aqi_pm25"].shift(3)
    df["lag_6"] = df["aqi_pm25"].shift(6)

    # Rolling Features
    df["roll_mean_6"] = df["aqi_pm25"].rolling(6).mean()
    df["roll_mean_12"] = df["aqi_pm25"].rolling(12).mean()

    # Multi-horizon targets
    df["target_h1"] = df["aqi_pm25"].shift(-24)
    df["target_h2"] = df["aqi_pm25"].shift(-48)
    df["target_h3"] = df["aqi_pm25"].shift(-72)

    print("Before dropna:", df.shape)

    df = df.dropna().reset_index(drop=True)

    print("After dropna:", df.shape)

    return df
