import requests
import pandas as pd
from datetime import datetime
from app.db.mongo import get_db


def download_historical_data(start_date: str, end_date: str):
    """
    Downloads historical hourly air quality data from Open-Meteo
    and stores it in MongoDB.
    """

    latitude = 24.8608
    longitude = 67.0104

    url = (
        "https://archive-api.open-meteo.com/v1/archive"
        f"?latitude={latitude}"
        f"&longitude={longitude}"
        f"&start_date={start_date}"
        f"&end_date={end_date}"
        "&hourly="
        "pm2_5,"
        "pm10,"
        "carbon_monoxide,"
        "nitrogen_dioxide,"
        "sulphur_dioxide,"
        "ozone"
        "&timezone=Asia/Karachi"
    )

    print("Fetching historical data...")
    response = requests.get(url, timeout=60)
    response.raise_for_status()

    data = response.json()
    df = pd.DataFrame(data["hourly"])

    df["datetime"] = pd.to_datetime(df["time"])
    df = df.drop(columns=["time"])

    print("Rows downloaded:", len(df))

    # Store in Mongo
    db = get_db()
    collection = db["historical_hourly_data"]

    # Clear old historical data
    collection.delete_many({})

    records = df.to_dict("records")
    collection.insert_many(records)

    print("✅ Historical data saved to Mongo")


if __name__ == "__main__":
    # 5 months example
    download_historical_data(
        start_date="2024-09-01",
        end_date="2025-02-01"
    )
