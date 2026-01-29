import requests
import pandas as pd


def fetch_karachi_air_quality():
    """
    Fetch hourly air quality data for Karachi using Open-Meteo
    Returns: pandas DataFrame
    """

    latitude = 24.8608
    longitude = 67.0104

    url = (
        "https://air-quality-api.open-meteo.com/v1/air-quality"
        f"?latitude={latitude}"
        f"&longitude={longitude}"
        "&hourly="
        "pm2_5,"
        "pm10,"
        "carbon_monoxide,"
        "nitrogen_dioxide,"
        "sulphur_dioxide,"
        "ozone"
        "&timezone=Asia/Karachi"
    )

    response = requests.get(url, timeout=30)
    response.raise_for_status()

    data = response.json()

    df = pd.DataFrame(data["hourly"])
    df["timestamp"] = pd.to_datetime(df["time"])
    df = df.drop(columns=["time"])

    return df


if __name__ == "__main__":
    df = fetch_karachi_air_quality()

    print("First 5 rows:")
    print(df.head())

    print("\nLast 5 rows:")
    print(df.tail())

    print("\nShape:", df.shape)
    print("\nColumns:", list(df.columns))