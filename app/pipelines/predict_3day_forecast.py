import pandas as pd

from app.utils.load_production_model import load_production_model
from app.db.mongo import get_db
from app.pipelines.fetch_live_openmeteo import fetch_live_weather
from app.pipelines.aqi_calculation import add_aqi_column
from app.pipelines.feature_engineering_time import add_time_features
from app.pipelines.feature_engineering_lag import add_lag_features
from app.pipelines.feature_engineering_rolling import add_rolling_features


def load_last_historical_rows(n=24):
   db = get_db()
    collection = db["forecast_results"]

    collection.insert_many(
    df_forecast[["datetime", "predicted_aqi"]]
    .assign(
        horizon=horizon,
        generated_at=datetime.utcnow()
    )
    .to_dict("records")
)


    df = pd.DataFrame(data)
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values("datetime").reset_index(drop=True)

    return df


def predict_next_3_days(horizon: int = 3):

    print("🔮 Predicting next 3 days...")

    # 1️⃣ Load production model
    model, feature_list = load_production_model(horizon)

    # 2️⃣ Load last historical rows (for lag warm start)
    historical_df = load_last_historical_rows(24)

    # 3️⃣ Fetch live 3-day forecast weather
    forecast_df = fetch_live_weather()
    print("Live rows fetched:", len(forecast_df))

    # 4️⃣ Combine historical + forecast
    df = pd.concat([historical_df, forecast_df]).reset_index(drop=True)

    # 5️⃣ Feature engineering
    df = add_aqi_column(df)
    df["aqi"] = df["aqi_pm25"]

    df = add_time_features(df)
    df = add_lag_features(df)
    df = add_rolling_features(df)

    # Only drop rows missing lag features
    required_cols = [col for col in feature_list if col in df.columns]
    df = df.dropna(subset=required_cols).reset_index(drop=True)


    # 6️⃣ Keep only forecast horizon rows
    df_forecast = df.tail(horizon * 24)

    # Ensure correct feature order
    X = df_forecast[feature_list]

    # 7️⃣ Predict
    predictions = model.predict(X)

    df_forecast["predicted_aqi"] = predictions

    print("\n📊 3-Day AQI Forecast:")
    print(df_forecast[["datetime", "predicted_aqi"]])

    return df_forecast[["datetime", "predicted_aqi"]]


if __name__ == "__main__":
    predict_next_3_days(horizon=3)
