import os
import joblib
import numpy as np

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

from app.pipelines.final_feature_table import build_training_dataset


def train_models():

    df = build_training_dataset()

    horizons = {
        "h1": "target_h1",
        "h2": "target_h2",
        "h3": "target_h3"
    }

    os.makedirs("models", exist_ok=True)

    for h_name, target_col in horizons.items():

        print(f"\n==============================")
        print(f"Training models for {h_name}")
        print(f"==============================")

        y = df[target_col]
        X = df.drop(
            columns=["datetime", "aqi_pm25", "target_h1", "target_h2", "target_h3"],
            errors="ignore"
        )

        models = {
            "rf": RandomForestRegressor(n_estimators=200, random_state=42),
            "gb": GradientBoostingRegressor(random_state=42),
            "ridge": Ridge()
        }

        best_rmse = float("inf")
        best_model_name = None

        for model_name, model in models.items():

            model.fit(X, y)
            preds = model.predict(X)
            rmse = np.sqrt(mean_squared_error(y, preds))

            print(f"{model_name.upper()} RMSE: {rmse:.4f}")

            save_path = f"models/{model_name}_{h_name}.joblib"
            joblib.dump(model, save_path)

            if rmse < best_rmse:
                best_rmse = rmse
                best_model_name = model_name

        print(f"\n🏆 Best model for {h_name}: {best_model_name.upper()} (RMSE={best_rmse:.4f})")


if __name__ == "__main__":
    train_models()
