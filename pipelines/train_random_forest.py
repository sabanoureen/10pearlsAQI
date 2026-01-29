import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

from prepare_forecast_target import build_forecasting_dataset
from pipelines.final_feature_table import build_training_dataset


def train_random_forest():
    # Load forecasting dataset (1-hour ahead)
    X, y = build_forecasting_dataset(horizon=1)

    # -----------------------------
    # Time-based split
    # -----------------------------
    split_idx = int(len(X) * 0.8)

    X_train = X.iloc[:split_idx]
    X_test = X.iloc[split_idx:]

    y_train = y.iloc[:split_idx]
    y_test = y.iloc[split_idx:]

    # -----------------------------
    # Random Forest Model
    # -----------------------------
    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=None,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    # -----------------------------
    # Evaluation
    # -----------------------------
    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    print("Random Forest Results")
    print("---------------------")
    print(f"RMSE: {rmse:.2f}")
    print(f"R²:   {r2:.3f}")

        # -----------------------------
    # Feature Importance
    # -----------------------------
    import pandas as pd

    feature_importance = pd.Series(
        model.feature_importances_,
        index=X.columns
    ).sort_values(ascending=False)

    print("\nTop 10 Feature Importances:")
    print(feature_importance.head(10))

    return model
    # -----------------------------
    # Feature Importance
    # -----------------------------
    import pandas as pd

    feature_importance = pd.Series(
        model.feature_importances_,
        index=X.columns
    ).sort_values(ascending=False)

    print("\nTop 10 Feature Importances:")
    print(feature_importance.head(10))


if __name__ == "__main__":
    train_random_forest()