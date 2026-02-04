import numpy as np
from sklearn.ensemble import VotingRegressor
from sklearn.metrics import mean_squared_error, r2_score


def train_ensemble(
    rf_model,
    xgb_model,
    gb_model,
    X_train,
    y_train,
    X_val,
    y_val,
):
    """
    Train ensemble using already-trained base models.
    """

    print("🤝 Training Ensemble Model")

    ensemble = VotingRegressor(
        estimators=[
            ("rf", rf_model),
            ("xgb", xgb_model),
            ("gb", gb_model),
        ]
    )

    ensemble.fit(X_train, y_train)

    preds = ensemble.predict(X_val)
    rmse = mean_squared_error(y_val, preds) ** 0.5
    r2 = r2_score(y_val, preds)

    print(f"✅ Ensemble done | RMSE={rmse:.2f} | R²={r2:.3f}")

    return ensemble, {
        "rmse": rmse,
        "r2": r2
    }

