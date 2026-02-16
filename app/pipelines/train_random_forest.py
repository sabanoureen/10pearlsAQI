from pathlib import Path
import joblib
import numpy as np
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from app.db.mongo import get_model_registry


def train_random_forest(X_train, y_train, X_val, y_val, horizon: int, run_id: str):

    print("🌲 Training Random Forest...")

    model = RandomForestRegressor(
        n_estimators=300,
        random_state=42,
        n_jobs=-1
    )

    # 🔹 TimeSeries Cross Validation
    tscv = TimeSeriesSplit(n_splits=5)
    cv_scores = []

    for train_idx, test_idx in tscv.split(X_train):
        X_tr, X_te = X_train.iloc[train_idx], X_train.iloc[test_idx]
        y_tr, y_te = y_train.iloc[train_idx], y_train.iloc[test_idx]

        model.fit(X_tr, y_tr)
        preds = model.predict(X_te)

        rmse = np.sqrt(mean_squared_error(y_te, preds))
        cv_scores.append(rmse)

    print("TimeSeries CV RMSE:", np.mean(cv_scores))

    # 🔹 Train on full train set
    model.fit(X_train, y_train)

    # Predictions (log scale)
    preds_log = model.predict(X_val)

# Convert back to original scale
    preds = np.expm1(preds_log)
    y_true = np.expm1(y_val)

    rmse = float(np.sqrt(mean_squared_error(y_true, preds)))
    mae = float(mean_absolute_error(y_true, preds))
    r2 = float(r2_score(y_true, preds))



    print(f"RMSE: {rmse:.4f}")
    print(f"R2  : {r2:.4f}")

    # Save model
    model_dir = Path(f"models/rf_h{horizon}")
    model_dir.mkdir(parents=True, exist_ok=True)

    model_path = model_dir / f"model_{run_id}.joblib"
    joblib.dump(model, model_path)

    # Register
    registry = get_model_registry()

    registry.insert_one({
        "model_name": "random_forest",
        "horizon": horizon,
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "model_path": str(model_path),
        "features": list(X_train.columns),
        "status": "candidate",
        "is_best": False,
        "registered_at": datetime.utcnow()
    })

    print("✅ Random Forest registered")

    return model, {"rmse": rmse, "mae": mae, "r2": r2}
