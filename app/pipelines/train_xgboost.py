from pathlib import Path
import joblib
import numpy as np
from datetime import datetime
import xgboost as xgb

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit

from app.db.mongo import get_model_registry


def train_xgboost(
    X_train,
    y_train,
    X_val,
    y_val,
    horizon: int,
    run_id: str
):

    print("⚡ Training XGBoost with GridSearch...")

    model = xgb.XGBRegressor(
        objective="reg:squarederror",
        random_state=42,
        n_jobs=-1
    )

    param_grid = {
        "n_estimators": [200, 300],
        "max_depth": [3, 4],
        "learning_rate": [0.03, 0.05],
        "subsample": [0.8, 1.0],
        "colsample_bytree": [0.8, 1.0]
    }

    tscv = TimeSeriesSplit(n_splits=3)

    grid = GridSearchCV(
        model,
        param_grid,
        cv=tscv,
        scoring="neg_root_mean_squared_error",
        verbose=0,
        n_jobs=-1
    )

    # ✅ THIS LINE IS CRITICAL
    grid.fit(X_train, y_train)

    print("🔎 Best Parameters:", grid.best_params_)

    best_model = grid.best_estimator_

    # -------------------------------
    # Validation Evaluation
    # -------------------------------
    preds_log = best_model.predict(X_val)

    preds = np.expm1(preds_log)
    y_true = np.expm1(y_val)

    rmse = float(np.sqrt(mean_squared_error(y_true, preds)))
    mae = float(mean_absolute_error(y_true, preds))
    r2 = float(r2_score(y_true, preds))


    print(f"XGB RMSE: {rmse:.4f}")
    print(f"XGB MAE : {mae:.4f}")
    print(f"XGB R2  : {r2:.4f}")

    # -------------------------------
    # Save Model
    # -------------------------------
    model_dir = Path(f"models/xgb_h{horizon}")
    model_dir.mkdir(parents=True, exist_ok=True)

    model_path = model_dir / f"model_{run_id}.joblib"
    joblib.dump(best_model, model_path)

    # -------------------------------
    # Register Model
    # -------------------------------
    registry = get_model_registry()

    registry.insert_one({
        "model_name": "xgboost",
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

    print("✅ XGBoost registered")

    return best_model, {
        "rmse": rmse,
        "mae": mae,
        "r2": r2
    }
