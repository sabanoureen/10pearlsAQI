from pathlib import Path
import joblib
import xgboost as xgb
import numpy as np
from datetime import datetime
from sklearn.metrics import mean_squared_error, mean_absolute_error

from app.db.mongo import get_model_registry


def train_xgboost(
    X_train,
    y_train,
    X_val,
    y_val,
    horizon: int
):

    print("⚡ Training XGBoost...")

    model = xgb.XGBRegressor(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    preds = model.predict(X_val)

    rmse = float(np.sqrt(mean_squared_error(y_val, preds)))
    mae = float(mean_absolute_error(y_val, preds))

    print(f"XGB RMSE: {rmse:.2f}")
    print(f"XGB MAE : {mae:.2f}")

    model_dir = Path(f"models/xgb_h{horizon}")
    model_dir.mkdir(parents=True, exist_ok=True)

    model_path = model_dir / "model.joblib"
    joblib.dump(model, model_path)

    print(f"✅ Model saved to: {model_path}")

    registry = get_model_registry()

    registry.insert_one({
        "model_name": "xgboost",
        "horizon": horizon,
        "rmse": rmse,
        "mae": mae,
        "model_path": str(model_path),
        "features": list(X_train.columns),
        "status": "registered",   # ✅ FIXED
        "is_best": False,
        "registered_at": datetime.utcnow()
    })

    print("✅ XGBoost registered in Mongo")

    return model, {"rmse": rmse, "mae": mae}
