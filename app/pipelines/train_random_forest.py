from pathlib import Path
import joblib
import numpy as np
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error

from app.db.mongo import get_model_registry


def train_random_forest(X_train, y_train, X_val, y_val, horizon: int):

    print("🌲 Training Random Forest...")

    model = RandomForestRegressor(
        n_estimators=300,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    # -------------------------------
    # Validation Evaluation
    # -------------------------------
    preds = model.predict(X_val)

    rmse = float(np.sqrt(mean_squared_error(y_val, preds)))
    mae = float(mean_absolute_error(y_val, preds))

    print(f"RF RMSE: {rmse:.2f}")
    print(f"RF MAE : {mae:.2f}")

    # -------------------------------
    # Save Model (STABLE PATH)
    # -------------------------------
    model_dir = Path(f"models/rf_h{horizon}")
    model_dir.mkdir(parents=True, exist_ok=True)

    model_path = model_dir / "model.joblib"   # ✅ stable filename

    joblib.dump(model, model_path)

    print(f"✅ Model saved to: {model_path}")

    # -------------------------------
    # Register Model in Mongo
    # -------------------------------
    registry = get_model_registry()

    registry.insert_one({
        "model_name": "random_forest",
        "horizon": horizon,
        "rmse": rmse,
        "mae": mae,
        "model_path": str(model_path),
        "features": list(X_train.columns),
        "status": "candidate",
        "is_best": False,
        "registered_at": datetime.utcnow()
    })

    print("✅ Model registered as candidate in Mongo")

    return model, {"rmse": rmse, "mae": mae}
