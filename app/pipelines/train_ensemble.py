from pathlib import Path
import joblib
import numpy as np
from datetime import datetime
from sklearn.metrics import mean_squared_error, mean_absolute_error

from app.db.mongo import get_model_registry


class SimpleEnsemble:
    def __init__(self, models):
        self.models = models

    def predict(self, X):
        preds = np.column_stack([m.predict(X) for m in self.models])
        return preds.mean(axis=1)


def train_ensemble(
    rf_model,
    xgb_model,
    gb_model,
    X_train,
    y_train,
    X_val,
    y_val,
    horizon: int,
    run_id: str,
):

    print("🤝 Training Ensemble...")

    model = SimpleEnsemble([rf_model, xgb_model, gb_model])

    # -------------------------------
    # Validation Evaluation
    # -------------------------------
    preds = model.predict(X_val)

    rmse = float(np.sqrt(mean_squared_error(y_val, preds)))
    mae = float(mean_absolute_error(y_val, preds))

    print(f"Ensemble RMSE: {rmse:.2f}")
    print(f"Ensemble MAE : {mae:.2f}")

    # -------------------------------
    # Save Model (Versioned)
    # -------------------------------
    model_dir = Path(f"models/ensemble_h{horizon}")
    model_dir.mkdir(parents=True, exist_ok=True)

    model_filename = f"ensemble_{run_id}.joblib"
    model_path = model_dir / model_filename

    joblib.dump(model, model_path)

    # -------------------------------
    # Register Model in Mongo
    # -------------------------------
    # -------------------------------
# Register Model in Mongo
# -------------------------------
    registry = get_model_registry()

    registry.insert_one({
    "model_name": "random_forest",
    "horizon": horizon,
    "rmse": rmse,
    "mae": mae,
    "model_path": str(model_path),   # 🔥 important
    "features": list(X_train.columns),  # 🔥 important
    "status": "candidate",
    "is_best": False,
    "registered_at": datetime.utcnow()
    })

    return model, {"rmse": rmse, "mae": mae}
