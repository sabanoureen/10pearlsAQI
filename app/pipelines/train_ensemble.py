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
):

    print("🤝 Training Ensemble...")

    model = SimpleEnsemble([rf_model, xgb_model, gb_model])

    # Validation
    preds = model.predict(X_val)

    rmse = float(np.sqrt(mean_squared_error(y_val, preds)))
    mae = float(mean_absolute_error(y_val, preds))

    print(f"Ensemble RMSE: {rmse:.2f}")
    print(f"Ensemble MAE : {mae:.2f}")

    # Save model (stable path)
    model_dir = Path(f"models/ensemble_h{horizon}")
    model_dir.mkdir(parents=True, exist_ok=True)

    model_path = model_dir / "model.joblib"

    joblib.dump(model, model_path)

    print(f"✅ Ensemble saved to: {model_path}")

    # Register in Mongo
    registry = get_model_registry()

    registry.insert_one({
    "model_name": "ensemble",
    "horizon": horizon,
    "rmse": rmse,
    "mae": mae,
    "model_path": str(model_path),
    "features": list(X_train.columns),
    "status": "registered",   # ✅ changed
    "is_best": False,
    "registered_at": datetime.utcnow()
})

    print("✅ Ensemble registered in MongoDB")

    return model, {"rmse": rmse, "mae": mae}
