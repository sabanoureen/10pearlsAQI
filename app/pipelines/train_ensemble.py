from pathlib import Path
import joblib
import numpy as np
from datetime import datetime
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from app.db.mongo import get_model_registry


class WeightedEnsemble:
    def __init__(self, models, weights):
        self.models = models
        self.weights = weights

    def predict(self, X):
        preds = np.column_stack([m.predict(X) for m in self.models])
        return np.dot(preds, self.weights)


def train_ensemble(
    rf_model,
    xgb_model,
    gb_model,
    X_train,
    y_train,
    X_val,
    y_val,
    horizon: int,
    run_id: str
):

    print("🤝 Training Weighted Ensemble...")

    # Equal weights (simple and stable)
    weights = np.array([1/3, 1/3, 1/3])

    model = WeightedEnsemble(
        models=[rf_model, xgb_model, gb_model],
        weights=weights
    )

    preds_log = model.predict(X_val)

    preds = np.expm1(preds_log)
    y_true = np.expm1(y_val)

    rmse = float(np.sqrt(mean_squared_error(y_true, preds)))
    mae = float(mean_absolute_error(y_true, preds))
    r2 = float(r2_score(y_true, preds))


    print(f"Ensemble RMSE: {rmse:.4f}")
    print(f"Ensemble MAE : {mae:.4f}")
    print(f"Ensemble R2  : {r2:.4f}")

    # Save locally
    model_dir = Path(f"models/ensemble_h{horizon}")
    model_dir.mkdir(parents=True, exist_ok=True)

    model_path = model_dir / f"model_{run_id}.joblib"
    joblib.dump(model, model_path)

    # Register metadata only
    registry = get_model_registry()

    registry.insert_one({
        "model_name": "ensemble",
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

    print("✅ Ensemble registered")

    return model, {"rmse": rmse, "mae": mae, "r2": r2}
