from pathlib import Path
import joblib
import numpy as np
from sklearn.metrics import mean_squared_error
import numpy as np
from datetime import datetime

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
    horizon: int
):
    model = SimpleEnsemble([rf_model, xgb_model, gb_model])

    preds = model.predict(X_val)
    mse = mean_squared_error(y_val, preds)
    rmse = float(np.sqrt(mse))

    model_dir = Path(f"models/ensemble_h{horizon}")
    model_dir.mkdir(parents=True, exist_ok=True)
    from pathlib import Path

    model_dir = Path("models") / f"gbr_h{horizon}"
    model_dir.mkdir(parents=True, exist_ok=True)

    from pathlib import Path

    model_path = Path(model_doc["model_path"])

    joblib.dump(model, model_path)

    model_path_str = model_path.as_posix()

    joblib.dump(model, model_path)

    registry = get_model_registry()
    registry.insert_one({
    "model_name": "gradient_boosting",
    "horizon": horizon,
    "rmse": rmse,                     # ✅ TOP LEVEL
    "model_path": str(model_path),
    "features": list(X_train.columns),
    "is_best": False,
    "status": "candidate",
    "created_at": datetime.utcnow(),
    })


    return model, {"rmse": rmse}
