from pathlib import Path
import joblib
import numpy as np
from sklearn.metrics import mean_squared_error

from app.db.mongo import register_model


class MeanEnsemble:
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
    horizon,
):
    model = MeanEnsemble([rf_model, xgb_model, gb_model])

    preds = model.predict(X_val)
    rmse = mean_squared_error(y_val, preds, squared=False)

    # ✅ SAVE MODEL
    model_dir = Path(f"models/ensemble_h{horizon}")
    model_dir.mkdir(parents=True, exist_ok=True)

    model_path = model_dir / "model.joblib"
    joblib.dump(model, model_path)

    # ✅ REGISTER MODEL
    register_model(
        model_name="ensemble",
        horizon=horizon,
        model_path=str(model_path),
        features=list(X_train.columns),
        metrics={"rmse": rmse},
    )

    return model, {"rmse": rmse}
