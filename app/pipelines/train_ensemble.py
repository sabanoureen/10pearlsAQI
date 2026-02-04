from pathlib import Path
import json
import joblib
import numpy as np

from sklearn.metrics import mean_squared_error, r2_score

from app.pipelines.register_model import register_model


class MeanEnsemble:
    def __init__(self, models):
        self.models = models

    def predict(self, X):
        preds = [m.predict(X) for m in self.models]
        return np.mean(preds, axis=0)


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
    print("🤝 Training Ensemble Model")

    model = MeanEnsemble([rf_model, xgb_model, gb_model])

    preds = model.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, preds))
    r2 = r2_score(y_val, preds)

    model_dir = Path(f"models/ensemble_h{horizon}")
    model_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, model_dir / "model.joblib")
    (model_dir / "features.json").write_text(json.dumps(list(X_train.columns)))

    register_model(
        model_name="ensemble",
        horizon=horizon,
        rmse=rmse,
        r2=r2,
        model_path=str(model_dir / "model.joblib"),
        features=list(X_train.columns)
    )

    print(f"✅ Ensemble done | RMSE={rmse:.2f} | R²={r2:.3f}")
    return model, {"rmse": rmse, "r2": r2}
