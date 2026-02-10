from pathlib import Path
import joblib
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error

from app.db.mongo import register_model


def train_gradient_boosting(X_train, y_train, X_val, y_val, horizon):
    model = GradientBoostingRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=4,
        random_state=42,
    )

    model.fit(X_train, y_train)

    preds = model.predict(X_val)
    rmse = mean_squared_error(y_val, preds, squared=False)

    # ✅ SAVE MODEL
    model_dir = Path(f"models/gbr_h{horizon}")
    model_dir.mkdir(parents=True, exist_ok=True)

    model_path = model_dir / "model.joblib"
    joblib.dump(model, model_path)

    # ✅ REGISTER MODEL
    register_model(
        model_name="gradient_boosting",
        horizon=horizon,
        model_path=str(model_path),
        features=list(X_train.columns),
        metrics={"rmse": rmse},
    )

    return model, {"rmse": rmse}
