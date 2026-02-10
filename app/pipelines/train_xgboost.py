from pathlib import Path
import joblib
import xgboost as xgb
from sklearn.metrics import mean_squared_error

from app.db.mongo import register_model


def train_xgboost(X_train, y_train, X_val, y_val, horizon):
    model = xgb.XGBRegressor(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        tree_method="hist",
    )

    model.fit(X_train, y_train)

    preds = model.predict(X_val)
    rmse = mean_squared_error(y_val, preds, squared=False)

    # ✅ SAVE MODEL
    model_dir = Path(f"models/xgb_h{horizon}")
    model_dir.mkdir(parents=True, exist_ok=True)

    model_path = model_dir / "model.joblib"
    joblib.dump(model, model_path)

    # ✅ REGISTER MODEL
    register_model(
        model_name="xgboost",
        horizon=horizon,
        model_path=str(model_path),
        features=list(X_train.columns),
        metrics={"rmse": rmse},
    )

    return model, {"rmse": rmse}
