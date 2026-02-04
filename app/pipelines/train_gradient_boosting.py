from pathlib import Path
import json
import joblib
import numpy as np

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score

from app.pipelines.register_model import register_model


def train_gradient_boosting(X_train, y_train, X_val, y_val, horizon: int):
    print("🌿 Training Gradient Boosting")

    model = GradientBoostingRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=4,
        random_state=42
    )
    model.fit(X_train, y_train)

    preds = model.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, preds))
    r2 = r2_score(y_val, preds)

    model_dir = Path(f"models/gradient_boosting_h{horizon}")
    model_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, model_dir / "model.joblib")
    (model_dir / "features.json").write_text(json.dumps(list(X_train.columns)))

    register_model(
        model_name="gradient_boosting",
        horizon=horizon,
        rmse=rmse,
        r2=r2,
        model_path=str(model_dir / "model.joblib"),
        features=list(X_train.columns)
    )

    print(f"✅ Gradient Boosting done | RMSE={rmse:.2f} | R²={r2:.3f}")
    return model, {"rmse": rmse, "r2": r2}
