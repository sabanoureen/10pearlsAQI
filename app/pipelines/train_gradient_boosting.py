from pathlib import Path
import joblib
import numpy as np
from datetime import datetime
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error

from app.db.mongo import get_model_registry


def train_gradient_boosting(X_train, y_train, X_val, y_val, horizon, run_id : int):
    model = GradientBoostingRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=3,
        random_state=42
    )

    model.fit(X_train, y_train)

    preds = model.predict(X_val)
    rmse = float(np.sqrt(mean_squared_error(y_val, preds)))

    # ✅ Correct directory
    model_dir = Path(f"models/gbr_h{horizon}")
    model_dir.mkdir(parents=True, exist_ok=True)

    model_path = model_dir / "model.joblib"
    joblib.dump(model, model_path)

    registry = get_model_registry()
    registry.insert_one({
        "model_name": "gradient_boosting",
        "horizon": horizon,
        "rmse": rmse,
        "model_path": model_path.as_posix(),
        "features": list(X_train.columns),
        "is_best": False,
        "status": "candidate",
        "created_at": datetime.utcnow(),
        "version": run_id,

    })

    return model, {"rmse": rmse}
