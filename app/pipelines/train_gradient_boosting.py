from pathlib import Path
import joblib
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
import numpy as np
from datetime import datetime

from app.db.mongo import get_model_registry


def train_gradient_boosting(X_train, y_train, X_val, y_val, horizon: int):
    model = GradientBoostingRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=4,
        random_state=42
    )

    model.fit(X_train, y_train)

    preds = model.predict(X_val)
    mse = mean_squared_error(y_val, preds)
    rmse = float(np.sqrt(mse))

    model_dir = Path(f"models/gbr_h{horizon}")
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "model.joblib"
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
