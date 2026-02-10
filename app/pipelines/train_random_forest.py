from pathlib import Path
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import numpy as np
from datetime import datetime
from app.db.mongo import get_model_registry


def train_random_forest(X_train, y_train, X_val, y_val, horizon: int):
    model = RandomForestRegressor(
        n_estimators=300,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    preds = model.predict(X_val)
    mse = mean_squared_error(y_val, preds)
    rmse = float(np.sqrt(mse))

    # 📁 Save model
    model_dir = Path(f"models/rf_h{horizon}")
    model_dir.mkdir(parents=True, exist_ok=True)
    from pathlib import Path

    model_dir = Path("models") / f"gbr_h{horizon}"
    model_dir.mkdir(parents=True, exist_ok=True)

    model_path = model_dir / "model.joblib"
    joblib.dump(model, model_path)

    model_path_str = model_path.as_posix()

    joblib.dump(model, model_path)

    # 🧾 Register in MongoDB
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
