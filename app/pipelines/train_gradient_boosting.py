import pickle
import numpy as np
from datetime import datetime
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error

from app.db.mongo import get_model_registry


def train_gradient_boosting(X_train, y_train, X_val, y_val, horizon: int, run_id: str
):

    print("🌊 Training Gradient Boosting...")

    model = GradientBoostingRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=3,
        random_state=42
    )

    model.fit(X_train, y_train)

    preds = model.predict(X_val)

    rmse = float(np.sqrt(mean_squared_error(y_val, preds)))
    mae = float(mean_absolute_error(y_val, preds))

    print(f"GB RMSE: {rmse:.4f}")
    print(f"GB MAE : {mae:.4f}")

    model_binary = pickle.dumps(model)

    registry = get_model_registry()

    registry.insert_one({
        "model_name": "gradient_boosting",
        "horizon": horizon,
        "rmse": rmse,
        "mae": mae,
        "model_binary": model_binary,
        "features": list(X_train.columns),
        "status": "candidate",
        "is_best": False,
        "registered_at": datetime.utcnow()
    })

    print("✅ Gradient Boosting stored in MongoDB")

    return model, {"rmse": rmse, "mae": mae}
