from datetime import datetime
import numpy as np
import pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from app.db.mongo import get_model_registry


def train_random_forest(X_train, y_train, X_val, y_val, horizon: int, run_id: str):

    print("🌲 Training Random Forest...")

    model = RandomForestRegressor(
        n_estimators=300,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    preds = model.predict(X_val)

    rmse = float(np.sqrt(mean_squared_error(y_val, preds)))
    mae = float(mean_absolute_error(y_val, preds))
    r2 = float(r2_score(y_val, preds))

    print(f"RMSE: {rmse:.4f}")

    # 🔥 Convert model to binary for Mongo storage
    model_binary = pickle.dumps(model)

    # 🔥 NOW get registry (INSIDE FUNCTION)
    registry = get_model_registry()

    result = registry.insert_one({
        "model_name": "random_forest",
        "run_id": run_id,
        "horizon": horizon,
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "model_path": str(model_path), 
        "features": list(X_train.columns),
        "status": "candidate",
        "is_best": False,
        "registered_at": datetime.utcnow()
    })

    print("✅ Model stored in Mongo:", result.inserted_id)

    return model, {"rmse": rmse, "mae": mae, "r2": r2}
