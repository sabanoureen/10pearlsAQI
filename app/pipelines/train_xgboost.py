import pickle
import numpy as np
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from app.db.mongo import get_model_registry


def train_random_forest(X_train, y_train, X_val, y_val, horizon: int):

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

    print(f"RF RMSE: {rmse:.4f}")
    print(f"RF MAE : {mae:.4f}")
    print(f"RF R²  : {r2:.4f}")

    # 🔥 Serialize model
    model_binary = pickle.dumps(model)

    registry = get_model_registry()

    registry.insert_one({
        "model_name": "random_forest",
        "horizon": horizon,
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "model_binary": model_binary,  # 🔥 STORED IN MONGO
        "features": list(X_train.columns),
        "status": "candidate",
        "is_best": False,
        "registered_at": datetime.utcnow()
    })

    print("✅ Random Forest stored in MongoDB")

    return model, {"rmse": rmse, "mae": mae, "r2": r2}
