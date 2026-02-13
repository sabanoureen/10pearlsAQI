from pathlib import Path
import joblib
import numpy as np
from datetime import datetime

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from app.db.mongo import get_model_registry


def train_random_forest(X_train, y_train, X_val, y_val, horizon: int):

    print("🌲 Training Random Forest...")

    # -----------------------------
    # 1️⃣ Model Initialization
    # -----------------------------
    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=None,
        random_state=42,
        n_jobs=-1
    )

    # -----------------------------
    # 2️⃣ Train Model
    # -----------------------------
    model.fit(X_train, y_train)

    # -----------------------------
    # 3️⃣ Validation Evaluation
    # -----------------------------
    preds = model.predict(X_val)

    rmse = float(np.sqrt(mean_squared_error(y_val, preds)))
    mae = float(mean_absolute_error(y_val, preds))
    r2 = float(r2_score(y_val, preds))

    print(f"RF RMSE: {rmse:.4f}")
    print(f"RF MAE : {mae:.4f}")
    print(f"RF R²  : {r2:.4f}")

    # -----------------------------
    # 4️⃣ Save Model (Stable Path)
    # -----------------------------
    model_dir = Path(f"models/rf_h{horizon}")
    model_dir.mkdir(parents=True, exist_ok=True)

    model_path = model_dir / "model.joblib"

    joblib.dump(model, model_path)

    print(f"✅ Model saved to: {model_path}")

    # -----------------------------
    # 5️⃣ Register in MongoDB
    # -----------------------------
    registry = get_model_registry()

    registry.insert_one({
        "model_name": "random_forest",
        "horizon": horizon,
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "model_path": str(model_path),
        "features": list(X_train.columns),
        "status": "registered",
        "is_best": False,
        "registered_at": datetime.utcnow()
    })

    print("✅ Random Forest registered in MongoDB")

    return model, {
        "rmse": rmse,
        "mae": mae,
        "r2": r2
    }
