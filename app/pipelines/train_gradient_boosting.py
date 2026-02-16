from pathlib import Path
import joblib
import numpy as np
from datetime import datetime
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from app.db.mongo import get_model_registry


def train_gradient_boosting(
    X_train,
    y_train,
    X_val,
    y_val,
    horizon: int,
    run_id: str
):

    print("🌊 Training Gradient Boosting...")

    model = GradientBoostingRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=3,
        random_state=42
    )

    model.fit(X_train, y_train)

    # Predictions (log scale)
    preds_log = model.predict(X_val)

# Convert back to original scale
    preds = np.expm1(preds_log)
    y_true = np.expm1(y_val)

    rmse = float(np.sqrt(mean_squared_error(y_true, preds)))
    mae = float(mean_absolute_error(y_true, preds))
    r2 = float(r2_score(y_true, preds))

    print(f"RMSE: {rmse:.4f}")
    print(f"MAE : {mae:.4f}")
    print(f"R2  : {r2:.4f}")

    # ===============================
    # 🔎 Feature Importance (SAFE)
    # ===============================
    print("\n🔎 Top 10 Important Features:")

    importances = model.feature_importances_
    feature_names = X_train.columns

    sorted_idx = np.argsort(importances)[::-1]

    for i in sorted_idx[:10]:
        print(f"{feature_names[i]} → {importances[i]:.4f}")

    # ===============================
    # Save Model
    # ===============================
    model_dir = Path(f"models/gbr_h{horizon}")
    model_dir.mkdir(parents=True, exist_ok=True)

    model_path = model_dir / f"model_{run_id}.joblib"

    joblib.dump(model, model_path)

    # ===============================
    # Register Metadata
    # ===============================
    registry = get_model_registry()

    registry.insert_one({
        "model_name": "gradient_boosting",
        "horizon": horizon,
        "rmse": rmse,
        "mae": mae,
        "model_path": str(model_path),
        "features": list(X_train.columns),
        "status": "candidate",
        "is_best": False,
        "registered_at": datetime.utcnow()
    })

    print("✅ Gradient Boosting registered")

    return model, {"rmse": rmse, "mae": mae}
