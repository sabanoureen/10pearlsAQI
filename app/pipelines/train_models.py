"""
Train All Candidate Models
--------------------------
- Random Forest
- Gradient Boosting
- XGBoost
- Registers models in MongoDB
"""

import joblib
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import xgboost as xgb
import numpy as np

from app.db.mongo import get_model_registry


# ==========================================================
# Utility: Evaluate Model
# ==========================================================
def evaluate_model(model, X_val, y_val):

    preds = model.predict(X_val)

    rmse = np.sqrt(mean_squared_error(y_val, preds))
    mae = mean_absolute_error(y_val, preds)

    return rmse, mae


# ==========================================================
# Register Model in MongoDB
# ==========================================================
def register_model(model, model_name, rmse, mae, horizon, run_id):

    registry = get_model_registry()

    registry.insert_one({
        "model_name": model_name,
        "horizon": horizon,
        "run_id": run_id,
        "rmse": float(rmse),
        "mae": float(mae),
        "is_best": False
    })


# ==========================================================
# MAIN FUNCTION (THIS WAS MISSING)
# ==========================================================
def train_all_models(X_train, y_train, X_val, y_val, horizon, run_id):

    print("\n🔵 Training Random Forest...")
    rf = RandomForestRegressor(
        n_estimators=200,
        max_depth=10,
        random_state=42
    )
    rf.fit(X_train, y_train)
    rf_rmse, rf_mae = evaluate_model(rf, X_val, y_val)
    print(f"RF RMSE: {rf_rmse:.4f}")
    register_model(rf, "RandomForest", rf_rmse, rf_mae, horizon, run_id)


    print("\n🟢 Training Gradient Boosting...")
    gb = GradientBoostingRegressor(
        n_estimators=200,
        learning_rate=0.05,
        random_state=42
    )
    gb.fit(X_train, y_train)
    gb_rmse, gb_mae = evaluate_model(gb, X_val, y_val)
    print(f"GB RMSE: {gb_rmse:.4f}")
    register_model(gb, "GradientBoosting", gb_rmse, gb_mae, horizon, run_id)


    print("\n🟣 Training XGBoost...")
    xgb_model = xgb.XGBRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=6,
        random_state=42
    )
    xgb_model.fit(X_train, y_train)
    xgb_rmse, xgb_mae = evaluate_model(xgb_model, X_val, y_val)
    print(f"XGB RMSE: {xgb_rmse:.4f}")
    register_model(xgb_model, "XGBoost", xgb_rmse, xgb_mae, horizon, run_id)


    print("\n✅ All candidate models trained & registered")
