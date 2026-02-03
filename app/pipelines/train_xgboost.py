from pathlib import Path
import json
import joblib
import numpy as np

from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score

from app.pipelines.final_feature_table import build_training_dataset
from app.pipelines.register_model import register_model


def train_xgboost(horizon: int):
    print(f"🚀 Training XGBoost | horizon={horizon}")

    # 1. Load dataset
    X, y = build_training_dataset()

    # 2. Time-based split
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    # 3. Train model
    model = XGBRegressor(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    # 4. Evaluate
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    # 5. Save model
    model_dir = Path(f"models/xgb_h{horizon}")
    model_dir.mkdir(parents=True, exist_ok=True)

    model_path = model_dir / "model.joblib"
    joblib.dump(model, model_path)

    feature_path = model_dir / "features.json"
    feature_path.write_text(json.dumps(list(X.columns)))

    # 6. Register model
    register_model(
        model_name="xgboost",
        horizon=horizon,
        rmse=rmse,
        r2=r2,
        model_path=str(model_path),
        features=list(X.columns)
    )

    print(f"✅ XGBoost done | RMSE={rmse:.2f} | R²={r2:.3f}")
