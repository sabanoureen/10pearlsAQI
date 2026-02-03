from pathlib import Path
import json
import joblib
import numpy as np

from sklearn.ensemble import VotingRegressor
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor
from sklearn.ensemble import GradientBoostingRegressor

from app.pipelines.final_feature_table import build_training_dataset
from app.pipelines.register_model import register_model


def train_ensemble(horizon: int):
    print(f"🤝 Training Ensemble Model | horizon={horizon}")

    # 1. Load dataset
    X, y = build_training_dataset()

    # 2. Time-based split
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    # 3. Define base models
    ensemble = VotingRegressor(
        estimators=[
            ("ridge", Ridge(alpha=1.0)),
            ("rf", RandomForestRegressor(n_estimators=200, random_state=42)),
            ("xgb", XGBRegressor(n_estimators=200, random_state=42)),
            ("gbr", GradientBoostingRegressor(n_estimators=300, random_state=42)),
        ]
    )

    # 4. Train ensemble
    ensemble.fit(X_train, y_train)

    # 5. Evaluate
    y_pred = ensemble.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    # 6. Save model
    model_dir = Path(f"models/ensemble_h{horizon}")
    model_dir.mkdir(parents=True, exist_ok=True)

    model_path = model_dir / "model.joblib"
    joblib.dump(ensemble, model_path)

    feature_path = model_dir / "features.json"
    feature_path.write_text(json.dumps(list(X.columns)))

    # 7. Register in MongoDB
    register_model(
        model_name="ensemble",
        horizon=horizon,
        rmse=rmse,
        r2=r2,
        model_path=str(model_path),
        features=list(X.columns)
    )

    print(f"✅ Ensemble done | RMSE={rmse:.2f} | R²={r2:.3f}")
