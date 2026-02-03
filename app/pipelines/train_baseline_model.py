from pathlib import Path
import json
import joblib
from sklearn.linear_model import Ridge

from app.pipelines.final_feature_table import build_training_dataset


def train_and_return_model(horizon: int):
    print(f"Training model for horizon={horizon}")

    # 1. Build dataset
    X, y = build_training_dataset()

    # 2. Train model
    model = Ridge(alpha=1.0)
    model.fit(X, y)

    # 3. Save model
    model_dir = Path(f"models/ridge_h{horizon}")
    model_dir.mkdir(parents=True, exist_ok=True)

    model_path = model_dir / "model.joblib"
    joblib.dump(model, model_path)

    # 4. 🔥 Save feature order
    feature_path = model_dir / "features.json"
    feature_path.write_text(json.dumps(list(X.columns)))

    return model


# ❌ DO NOT TRAIN AT IMPORT TIME
# ❌ NO model.fit() OUTSIDE FUNCTIONS

from app.pipelines.register_model import register_model

register_model(
    model_name="ridge_regression",
    horizon=horizon,
    rmse=rmse,
    r2=r2,
    model_path=str(model_path),
    features=feature_list
)