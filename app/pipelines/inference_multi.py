import io
import joblib
from datetime import datetime, timedelta

from app.db.mongo import get_model_registry, get_fs
from app.pipelines.training_dataset import build_training_dataset


def load_production_model(horizon: int):
    registry = get_model_registry()

    model_doc = registry.find_one({
        "horizon": horizon,
        "is_best": True
    })

    if not model_doc:
        raise RuntimeError(f"No production model found for horizon {horizon}")

    model_path = model_doc["model_path"]

    model = joblib.load(model_path)

    features = model_doc.get("features", [])

    return model, features, model_doc["model_name"]

def predict_next_3_days():

    results = {}
    today = datetime.utcnow().date()

    for horizon in [1, 2, 3]:

        model, features, model_name = load_production_model(horizon)

        X, _ = build_training_dataset(horizon)
        X_latest = X.tail(1)
        X_latest = X_latest[features]

        prediction = model.predict(X_latest)[0]

        forecast_date = today + timedelta(days=horizon)

        results[f"{horizon}_day"] = {
            "date": str(forecast_date),
            "value": round(float(prediction), 2),
            "model": model_name
        }

    return results


if __name__ == "__main__":
    print(predict_next_3_days())
