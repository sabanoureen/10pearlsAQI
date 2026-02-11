import joblib
from app.db.mongo import get_model_registry


def load_production_model(horizon: int):

    registry = get_model_registry()

    model_doc = registry.find_one(
        {"horizon": horizon, "is_best": True}
    )

    if not model_doc:
        raise RuntimeError("No production model found")

    print(f"🚀 Loading production model: {model_doc['model_name']}")
    print(f"📁 Path: {model_doc['model_path']}")

    model = joblib.load(model_doc["model_path"])

    return model, model_doc["features"]
