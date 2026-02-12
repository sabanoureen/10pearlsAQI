
import joblib
from app.db.mongo import get_model_registry


def load_production_model(horizon: int):

    registry = get_model_registry()

    model_doc = registry.find_one({
        "horizon": horizon,
        "status": "production",
        "is_best": True
    })

    if not model_doc:
        raise RuntimeError(
            f"No production model found for horizon={horizon}"
        )

    model = joblib.load(model_doc["model_path"])

    return model, model_doc["features"]
