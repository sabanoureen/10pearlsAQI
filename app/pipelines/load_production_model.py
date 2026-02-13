import joblib
from pathlib import Path
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
    print("PRODUCTION MODEL DOC:", model_doc) 

    model_path = Path(model_doc["model_path"])

    # Make absolute path
    if not model_path.is_absolute():
        model_path = Path.cwd() / model_path

    if not model_path.exists():
        raise RuntimeError(
            f"Model file not found on server: {model_path}"
        )

    model = joblib.load(model_path)

    features = model_doc.get("features")
    model_version = model_doc.get("model_version", "production_v1")

    if not features:
        raise RuntimeError(
            "Production model has no feature list saved."
        )

    return model, features, model_version
