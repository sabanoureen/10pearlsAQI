import joblib
from pathlib import Path
from functools import lru_cache

from app.db.mongo import get_model_registry


# ==================================================
# Resolve model file safely on Railway / Local
# ==================================================
def _resolve_model_path(path_str: str) -> Path:
    path = Path(path_str)

    # Absolute path
    if path.is_absolute() and path.exists():
        return path

    # Try cwd
    cwd_path = Path.cwd() / path
    if cwd_path.exists():
        return cwd_path

    # Try app directory
    app_path = Path(__file__).resolve().parents[2] / path
    if app_path.exists():
        return app_path

    raise RuntimeError(f"Model file not found: {path_str}")


# ==================================================
# Cached loader
# ==================================================
@lru_cache(maxsize=10)
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

    model_path_str = model_doc.get("model_path")

    if not model_path_str:
        raise RuntimeError("Model registry missing model_path")

    model_path = _resolve_model_path(model_path_str)

    model = joblib.load(model_path)

    features = model_doc.get("features")
    if not features:
        raise RuntimeError("Model registry missing features")

    model_version = model_doc.get(
        "model_version",
        f"{model_doc.get('model_name','model')}_h{horizon}"
    )

    return model, features, model_version
    print("Loading model from:", model_doc["model_path"])
