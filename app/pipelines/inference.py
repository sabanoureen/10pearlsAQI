import io
import joblib
from gridfs import GridFS
from app.db.mongo import get_model_registry, get_database


def load_production_model(horizon: int):

    registry = get_model_registry()

    model_doc = registry.find_one({
        "horizon": horizon,
        "status": "production",
        "is_best": True
    })

    if not model_doc:
        raise RuntimeError(f"No production model found for horizon {horizon}")

    if "gridfs_id" not in model_doc:
        raise RuntimeError("Model does not contain gridfs_id")

    db = get_database()
    fs = GridFS(db)

    model_bytes = fs.get(model_doc["gridfs_id"]).read()
    model = joblib.load(io.BytesIO(model_bytes))

    return model, model_doc["features"], model_doc["model_name"]
