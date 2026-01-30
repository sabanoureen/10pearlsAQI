from datetime import datetime
from ..db.mongo import model_registry


def register_model(
    model_name,
    horizon,
    rmse,
    r2,
    model_path,
    features
):
    model_registry.insert_one({
        "model_name": model_name,
        "horizon": horizon,
        "rmse": rmse,
        "r2": r2,
        "model_path": model_path,
        "features": features,
        "registered_at": datetime.utcnow(),
        "status": "production"
    })