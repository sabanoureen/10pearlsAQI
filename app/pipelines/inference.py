import io
import joblib
from gridfs import GridFS
from datetime import datetime, timedelta

from app.db.mongo import (
    get_model_registry,
    get_database,
    get_feature_store
)


# -------------------------------------------------------
# Load Production Model
# -------------------------------------------------------
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


# -------------------------------------------------------
# Predict Next 3 Days
# -------------------------------------------------------
def predict_next_3_days():

    results = {}

    feature_store = get_feature_store()

    latest_doc = feature_store.find_one(
        sort=[("datetime", -1)]
    )

    if not latest_doc:
        raise RuntimeError("No feature data available")

    for horizon in [1, 2, 3]:

        model, features, model_name = load_production_model(horizon)

        latest_row = [latest_doc[f] for f in features]

        prediction = model.predict([latest_row])[0]

        future_date = (
            datetime.utcnow() + timedelta(days=horizon)
        ).strftime("%Y-%m-%d")

        results[f"{horizon}_day"] = {
            "value": round(float(prediction), 2),
            "date": future_date,
            "model": model_name
        }

    return results