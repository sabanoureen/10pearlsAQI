import joblib
import io
from gridfs import GridFS
from datetime import datetime, timedelta

from app.db.mongo import get_database, get_model_registry

from app.pipelines.final_feature_table import build_training_dataset


def load_production_model(horizon):

    registry = get_model_registry()

    model_doc = registry.find_one({
        "horizon": horizon,
        "is_best": True
    })

    if not model_doc:
        raise RuntimeError(f"No production model for horizon {horizon}")

    db = get_database()

    fs = GridFS(db)

    model_bytes = fs.get(model_doc["gridfs_id"]).read()
    model = joblib.load(io.BytesIO(model_bytes))

    return model, model_doc["features"], model_doc["model_name"]


from app.db.mongo import get_feature_store

def predict_next_3_days():

    results = {}

    feature_store = get_feature_store()

    # Get latest feature row
    latest_doc = feature_store.find_one(
        sort=[("timestamp", -1)]
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