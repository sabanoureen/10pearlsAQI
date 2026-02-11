import os
import pandas as pd
from pymongo import MongoClient
from datetime import datetime

_client = None
_db = None


# -----------------------------------
# Database Connection
# -----------------------------------
def get_db():
    global _client, _db

    if _db is not None:
        return _db

    MONGODB_URI = os.getenv("MONGODB_URI")
    if not MONGODB_URI:
        raise RuntimeError("MONGODB_URI not set")

    MONGODB_URI = MONGODB_URI.strip()

    _client = MongoClient(MONGODB_URI)
    _db = _client["aqi_system"]

    return _db


# -----------------------------------
# Collection Accessors
# -----------------------------------
def get_feature_store():
    return get_db()["feature_store"]


def get_model_registry():
    return get_db()["model_registry"]


# -----------------------------------
# Load Feature Store as DataFrame
# -----------------------------------
def load_feature_store_df():
    collection = get_feature_store()

    # Fetch all documents
    data = list(collection.find({}, {"_id": 0}))

    if not data:
        raise RuntimeError("Feature store is empty")

    # If features are nested inside "features"
    if "features" in data[0]:
        records = [doc["features"] for doc in data if "features" in doc]
        df = pd.DataFrame(records)
    else:
        df = pd.DataFrame(data)

    if df.empty:
        raise RuntimeError("No valid feature records found")

    return df


# -----------------------------------
# Upsert Features
# -----------------------------------
def upsert_features(city: str, features: dict):
    get_feature_store().update_one(
        {"city": city},
        {
            "$set": {
                "city": city,
                "features": features,
                "updated_at": datetime.utcnow(),
            }
        },
        upsert=True,
    )
