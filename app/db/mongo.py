import os
from pymongo import MongoClient
from datetime import datetime

_client = None
_db = None


def get_db():
    global _client, _db

    if _db is not None:
        return _db

    MONGODB_URI = os.getenv("MONGODB_URI")
    if not MONGODB_URI:
        raise RuntimeError("MONGODB_URI not set")

    _client = MongoClient(MONGODB_URI)
    _db = _client["aqi_system"]
    return _db


def get_feature_store():
    return get_db()["feature_store"]


def get_model_registry():
    return get_db()["model_registry"]


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
