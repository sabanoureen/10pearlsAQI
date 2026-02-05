import os
from pymongo import MongoClient
from datetime import datetime

_client = None
_db = None


def get_db():
    global _client, _db

    if _db is not None:
        return _db

    mongo_uri = os.getenv("MONGODB_URI")
    if not mongo_uri:
        raise RuntimeError("MONGODB_URI not set")

    _client = MongoClient(
        mongo_uri,
        serverSelectionTimeoutMS=3000,
        connectTimeoutMS=3000,
        socketTimeoutMS=3000,
    )

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
