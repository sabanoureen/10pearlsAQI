from pymongo import MongoClient
import os
from datetime import datetime

_client = None
_db = None


def get_db():
    global _client, _db

    if _db is None:
        mongo_uri = os.getenv("MONGO_URI")
        if not mongo_uri:
            raise RuntimeError("MONGO_URI not set")

        _client = MongoClient(
            mongo_uri,
            serverSelectionTimeoutMS=3000,
            connectTimeoutMS=3000,
            socketTimeoutMS=3000,
        )
        _db = _client["aqi_system"]

    return _db


# -----------------------------
# Collection getters
# -----------------------------
def get_feature_store():
    return get_db()["feature_store"]


def get_model_registry():
    return get_db()["model_registry"]


# -----------------------------
# Feature upsert helper
# -----------------------------
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
