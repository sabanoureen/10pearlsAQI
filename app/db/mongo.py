import os
from datetime import datetime
from pymongo import MongoClient
from pymongo.errors import PyMongoError

# -----------------------------
# MongoDB connection
# -----------------------------

MONGODB_URI = os.getenv("MONGODB_URI")

if not MONGODB_URI:
    raise RuntimeError("MONGODB_URI not set")

_client = MongoClient(
    MONGODB_URI,
    serverSelectionTimeoutMS=3000,
    connectTimeoutMS=3000,
    socketTimeoutMS=3000,
)

_db = _client["aqi_system"]


# -----------------------------
# DB getter
# -----------------------------
def get_db():
    try:
        # Force connection check
        _client.admin.command("ping")
        return _db
    except PyMongoError as e:
        raise RuntimeError(f"MongoDB connection failed: {e}")


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
