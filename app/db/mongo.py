import os
from pymongo import MongoClient
from datetime import datetime

# -----------------------------
# Mongo connection
# -----------------------------
MONGODB_URI = os.getenv("MONGODB_URI")

if not MONGODB_URI:
    raise RuntimeError("MONGODB_URI not set")

_client = MongoClient(
    MONGODB_URI,
    serverSelectionTimeoutMS=5000,
    connectTimeoutMS=5000,
    socketTimeoutMS=5000,
)

_db = _client["aqi_system"]

# -----------------------------
# Collection getters
# -----------------------------
def get_feature_store():
    return _db["feature_store"]

def get_model_registry():
    return _db["model_registry"]

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
