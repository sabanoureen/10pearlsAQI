from pymongo import MongoClient
import os
from datetime import datetime

MONGO_URI = os.getenv("MONGO_URI")
if not MONGO_URI:
    raise RuntimeError("MONGO_URI not set")

_client = MongoClient(
    MONGO_URI,
    serverSelectionTimeoutMS=3000,
    connectTimeoutMS=3000,
    socketTimeoutMS=3000,
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
                "updated_at": datetime.utcnow()
            }
        },
        upsert=True
    )
