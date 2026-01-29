from pymongo import MongoClient
import os
from datetime import datetime

MONGO_URI = os.getenv("MONGO_URI")
if not MONGO_URI:
    raise RuntimeError("MONGO_URI not set")

client = MongoClient(MONGO_URI)
db = client["aqi_system"]

feature_store = db["feature_store"]
model_registry = db["model_registry"]

def upsert_features(city: str, features: dict):
    feature_store.update_one(
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
    # -----------------------------------
# Feature freshness helper
# -----------------------------------
def get_feature_freshness(city: str):
    return feature_store.find_one(
        {"city": city},
        {"_id": 0, "city": 1, "updated_at": 1}
    )