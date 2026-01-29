from pymongo import MongoClient
import os
from datetime import datetime

# -----------------------------------
# MongoDB Connection
# -----------------------------------
MONGO_URI = os.getenv("MONGO_URI")

print("🚀 MONGO_URI from ENV =", MONGO_URI)

if not MONGO_URI:
    raise RuntimeError("❌ MONGO_URI environment variable is NOT set")

client = MongoClient(MONGO_URI)
db = client["aqi_system"]

# Collections
feature_store = db["feature_store"]
model_registry = db["model_registry"]

# -----------------------------------
# Feature Store Helpers
# -----------------------------------
def upsert_features(city: str, features: dict):
    result = feature_store.update_one(
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

    print(
        f"✅ Feature store write | city={city} | "
        f"matched={result.matched_count} | upserted={result.upserted_id}"
    )

def get_feature_freshness(city: str):
    return feature_store.find_one(
        {"city": city},
        {"_id": 0, "updated_at": 1}
    )