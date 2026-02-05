from pymongo import MongoClient
import os

MONGO_URI = os.getenv("MONGO_URI")
if not MONGO_URI:
    raise RuntimeError("MONGO_URI not set")

client = MongoClient(
    MONGO_URI,
    serverSelectionTimeoutMS=3000,
    connectTimeoutMS=3000,
    socketTimeoutMS=3000,
)

db = client["aqi_system"]

# collections
_feature_store = db["feature_store"]
_model_registry = db["model_registry"]

# -----------------------
# getters (IMPORTANT)
# -----------------------
def get_feature_store():
    return _feature_store

def get_model_registry():
    return _model_registry
