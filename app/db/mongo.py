import os
from pymongo import MongoClient
from pymongo.errors import ServerSelectionTimeoutError

_client = None
_db = None


def get_db():
    global _client, _db

    if _db is not None:
        return _db

    mongo_uri = os.getenv("MONGODB_URI")
    db_name = os.getenv("MONGODB_DB", "aqi_system")

    if not mongo_uri:
        raise RuntimeError("MONGODB_URI not set")

    try:
        _client = MongoClient(
            mongo_uri,
            serverSelectionTimeoutMS=5000
        )

        # 🔎 Force connection check
        _client.admin.command("ping")

        _db = _client[db_name]

        return _db

    except ServerSelectionTimeoutError:
        raise RuntimeError("MongoDB connection failed")


# --------------------------------------------------
# Collections
# --------------------------------------------------

def get_model_registry():
    db = get_db()
    collection = db["model_registry"]

    # Ensure index (one best model per horizon)
    collection.create_index(
        [("horizon", 1), ("is_best", 1)],
        unique=True
    )

    return collection


def get_feature_store():
    return get_db()["feature_store"]


def get_historical_data():
    return get_db()["historical_hourly_data"]
