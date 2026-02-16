"""
MongoDB connection and collection access
Production-safe for Railway deployment
"""

import os
from pymongo import MongoClient
from pymongo.errors import ServerSelectionTimeoutError

_client = None
_db = None


# ======================================================
# DB CONNECTION
# ======================================================
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

        # Force connection check
        _client.admin.command("ping")

        _db = _client[db_name]

        return _db

    except ServerSelectionTimeoutError:
        raise RuntimeError("MongoDB connection failed")


# ======================================================
# MODEL REGISTRY
# ======================================================
def get_model_registry():
    db = get_db()
    collection = db["model_registry"]

    # Partial unique index:
    # Only ONE production model per horizon
    collection.create_index(
        [("horizon", 1)],
        unique=True,
        partialFilterExpression={"is_best": True}
    )

    return collection


# ======================================================
# FEATURE STORE
# ======================================================
def get_feature_store():
    return get_db()["feature_store"]


# ======================================================
# HISTORICAL DATA
# ======================================================
def get_historical_data():
    return get_db()["historical_hourly_data"]


# ======================================================
# DAILY FORECASTS
# ======================================================
def get_daily_forecast():
    return get_db()["daily_forecast"]
