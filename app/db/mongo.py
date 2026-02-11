import os
import pandas as pd
from pymongo import MongoClient
from datetime import datetime

_client = None
_db = None


# -----------------------------------
# Database Connection
# -----------------------------------
def get_db():
    global _client, _db

    if _db is not None:
        return _db

    MONGODB_URI = os.getenv("MONGODB_URI")
    if not MONGODB_URI:
        raise RuntimeError("MONGODB_URI not set")

    MONGODB_URI = MONGODB_URI.strip()

    _client = MongoClient(MONGODB_URI)
    _db = _client["aqi_system"]

    return _db


# -----------------------------------
# Collection Accessors
# -----------------------------------
# -----------------------------------
# Collection Accessors
# -----------------------------------

def get_feature_store():
    return get_db()["feature_store"]


def get_model_registry():
    return get_db()["model_registry"]

def load_feature_store_df():
    collection = get_feature_store()

    data = list(collection.find({}, {"_id": 0}))

    if not data:
        raise RuntimeError("Feature store is empty")

    records = []

    for doc in data:
        record = {}

        # Add feature values
        if "features" in doc:
            record.update(doc["features"])
        else:
            record.update(doc)

        # Add datetime column
        if "updated_at" in doc:
            record["datetime"] = doc["updated_at"]

        records.append(record)

    df = pd.DataFrame(records)

    if df.empty:
        raise RuntimeError("No valid feature records found")

# ✅ ENSURE datetime column exists
    if "datetime" not in df.columns:
        df["datetime"] = pd.Timestamp.utcnow()

    df["datetime"] = pd.to_datetime(df["datetime"])

    return df
