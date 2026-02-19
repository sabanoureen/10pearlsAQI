import os
from pymongo import MongoClient

# -----------------------------------------
# MongoDB Connection
# -----------------------------------------
MONGO_URI = os.getenv("MONGODB_URI")

if not MONGO_URI:
    raise RuntimeError("❌ MONGODB_URI not set")

client = MongoClient(MONGO_URI)

# -----------------------------------------
# Database
# -----------------------------------------
DATABASE_NAME = "aqi_system"
db = client[DATABASE_NAME]


# -----------------------------------------
# Generic Database Getter
# -----------------------------------------
def get_database():
    return db


# -----------------------------------------
# Collections
# -----------------------------------------
def get_model_registry():
    return db["model_registry"]


def get_feature_store():
    return db["feature_store"]


def get_daily_forecast():
    return db["daily_forecast"]
