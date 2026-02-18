import os
from pymongo import MongoClient
from gridfs import GridFS


# -------------------------------------------------
# Environment Variables
# -------------------------------------------------
MONGODB_URI = os.getenv("MONGODB_URI")
DATABASE_NAME = os.getenv("DATABASE_NAME", "aqi_system")

if not MONGODB_URI:
    raise ValueError("❌ MONGODB_URI not set")

# -------------------------------------------------
# Mongo Client
# -------------------------------------------------
client = MongoClient(MONGODB_URI)
db = client[DATABASE_NAME]

# -------------------------------------------------
# GridFS for Model Storage
# -------------------------------------------------
fs = GridFS(db)


# -------------------------------------------------
# Helper Functions
# -------------------------------------------------
def get_db():
    return db


def get_model_registry():
    return db["model_registry"]


def get_feature_store():
    return db["feature_store"]


def get_fs():
    return fs
