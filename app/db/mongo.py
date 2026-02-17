import os
from pymongo import MongoClient

MONGODB_URI = os.getenv("MONGODB_URI")
DATABASE_NAME = os.getenv("DATABASE_NAME", "aqi_system")

client = MongoClient(MONGODB_URI)
db = client[DATABASE_NAME]

def get_feature_store():
    return db["feature_store"]

def get_model_registry():
    return db["model_registry"]
