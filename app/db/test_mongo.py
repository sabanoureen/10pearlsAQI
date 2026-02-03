# test_mongo.py
from db.mongo import upsert_features

upsert_features(
    city="Karachi",
    features={"test": "it works"}
)

print("Mongo write successful")