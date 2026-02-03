from datetime import datetime
from db.mongo import feature_collection

def save_features(city, features: dict):
    doc = {
        "city": city,
        "timestamp": datetime.utcnow(),
        "features": features
    }
    feature_collection.insert_one(doc)
from db.mongo import feature_store
from datetime import datetime

feature_store.insert_one({
    "city": "Karachi",
    "features": X_last.to_dict(),
    "timestamp": datetime.utcnow()
})