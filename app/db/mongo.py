from pymongo import MongoClient
from datetime import datetime

client = MongoClient("mongodb+srv://saba31860_db_user:3LZwwTVy1yMFvzjD@cluster0.zpezyti.mongodb.net/aqi_system?retryWrites=true&w=majority")
db = client["aqi_system"]
registry = db["model_registry"]

features = [
    "pm10","carbon_monoxide","nitrogen_dioxide","sulphur_dioxide","ozone",
    "hour","day_of_week","day_of_month","month","week_of_year",
    "hour_sin","hour_cos","dow_sin","dow_cos",
    "pm2_5_lag_1","pm2_5_lag_6","pm2_5_lag_12","pm2_5_lag_24",
    "pm2_5_lag_48","pm2_5_lag_72","pm2_5_lag_168",
    "pm2_5_roll_mean_6","pm2_5_roll_std_6",
    "pm2_5_roll_mean_12","pm2_5_roll_std_12",
    "pm2_5_roll_mean_24","pm2_5_roll_std_24",
    "pm2_5_roll_mean_48"
]

docs = [
    {
        "model_name": "random_forest",
        "horizon": 1,
        "status": "production",
        "is_best": True,
        "model_path": "models/rf_model_h1.joblib",
        "features": features,
        "model_version": "rf_h1_v1",
        "created_at": datetime.utcnow()
    },
    {
        "model_name": "random_forest",
        "horizon": 3,
        "status": "production",
        "is_best": True,
        "model_path": "models/rf_model_h3.joblib",
        "features": features,
        "model_version": "rf_h3_v1",
        "created_at": datetime.utcnow()
    },
    {
        "model_name": "random_forest",
        "horizon": 5,
        "status": "production",
        "is_best": True,
        "model_path": "models/rf_model_h5.joblib",
        "features": features,
        "model_version": "rf_h5_v1",
        "created_at": datetime.utcnow()
    }
]

for d in docs:
    registry.update_one(
    {"horizon": 3, "is_best": True},
    {"$set": {"model_path": "models/rf_h3/model.joblib"}}
)

registry.update_one(
    {"horizon": 1, "is_best": True},
    {"$set": {"model_path": "models/rf_h1/model.joblib"}}
)

registry.update_one(
    {"horizon": 5, "is_best": True},
    {"$set": {"model_path": "models/rf_h5/model.joblib"}}
)

print("paths updated")

print("✅ Production models registered")
