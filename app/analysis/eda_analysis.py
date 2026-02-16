import pandas as pd
from app.db.mongo import get_db


def run_eda():

    print("📊 Running EDA...")

    db = get_db()
    collection = db["historical_hourly_data"]

    data = list(collection.find({}, {"_id": 0}))

    if not data:
        print("❌ No data found in historical_hourly_data")
        return

    df = pd.DataFrame(data)

    print("\n========== BASIC INFO ==========")
    print(df.info())

    print("\n========== SUMMARY STATS ==========")
    print(df.describe())

    print("\n========== MISSING VALUES ==========")
    print(df.isna().sum())

    print("\n========== CORRELATION ==========")
    print(df.corr(numeric_only=True)["pm2_5"].sort_values(ascending=False))


if __name__ == "__main__":
    run_eda()
