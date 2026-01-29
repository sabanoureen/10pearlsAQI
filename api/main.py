from fastapi import FastAPI
from datetime import datetime, timezone
from db.mongo import feature_store
from db.mongo import get_feature_freshness

@app.get("/features/freshness")
def feature_freshness(city: str = "Karachi"):
    try:
        doc = feature_store.find_one(
            {"city": city},
            {"_id": 0, "city": 1, "updated_at": 1}
        )

        if not doc or "updated_at" not in doc:
            return {
                "status": "no_data",
                "message": f"No feature timestamp found for city={city}"
            }

        updated_at = doc["updated_at"]

        # 🔒 Ensure timezone safety
        if updated_at.tzinfo is None:
            updated_at = updated_at.replace(tzinfo=timezone.utc)

        now = datetime.now(timezone.utc)
        age_minutes = round(
            (now - updated_at).total_seconds() / 60, 2
        )

        return {
            "status": "ok",
            "city": city,
            "updated_at": updated_at.isoformat(),
            "age_minutes": age_minutes
        }

    except Exception as e:
        return {
            "status": "error",
            "error": str(e)
        }