from datetime import datetime
from app.db.mongo import get_model_registry



def register_model(
    model_name: str,
    horizon: int,
    rmse: float,
    r2: float,
    model_path: str,
    features: list,
):
    version = datetime.utcnow().strftime("%Y-%m-%d_%H-%M-%S")

    # 1️⃣ Deactivate old best models
    model_registry.update_many(
        {"horizon": horizon},
        {"$set": {"is_best": False}}
    )

    # 2️⃣ Insert new versioned model
    get_model_registry().insert_one(doc)
    ({
        "model_name": model_name,
        "horizon": horizon,
        "version": version,              # ✅ IMPORTANT
        "rmse": rmse,
        "r2": r2,
        "model_path": model_path,
        "features": features,
        "registered_at": datetime.utcnow(),
        "status": "production",
        "is_best": True
    })

    print(f"📦 Registered {model_name} | horizon={horizon} | version={version}")
