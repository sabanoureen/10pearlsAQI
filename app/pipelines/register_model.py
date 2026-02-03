from datetime import datetime
from app.db.mongo import model_registry



def register_model(
    model_name,
    horizon,
    rmse,
    r2,
    model_path,
    features
):
    version = datetime.utcnow().strftime("%Y-%m-%d_%H-%M")

    # ❌ mark old best as not best
    model_registry.update_many(
        {"horizon": horizon},
        {"$set": {"is_best": False}}
    )

    # ✅ insert new model
    model_registry.insert_one({
        "model_name": model_name,
        "horizon": horizon,
        "version": version,
        "rmse": rmse,
        "r2": r2,
        "model_path": model_path,
        "features": features,
        "registered_at": datetime.utcnow(),
        "status": "production",
        "is_best": True
    })
def rollback_model(horizon: int, version: str):
    """
    Roll back production model to a previous version
    """

    # 1️⃣ Deactivate current production model(s)
    model_registry.update_many(
        {"horizon": horizon},
        {"$set": {"is_best": False, "status": "archived"}}
    )

    # 2️⃣ Activate requested version
    result = model_registry.update_one(
        {"horizon": horizon, "version": version},
        {"$set": {"is_best": True, "status": "production"}}
    )

    if result.matched_count == 0:
        raise ValueError(f"No model found for version={version}")

    print(f"✅ Rolled back horizon={horizon} to version={version}")

    # -------------------------------------------------
# CLI ENTRY POINT
# -------------------------------------------------
if __name__ == "__main__":
    import sys

    if len(sys.argv) != 4:
        print("Usage: python -m pipelines.register_model rollback <horizon> <version>")
        sys.exit(1)

    command = sys.argv[1]

    if command == "rollback":
        horizon = int(sys.argv[2])
        version = sys.argv[3]
        rollback_model(horizon, version)
    else:
        print(f"Unknown command: {command}")
        sys.exit(1)
