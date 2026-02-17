from app.db.mongo import get_model_registry
import joblib


def select_best_model(horizon: int):

    registry = get_model_registry()

    models = list(
        registry.find({"horizon": horizon}).sort("rmse", 1)
    )

    if not models:
        raise RuntimeError("No models found")

    best_doc = models[0]

    # Reset all models
    registry.update_many(
        {"horizon": horizon},
        {"$set": {"status": "candidate", "is_best": False}}
    )

    # Set best model
    registry.update_one(
        {"_id": best_doc["_id"]},
        {"$set": {"status": "production", "is_best": True}}
    )

    model_name = best_doc["model_name"]
    rmse = best_doc["rmse"]
    mae = best_doc["mae"]

    print(f"🏆 Best model: {model_name} (RMSE={rmse:.4f})")

    # ✅ LOAD MODEL FROM STORED PATH
    model_path = best_doc["model_path"]

    model = joblib.load(model_path)

    return {
        "model_name": model_name,
        "rmse": rmse,
        "mae": mae,
        "model": model
    }
