from app.db.mongo import get_model_registry


def select_best_model(horizon: int):

    registry = get_model_registry()

    models = list(
        registry.find({"horizon": horizon}).sort("rmse", 1)
    )

    if not models:
        raise RuntimeError("No models found")

    best_model = models[0]

    # Reset all models
    registry.update_many(
        {"horizon": horizon},
        {"$set": {"status": "candidate", "is_best": False}}
    )

    # Set best model
    registry.update_one(
        {"_id": best_model["_id"]},
        {"$set": {"status": "production", "is_best": True}}
    )

    print(f"🏆 Best model: {best_model['model_name']} (RMSE={best_model['rmse']:.4f})")

    return best_model
