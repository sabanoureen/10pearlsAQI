from app.db.mongo import get_model_registry


def select_best_model(horizon: int):
    print(f"🏆 Selecting best model for horizon={horizon}")

    model_registry = get_model_registry()

    # Only consider models that actually have rmse
    models = list(
        model_registry.find(
            {"horizon": horizon, "rmse": {"$exists": True}},
            {"_id": 1, "rmse": 1, "model_name": 1}
        )
    )

    if not models:
        raise RuntimeError(f"No valid models with rmse found for horizon={horizon}")

    # Select model with lowest RMSE
    best_model = min(models, key=lambda m: m["rmse"])

    # Demote all models for this horizon
    model_registry.update_many(
        {"horizon": horizon},
        {"$set": {"is_best": False, "status": "archived"}}
    )

    # Promote best model
    model_registry.update_one(
        {"_id": best_model["_id"]},
        {"$set": {"is_best": True, "status": "production"}}
    )

    print(
        f"✅ Best model selected: {best_model['model_name']} "
        f"(RMSE={best_model['rmse']:.2f})"
    )

    return best_model
