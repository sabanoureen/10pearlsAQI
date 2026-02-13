from app.db.mongo import get_model_registry


def select_best_model(horizon: int):

    registry = get_model_registry()

    models = list(registry.find({"horizon": horizon}))

    if not models:
        raise RuntimeError("No models found")

    best = min(models, key=lambda x: x["rmse"])

    registry.update_many(
        {"horizon": horizon},
        {"$set": {"is_best": False}}
    )

    registry.update_one(
        {"_id": best["_id"]},
        {"$set": {"is_best": True, "status": "production"}}
    )

    print(f"🏆 Best model: {best['model_name']} (RMSE={best['rmse']:.4f})")

    return best
