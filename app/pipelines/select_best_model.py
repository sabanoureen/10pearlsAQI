from db.mongo import model_registry


def select_best_model(horizon: int):
    print(f"🏆 Selecting best model for horizon={horizon}")

    models = list(
        model_registry.find(
            {
                "horizon": horizon,
                "rmse": {"$exists": True}
            }
        )
    )

    if not models:
        raise RuntimeError(f"No models with RMSE found for horizon={horizon}")

    # Pick model with lowest RMSE
    best = min(models, key=lambda x: x["rmse"])

    # Reset all models for this horizon
    model_registry.update_many(
        {"horizon": horizon},
        {"$set": {"is_best": False}}
    )

    # Mark best model
    model_registry.update_one(
        {"_id": best["_id"]},
        {"$set": {"is_best": True}}
    )

    print(
        f"✅ Best model selected: {best['model_name']} "
        f"(RMSE={best['rmse']:.2f})"
    )
