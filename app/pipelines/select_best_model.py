from app.db.mongo import get_model_registry


def select_best_model(horizon: int):
    print(f"🏆 Selecting best model for horizon={horizon}")

    model_registry = get_model_registry()

    # Fetch only candidate models
    models = list(
        model_registry.find(
            {
                "horizon": horizon,
                "rmse": {"$exists": True},
                "status": "candidate"
            }
        )
    )

    if not models:
        raise RuntimeError(
            f"No valid candidate models found for horizon={horizon}"
        )

    # Select lowest RMSE
    best_model = min(models, key=lambda m: m["rmse"])

    # Demote all models for this horizon
    # Only demote previous production model
    model_registry.update_many(
    {"horizon": horizon, "status": "production"},
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

    return {
        "model_name": best_model["model_name"],
        "model_path": best_model["model_path"],
        "rmse": best_model["rmse"],
        "mae": best_model.get("mae")
    }
if __name__ == "__main__":
    select_best_model(horizon=1)
