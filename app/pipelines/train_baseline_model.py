"""
Select Best Model
-----------------
- Selects the best model for a given horizon
- Uses lowest RMSE as selection criteria
- Updates MongoDB:
    - is_best = True for best model
    - is_best = False for others
"""

from app.db.mongo import get_model_registry


def select_best_model(horizon: int):
    print(f"🏆 Selecting best model for horizon={horizon}")

    # 0️⃣ Get Mongo collection
    model_registry = get_model_registry()

    # 1️⃣ Fetch candidate models
    models = list(
        model_registry.find(
            {
                "horizon": horizon,
                "status": "candidate",
                "rmse": {"$exists": True}
            },
            {"_id": 1, "rmse": 1, "model_name": 1, "model_path": 1}
        )
    )

    if not models:
        raise RuntimeError(
            f"No valid candidate models found for horizon={horizon}"
        )

    # 2️⃣ Select lowest RMSE
    best_model = min(models, key=lambda m: m["rmse"])

    # 3️⃣ Archive all models
    model_registry.update_many(
        {"horizon": horizon},
        {"$set": {"is_best": False, "status": "archived"}}
    )

    # 4️⃣ Promote best model
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
    }


if __name__ == "__main__":
    select_best_model(horizon=1)
