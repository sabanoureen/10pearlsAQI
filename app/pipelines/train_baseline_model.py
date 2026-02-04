"""
Select Best Model
-----------------
- Selects the best model for a given horizon
- Uses lowest RMSE as selection criteria
- Updates MongoDB:
    - is_best = True for best model
    - is_best = False for others
"""

from app.db.mongo import model_registry


def select_best_model(horizon: int):
    print(f"🏆 Selecting best model for horizon={horizon}")

    # -------------------------------------------------
    # 1️⃣ Fetch all models for this horizon
    # -------------------------------------------------
    models = list(
        model_registry.find(
            {"horizon": horizon},
            {"_id": 1, "rmse": 1, "model_name": 1}
        )
    )

    if not models:
        raise RuntimeError(f"No models found for horizon={horizon}")

    # -------------------------------------------------
    # 2️⃣ Filter models that actually have RMSE
    # -------------------------------------------------
    valid_models = [m for m in models if "rmse" in m]

    if not valid_models:
        raise RuntimeError(f"No models with RMSE found for horizon={horizon}")

    # -------------------------------------------------
    # 3️⃣ Pick model with lowest RMSE
    # -------------------------------------------------
    best_model = min(valid_models, key=lambda m: m["rmse"])

    # -------------------------------------------------
    # 4️⃣ Mark all models as not best
    # -------------------------------------------------
    model_registry.update_many(
        {"horizon": horizon},
        {"$set": {"is_best": False}}
    )

    # -------------------------------------------------
    # 5️⃣ Mark selected model as best
    # -------------------------------------------------
    model_registry.update_one(
        {"_id": best_model["_id"]},
        {"$set": {"is_best": True, "status": "production"}}
    )

    print(
        f"✅ Best model selected: {best_model['model_name']} "
        f"(RMSE={best_model['rmse']:.2f})"
    )

    return best_model
