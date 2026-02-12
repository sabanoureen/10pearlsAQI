from app.db.mongo import get_model_registry


def select_best_model(horizon: int):
    print(f"🏆 Selecting best model for horizon={horizon}")

    model_registry = get_model_registry()

    # -----------------------------------------
    # 1️⃣ Try to fetch candidate models
    # -----------------------------------------
    candidates = list(
        model_registry.find(
            {
                "horizon": horizon,
                "rmse": {"$exists": True},
                "status": "candidate"
            }
        )
    )

    # -----------------------------------------
    # 2️⃣ If candidates exist → promote best
    # -----------------------------------------
    if candidates:

        best_model = min(candidates, key=lambda m: m["rmse"])

        # Demote previous production
        model_registry.update_many(
            {"horizon": horizon, "status": "production"},
            {"$set": {"is_best": False, "status": "archived"}}
        )

        # Promote best candidate
        model_registry.update_one(
            {"_id": best_model["_id"]},
            {"$set": {"is_best": True, "status": "production"}}
        )

        print(
            f"✅ Promoted best candidate: {best_model['model_name']} "
            f"(RMSE={best_model['rmse']:.2f})"
        )

    # -----------------------------------------
    # 3️⃣ If no candidates → return production
    # -----------------------------------------
    else:
        best_model = model_registry.find_one(
            {
                "horizon": horizon,
                "status": "production",
                "is_best": True
            }
        )

        if not best_model:
            raise RuntimeError(
                f"No production model found for horizon={horizon}"
            )

        print("ℹ️ No new candidates. Using existing production model.")

    # -----------------------------------------
    # 4️⃣ Return model info
    # -----------------------------------------
    return {
        "model_name": best_model["model_name"],
        "model_path": best_model["model_path"],
        "rmse": best_model["rmse"],
        "mae": best_model.get("mae"),
        "features": best_model.get("features")
    }


if __name__ == "__main__":
    select_best_model(horizon=1)
