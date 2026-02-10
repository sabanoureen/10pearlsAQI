from app.db.mongo import get_model_registry

registry = get_model_registry()
result = registry.delete_many({"rmse": {"$exists": False}})
print(f"Deleted {result.deleted_count} invalid model records")
