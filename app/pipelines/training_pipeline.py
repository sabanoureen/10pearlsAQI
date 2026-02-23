# -------------------------------------------
# Run Training
# -------------------------------------------
def run_training(horizon: int):

    print("🔥 Starting Daily Training Pipeline")

    df = build_training_dataset()

    print("✅ Training dataset built:", df.shape)

    # -------------------------------------------
    # Populate Feature Store (PRODUCTION SAFE)
    # -------------------------------------------
    from app.db.mongo import get_feature_store

    feature_store = get_feature_store()

    # Remove target columns before saving
    feature_columns = [
        col for col in df.columns
        if not col.startswith("target_")
    ]

    feature_docs = df[feature_columns].to_dict(orient="records")

    # Clear old features (recommended)
    feature_store.delete_many({})

    if feature_docs:
        feature_store.insert_many(feature_docs)

    print(f"📦 Feature store populated: {len(feature_docs)} documents")

    # -------------------------------------------
    # Train model for given horizon
    # -------------------------------------------
    train_horizon(df, horizon)