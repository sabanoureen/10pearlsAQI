def run_training_pipeline(horizon: int = 1):

    try:
        print("\n🚀 Starting training pipeline")
        print(f"📌 Forecast horizon: {horizon} day(s)\n")

        # 1️⃣ Build dataset
        X, y = build_training_dataset(horizon)

        if X.empty or y.empty:
            raise RuntimeError("Training dataset is empty")

        print(f"📊 Dataset size: {X.shape[0]} rows")

        # 2️⃣ Clean old models
        registry = get_model_registry()
        registry.delete_many({"horizon": horizon})
        print("🧹 Old models deleted")

        # 3️⃣ Split
        split_idx = int(len(X) * 0.8)

        X_train = X.iloc[:split_idx]
        X_val   = X.iloc[split_idx:]
        y_train = y.iloc[:split_idx]
        y_val   = y.iloc[split_idx:]

        # 4️⃣ Train models
        rf_model, _ = train_random_forest(
            X_train, y_train, X_val, y_val, horizon
        )

        xgb_model, _ = train_xgboost(
            X_train, y_train, X_val, y_val, horizon
        )

        gb_model, _ = train_gradient_boosting(
            X_train, y_train, X_val, y_val, horizon
        )

        ensemble_model, _ = train_ensemble(
            rf_model=rf_model,
            xgb_model=xgb_model,
            gb_model=gb_model,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            horizon=horizon
        )

        # 5️⃣ Select best model
        best_model_info = select_best_model(horizon)

        print(f"\n🎯 Production Model: {best_model_info['model_name']}")
        print("✅ Training pipeline completed")

    except Exception as e:
        print("\n❌ TRAINING FAILED")
        print(str(e))
        raise
