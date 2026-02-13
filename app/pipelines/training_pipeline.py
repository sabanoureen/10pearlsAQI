from datetime import datetime

def run_training_pipeline(horizon: int = 1):

    try:
        # ✅ DEFINE run_id HERE
        run_id = datetime.utcnow().strftime("%Y%m%d_%H%M%S")

        print(f"\n🆔 Training run_id = {run_id}")
        print("🚀 Starting training pipeline")
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

        # 4️⃣ Train models (PASS run_id)
        rf_model, _ = train_random_forest(
            X_train, y_train, X_val, y_val, horizon, run_id
        )

        xgb_model, _ = train_xgboost(
            X_train, y_train, X_val, y_val, horizon, run_id
        )

        gb_model, _ = train_gradient_boosting(
            X_train, y_train, X_val, y_val, horizon, run_id
        )

        ensemble_model, _ = train_ensemble(
            rf_model,
            xgb_model,
            gb_model,
            X_train,
            y_train,
            X_val,
            y_val,
            horizon,
            run_id
        )

        best_model_info = select_best_model(horizon)

        print(f"\n🎯 Production Model: {best_model_info['model_name']}")
        print("✅ Training pipeline completed")

    except Exception as e:
        print("\n❌ TRAINING FAILED")
        print(str(e))
        raise
