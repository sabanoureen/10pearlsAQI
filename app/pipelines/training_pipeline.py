def run_training_pipeline(horizon: int):

    print("\n" + "=" * 70)
    print(f"🚀 STARTING TRAINING PIPELINE | Horizon = H{horizon}")
    print("=" * 70)

    run_id = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    print(f"🆔 Run ID: {run_id}")

    # ------------------------------------------------------
    # 1️⃣ Load Training Dataset
    # ------------------------------------------------------
    print("\n📊 Building training dataset...")

    X, y = build_training_dataset(horizon)

    if X is None or X.empty:
        raise RuntimeError("❌ Training dataset is empty")

    print(f"✔ Dataset loaded | rows = {len(X)}")

    # ------------------------------------------------------
    # 2️⃣ Train / Validation Split
    # ------------------------------------------------------
    split_index = int(len(X) * 0.8)

    X_train = X.iloc[:split_index]
    X_val   = X.iloc[split_index:]
    y_train = y.iloc[:split_index]
    y_val   = y.iloc[split_index:]

    print(f"✔ Train size: {len(X_train)}")
    print(f"✔ Validation size: {len(X_val)}")

    # ------------------------------------------------------
    # 3️⃣ Train All Candidate Models
    # ------------------------------------------------------
    print("\n🤖 Training candidate models...")

    train_all_models(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        horizon=horizon,
        run_id=run_id
    )

    # ------------------------------------------------------
    # 4️⃣ Select Best Model Automatically
    # ------------------------------------------------------
    print("\n🏆 Selecting best model...")

    best_model_info = select_best_model(horizon)

    print("\n🎯 PRODUCTION MODEL SELECTED")
    print(f"Model Name : {best_model_info['model_name']}")
    print(f"RMSE       : {best_model_info['rmse']}")
    print(f"MAE        : {best_model_info['mae']}")

    print("\n✅ TRAINING PIPELINE COMPLETED SUCCESSFULLY")
    print("=" * 70)

    return best_model_info
