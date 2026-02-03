from app.pipelines.train_baseline_model import train_and_return_model

if __name__ == "__main__":
    for horizon in [1, 6, 24]:
        train_and_return_model(horizon)

    print("✅ All models trained and saved")

    # pipelines/save_trained_models.py

from app.pipelines.train_baseline_model import train_and_return_model

if __name__ == "__main__":
    for horizon in [6, 24]:
        train_and_return_model(horizon)

    print("✅ Models for horizons 6 and 24 trained successfully")