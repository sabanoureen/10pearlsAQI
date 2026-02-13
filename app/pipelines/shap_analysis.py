import shap
import pandas as pd
from datetime import datetime

from app.db.mongo import get_db
from app.pipelines.load_production_model import load_production_model


def generate_shap_analysis():

    db = get_db()

    # 1️⃣ Load production model (horizon 1 only)
    model, features, model_version = load_production_model(horizon=1)

    # 2️⃣ Get latest feature row
    latest_doc = list(
        db["feature_store"]
        .find()
        .sort("datetime", -1)
        .limit(1)
    )

    if not latest_doc:
        raise RuntimeError("No feature data found for SHAP")

    latest_doc = latest_doc[0]

    X = pd.DataFrame([{
        col: latest_doc.get(col)
        for col in features
    }])

    # 3️⃣ Predict
    prediction = float(model.predict(X)[0])

    # 4️⃣ SHAP Explainer
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    contributions = []

    for feature, value in zip(features, shap_values[0]):
        contributions.append({
            "feature": feature,
            "shap_value": float(value)
        })

    # Sort by importance
    contributions = sorted(
        contributions,
        key=lambda x: abs(x["shap_value"]),
        reverse=True
    )

    return {
        "status": "success",
        "model_version": model_version,
        "generated_at": datetime.utcnow().isoformat(),
        "prediction": prediction,
        "contributions": contributions
    }
