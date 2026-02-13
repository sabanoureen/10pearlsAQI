import shap
import pandas as pd
from datetime import datetime

from app.db.mongo import get_db
from app.pipelines.load_production_model import load_production_model

def generate_shap_analysis():

    db = get_db()

    model, features, model_version = load_production_model(horizon=1)

    latest_doc = list(
        db["feature_store"]
        .find()
        .sort("datetime", -1)
        .limit(1)
    )

    if not latest_doc:
        raise RuntimeError("No feature data found")

    latest_doc = latest_doc[0]

    X = pd.DataFrame([{
        col: latest_doc.get(col)
        for col in features
    }])

    prediction = float(model.predict(X)[0])

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    contributions = []

    for feature, value in zip(features, shap_values[0]):
        contributions.append({
            "feature": feature,
            "shap_value": float(value)
        })

    return {
        "status": "success",
        "generated_at": datetime.utcnow().isoformat(),
        "model_version": model_version,
        "prediction": prediction,
        "contributions": contributions
    }
