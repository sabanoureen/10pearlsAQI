import shap
import pandas as pd

from app.db.mongo import get_db
from app.pipelines.load_production_model import load_production_model


def generate_shap_analysis():

    db = get_db()

    # Load production model
    model, features, model_version = load_production_model(horizon=1)

    # Get latest feature row
    latest_doc = list(
        db["feature_store"]
        .find()
        .sort("datetime", -1)
        .limit(1)
    )

    if not latest_doc:
        raise RuntimeError("No feature data found")

    latest_doc = latest_doc[0]

    current_features = {
        col: latest_doc.get(col)
        for col in features
    }

    X = pd.DataFrame([current_features])

    # XGBoost → TreeExplainer
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    prediction = float(model.predict(X)[0])

    contributions = []

    for i, feature in enumerate(features):
        contributions.append({
            "feature": feature,
            "value": float(X.iloc[0][feature]),
            "shap_value": float(shap_values[0][i])
        })

    # Sort by impact
    contributions = sorted(
        contributions,
        key=lambda x: abs(x["shap_value"]),
        reverse=True
    )

    return {
        "model_version": model_version,
        "prediction": prediction,
        "contributions": contributions[:15]
    }
