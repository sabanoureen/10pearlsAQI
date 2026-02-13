"""
SHAP Analysis Pipeline
Generates feature contribution explanation
for latest prediction using production model
"""

import shap
import pandas as pd
from datetime import datetime

from app.db.mongo import get_db
from app.pipelines.load_production_model import load_production_model


def generate_shap_analysis():

    db = get_db()

    # --------------------------------------------------
    # 1️⃣ Load production model (horizon=1)
    # --------------------------------------------------
    model, features, model_version = load_production_model(horizon=1)

    # --------------------------------------------------
    # 2️⃣ Get latest feature row
    # --------------------------------------------------
    latest_doc = list(
        db["feature_store"]
        .find()
        .sort("datetime", -1)
        .limit(1)
    )

    if not latest_doc:
        raise RuntimeError("No feature data found for SHAP")

    latest_doc = latest_doc[0]

    # Build feature dataframe
    X = pd.DataFrame([{
        col: latest_doc.get(col)
        for col in features
    }])

    # --------------------------------------------------
    # 3️⃣ Create SHAP Explainer
    # --------------------------------------------------
    explainer = shap.TreeExplainer(model)

    shap_values = explainer.shap_values(X)

    # For regression → shap_values is 2D array
    shap_values = shap_values[0]

    prediction = float(model.predict(X)[0])

    # --------------------------------------------------
    # 4️⃣ Prepare contributions
    # --------------------------------------------------
    contributions = []

    for feature, value in zip(features, shap_values):
        contributions.append({
            "feature": feature,
            "shap_value": float(value)
        })

    # Sort by absolute importance
    contributions = sorted(
        contributions,
        key=lambda x: abs(x["shap_value"]),
        reverse=True
    )

    return {
        "status": "success",
        "model_version": model_version,
        "generated_at": datetime.utcnow(),
        "prediction": prediction,
        "contributions": contributions
    }
