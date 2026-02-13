import shap
import pandas as pd
from datetime import datetime
from fastapi.encoders import jsonable_encoder

from app.db.mongo import get_db
from app.pipelines.load_production_model import load_production_model


def generate_shap_analysis():

    db = get_db()

    # Load best production model (horizon 1)
    model, features, model_version = load_production_model(horizon=1)

    # Get latest feature row
    latest_doc = db["feature_store"].find_one(
        sort=[("created_at", -1)]
    )

    if not latest_doc:
        raise RuntimeError("No feature data found for SHAP")

    # Build feature dataframe
    X = pd.DataFrame([{
        col: latest_doc.get(col, 0)
        for col in features
    }])

    # Prediction
    prediction = float(model.predict(X)[0])

    # SHAP explainer
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    contributions = []

    for feature, value in zip(features, shap_values[0]):
        contributions.append({
            "feature": feature,
            "shap_value": float(value)
        })

    contributions = sorted(
        contributions,
        key=lambda x: abs(x["shap_value"]),
        reverse=True
    )

    result = {
        "status": "success",
        "model_version": model_version,
        "generated_at": datetime.utcnow(),
        "prediction": prediction,
        "contributions": contributions
    }

    return jsonable_encoder(result)
