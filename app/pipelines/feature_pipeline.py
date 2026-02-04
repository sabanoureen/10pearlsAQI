"""
Feature Pipeline
----------------
- Fetches latest AQI + weather data
- Performs feature engineering
- Stores features in MongoDB
"""

import os
import sys

# -------------------------------------------------
# Ensure project root is on PYTHONPATH
# -------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from app.pipelines.data_fetcher import fetch_and_store_data
from app.pipelines.feature_engineering import generate_features


def run_feature_pipeline():
    print("🚀 Starting feature pipeline")

    fetch_and_store_data()
    generate_features()

    print("✅ Feature pipeline completed successfully")


if __name__ == "__main__":
    run_feature_pipeline()
