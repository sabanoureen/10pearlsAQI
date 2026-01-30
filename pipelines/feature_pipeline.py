"""
Feature Pipeline
----------------
- Fetches latest AQI + weather data
- Performs feature engineering
- Stores features in MongoDB

Single authoritative entry point for feature generation.
"""

import sys
import os

# -----------------------------------
# Ensure project root is on PYTHONPATH
# -----------------------------------
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
)

from pipelines.data_fetcher import fetch_and_store_data
from pipelines.feature_engineering import run_feature_pipeline


def run_feature_pipeline():
    print("🚀 Starting feature pipeline")

    # Fetch latest data
    fetch_and_store_data()

    # Generate & store features
    run_feature_pipeline()

    print("✅ Feature pipeline completed successfully")


if __name__ == "__main__":
    run_feature_pipeline()
