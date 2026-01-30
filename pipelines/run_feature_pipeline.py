"""
Feature Pipeline Runner
-----------------------
- Builds latest feature dataframe
- Writes features to MongoDB feature store
- Single CI/CD entry point
"""

import sys
import os

# Ensure project root is on PYTHONPATH
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
)

from pipelines.final_feature_table import build_final_dataframe


def run():
    print("🚀 Starting feature pipeline")

    df = build_final_dataframe()

    if df is None or df.empty:
        raise RuntimeError("❌ Feature pipeline produced empty dataframe")

    print(f"✅ Feature pipeline success | rows={len(df)}")


if __name__ == "__main__":
    run()
