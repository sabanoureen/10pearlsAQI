"""
Feature Pipeline Runner
-----------------------
- Builds latest feature dataframe
- Writes features to MongoDB feature store
- Single CI/CD entry point
"""

import sys
import os

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
)

from app.pipelines.final_feature_table import build_final_dataframe
from app.pipelines.save_features import save_features


def run():
    print("🚀 Starting feature pipeline")

    df = build_final_dataframe()

    if df is None or df.empty:
        raise RuntimeError("❌ Feature pipeline produced empty dataframe")

    # 🔥 CLEAR OLD DATA (optional but recommended)
    from app.db.mongo import get_feature_store
    get_feature_store().delete_many({})

    # 🔥 INSERT EACH ROW
    for _, row in df.iterrows():
        save_features(city="Karachi", row=row.to_dict())

    print(f"✅ Feature pipeline success | rows={len(df)} inserted into Mongo")


if __name__ == "__main__":
    run()
