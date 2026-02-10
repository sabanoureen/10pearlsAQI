from app.pipelines.final_feature_table import build_final_dataframe


def build_training_dataset():
    df = build_final_dataframe()

    if df.empty:
        raise RuntimeError("Final dataframe is empty")

    # 🔥 IMPORTANT: drop rows with NaNs
    print("Rows before dropna:", len(df))
    df = df.dropna()
    print("Rows after dropna:", len(df))

    df = df.dropna().reset_index(drop=True)

    if df.empty:
        raise RuntimeError("All rows dropped after NaN removal")

    X = df.drop(columns=["aqi_pm25", "timestamp"], errors="ignore")
    y = df["aqi_pm25"]

    return X, y
