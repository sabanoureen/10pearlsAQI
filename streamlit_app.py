import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

# ======================================================
# PAGE CONFIG
# ======================================================
st.set_page_config(
    page_title="AQI Predictor Dashboard",
    layout="wide"
)

API_BASE_URL = "https://10pearlsaqi-production-848d.up.railway.app"

# ======================================================
# SIDEBAR
# ======================================================
st.sidebar.header("⚙️ Configuration")

forecast_days = st.sidebar.selectbox(
    "Forecast Horizon (Days)",
    [1, 2, 3, 5, 7],
    index=2
)

st.sidebar.markdown("---")
st.sidebar.info("Real-time AQI predictions using ML models deployed via FastAPI.")

# API Status Check
try:
    health = requests.get(f"{API_BASE_URL}/", timeout=5)
    if health.status_code == 200:
        st.sidebar.success("API Connected ✅")
    else:
        st.sidebar.error("API Error ❌")
except:
    st.sidebar.error("API Not Reachable ❌")

# ======================================================
# TITLE
# ======================================================
st.title("🌍 AQI Predictor Dashboard")

tab1, tab2, tab3 = st.tabs(
    ["📊 Current & Forecast", "📈 Model Comparison", "🧠 SHAP Analysis"]
)

# ======================================================
# TAB 1 — FORECAST
# ======================================================
with tab1:

    if st.button("Get Predictions"):

        try:
            response = requests.get(
                f"{API_BASE_URL}/forecast/multi?days={forecast_days}",
                timeout=10
            )

            if response.status_code != 200:
                st.error("API returned error.")
                st.stop()

            data = response.json()

        except Exception as e:
            st.error("Failed to fetch prediction.")
            st.stop()

        if "predictions" not in data:
            st.error(f"Unexpected API format: {data}")
            st.stop()

        df = pd.DataFrame(data["predictions"])

        if df.empty:
            st.warning("No predictions returned.")
            st.stop()

        # Detect correct value column
        if "aqi" in df.columns:
            value_col = "aqi"
        elif "prediction" in df.columns:
            value_col = "prediction"
        else:
            st.error(f"Prediction column not found: {df.columns}")
            st.stop()

        col1, col2, col3 = st.columns(3)

        col1.metric("Latest AQI", round(df[value_col].iloc[0], 2))
        col2.metric("Max AQI", round(df[value_col].max(), 2))
        col3.metric("Average AQI", round(df[value_col].mean(), 2))

        fig = px.line(
            df,
            x="date",
            y=value_col,
            markers=True,
            title="Forecast Trend"
        )

        st.plotly_chart(fig, use_container_width=True)

# ======================================================
# TAB 2 — MODEL METRICS
# ======================================================
with tab2:

    try:
        response = requests.get(
            f"{API_BASE_URL}/models/metrics",
            timeout=10
        )

        if response.status_code != 200:
            st.error("Unable to fetch model metrics.")
            st.stop()

        models_data = response.json()

    except:
        st.error("Model metrics request failed.")
        st.stop()

    if "models" not in models_data:
        st.error(f"Unexpected API format: {models_data}")
        st.stop()

    df = pd.DataFrame(models_data["models"])

    if df.empty:
        st.warning("No models registered.")
        st.stop()

    # Ensure correct columns
    required_cols = {"model_name", "rmse", "r2"}
    if not required_cols.issubset(df.columns):
        st.error(f"Missing required columns: {df.columns}")
        st.stop()

    best = df.sort_values("rmse").iloc[0]

    st.success(
        f"🏆 Best Model: {best['model_name']} "
        f"(RMSE: {best['rmse']:.2f}, R²: {best['r2']:.3f})"
    )

    st.dataframe(df)

    fig = px.bar(
        df,
        x="model_name",
        y="rmse",
        color="rmse",
        title="Model RMSE Comparison"
    )

    st.plotly_chart(fig, use_container_width=True)

# ======================================================
# TAB 3 — SHAP ANALYSIS
# ======================================================
with tab3:

    try:
        response = requests.get(
            f"{API_BASE_URL}/forecast/shap",
            timeout=10
        )

        if response.status_code != 200:
            st.error("API returned error.")
            st.stop()

        data = response.json()

    except:
        st.error("Failed to fetch SHAP data.")
        st.stop()

    if "contributions" not in data:
        st.error(f"Unexpected SHAP format: {data}")
        st.stop()

    shap_df = pd.DataFrame(data["contributions"])

    if shap_df.empty:
        st.warning("No SHAP contributions found.")
        st.stop()

    shap_df = shap_df.rename(columns={"shap_value": "value"})

    shap_df = shap_df.reindex(
        shap_df["value"].abs().sort_values(ascending=False).index
    ).head(5)

    shap_df["color"] = shap_df["value"].apply(
        lambda x: "green" if x > 0 else "red"
    )

    st.subheader("Top 5 Feature Impact")

    # BAR CHART
    fig_bar = go.Figure()

    fig_bar.add_trace(go.Bar(
        x=shap_df["value"],
        y=shap_df["feature"],
        orientation="h",
        marker_color=shap_df["color"]
    ))

    fig_bar.update_layout(
        title="Top 5 Feature Impact on AQI",
        xaxis_title="SHAP Value",
        yaxis_title="Feature"
    )

    st.plotly_chart(fig_bar, use_container_width=True)

    # WATERFALL
    prediction = data.get("prediction", 0)
    base_value = prediction - shap_df["value"].sum()

    fig_waterfall = go.Figure(go.Waterfall(
        orientation="h",
        measure=["absolute"] + ["relative"] * len(shap_df),
        y=["Base Value"] + shap_df["feature"].tolist(),
        x=[base_value] + shap_df["value"].tolist(),
    ))

    fig_waterfall.update_layout(
        title="Waterfall Explanation of AQI Prediction"
    )

    st.plotly_chart(fig_waterfall, use_container_width=True)

    st.success(f"Prediction Explained: {prediction:.2f}")
