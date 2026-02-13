import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

st.set_page_config(page_title="AQI Predictor Dashboard", layout="wide")

API_BASE_URL = "https://10pearlsaqi-production-848d.up.railway.app"

st.sidebar.header("⚙️ Configuration")

forecast_days = st.sidebar.selectbox(
    "Forecast Horizon (Days)",
    [1, 2, 3, 5, 7],
    index=2
)

st.sidebar.markdown("---")
st.sidebar.info("Real-time AQI predictions using ML models deployed via FastAPI.")

# =========================
# API HEALTH CHECK
# =========================
try:
    health = requests.get(f"{API_BASE_URL}/")
    if health.status_code == 200:
        st.sidebar.success("API Connected ✅")
    else:
        st.sidebar.error("API Not Connected")
except:
    st.sidebar.error("API Not Connected")

st.title("🌍 AQI Predictor Dashboard")

tab1, tab2, tab3 = st.tabs(
    ["📊 Current & Forecast", "📈 Model Comparison", "🧠 SHAP Analysis"]
)

# ====================================================
# TAB 1 — FORECAST
# ====================================================
with tab1:

    if st.button("Get Predictions"):

        response = requests.get(
            f"{API_BASE_URL}/forecast/multi?days={forecast_days}"
        )

        if response.status_code != 200:
            st.error("API returned error.")
            st.stop()

        data = response.json()

        df = pd.DataFrame(data["predictions"])

        # Rename for safety
        if "predicted_aqi" in df.columns:
            df.rename(columns={"predicted_aqi": "aqi"}, inplace=True)

        if "aqi" not in df.columns:
            st.error(f"Prediction column not found: {df.columns}")
            st.stop()

        col1, col2, col3 = st.columns(3)

        col1.metric("Latest AQI", round(df["aqi"].iloc[0], 2))
        col2.metric("Max AQI", round(df["aqi"].max(), 2))
        col3.metric("Average AQI", round(df["aqi"].mean(), 2))

        # AQI category
        latest = df["aqi"].iloc[0]

        if latest <= 50:
            status = "Good"
            color = "green"
        elif latest <= 100:
            status = "Moderate"
            color = "orange"
        elif latest <= 150:
            status = "Unhealthy for Sensitive Groups"
            color = "darkorange"
        elif latest <= 200:
            status = "Unhealthy"
            color = "red"
        elif latest <= 300:
            status = "Very Unhealthy"
            color = "purple"
        else:
            status = "Hazardous"
            color = "maroon"

        st.markdown(
            f"<h2 style='color:{color}'>Air Quality Status: {status}</h2>",
            unsafe_allow_html=True
        )

        fig = px.line(df, x="date", y="aqi", markers=True)
        st.plotly_chart(fig, use_container_width=True)

# ====================================================
# TAB 2 — MODEL METRICS
# ====================================================
with tab2:

    response = requests.get(f"{API_BASE_URL}/models/metrics")

    if response.status_code != 200:
        st.warning("Unable to fetch model metrics.")
        st.stop()

    models = response.json()["models"]

    if not models:
        st.warning("No model metrics available.")
        st.stop()

    df = pd.DataFrame(models)

    best = df.sort_values("rmse").iloc[0]

    st.success(
        f"🏆 Best Model: {best['model_name']} "
        f"(RMSE: {best['rmse']:.2f}, R²: {best['r2']:.3f})"
    )

    st.dataframe(df)

    fig = px.bar(df, x="model_name", y="rmse", color="rmse")
    st.plotly_chart(fig, use_container_width=True)

# ====================================================
# TAB 3 — SHAP
# ====================================================
with tab3:

    response = requests.get(f"{API_BASE_URL}/forecast/shap")

    if response.status_code != 200:
        st.warning("SHAP analysis not available.")
        st.stop()

    data = response.json()

    shap_df = pd.DataFrame(data["contributions"])

    shap_df.rename(columns={"shap_value": "value"}, inplace=True)

    shap_df = shap_df.reindex(
        shap_df["value"].abs().sort_values(ascending=False).index
    ).head(5)

    shap_df["color"] = shap_df["value"].apply(
        lambda x: "green" if x > 0 else "red"
    )

    fig_bar = go.Figure()
    fig_bar.add_trace(go.Bar(
        x=shap_df["value"],
        y=shap_df["feature"],
        orientation="h",
        marker_color=shap_df["color"]
    ))

    fig_bar.update_layout(title="Top 5 Feature Impact")

    st.plotly_chart(fig_bar, use_container_width=True)

    base_value = data["prediction"] - shap_df["value"].sum()

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

    st.success(f"Prediction Explained: {data['prediction']:.2f}")
