import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

# ==============================
# PAGE CONFIG
# ==============================
st.set_page_config(
    page_title="AQI Predictor Dashboard",
    layout="wide"
)

API_BASE_URL = "https://10pearlsaqi-production-848d.up.railway.app"

# ==============================
# SIDEBAR
# ==============================
st.sidebar.header("⚙️ Configuration")

forecast_days = st.sidebar.selectbox(
    "Forecast Horizon (Days)",
    [1, 2, 3, 5, 7],
    index=2
)

st.sidebar.markdown("---")

st.sidebar.subheader("About")
st.sidebar.info(
    "This dashboard provides real-time AQI predictions "
    "using multiple machine learning models."
)

st.sidebar.subheader("API Status")

try:
    health = requests.get(f"{API_BASE_URL}/")
    if health.status_code == 200:
        st.sidebar.success("API Connected")
    else:
        st.sidebar.error("API Error")
except:
    st.sidebar.error("API Not Connected")

# ==============================
# TITLE
# ==============================
st.title("🌍 AQI Predictor Dashboard")

tab1, tab2, tab3 = st.tabs(
    ["📊 Current & Forecast", "📈 Model Comparison", "🧠 SHAP Analysis"]
)

# ============================================================
# TAB 1 — FORECAST
# ============================================================
with tab1:

    if st.button("Get Predictions"):

        try:
            response = requests.get(
                f"{API_BASE_URL}/forecast/multi?days={forecast_days}"
            )

            if response.status_code != 200:
                st.error("API returned error.")
                st.stop()

            data = response.json()

        except:
            st.error("Failed to fetch prediction from API.")
            st.stop()

        if "predictions" not in data:
            st.error("Invalid API response.")
            st.stop()

        forecast_df = pd.DataFrame(data["predictions"])

        col1, col2, col3 = st.columns(3)

        col1.metric("Latest AQI", round(forecast_df["aqi"].iloc[0], 2))
        col2.metric("Max AQI", round(forecast_df["aqi"].max(), 2))
        col3.metric("Average AQI", round(forecast_df["aqi"].mean(), 2))

        latest_aqi = forecast_df["aqi"].iloc[0]

        def get_status(aqi):
            if aqi <= 50:
                return "Good", "green"
            elif aqi <= 100:
                return "Moderate", "orange"
            elif aqi <= 150:
                return "Unhealthy for Sensitive Groups", "darkorange"
            elif aqi <= 200:
                return "Unhealthy", "red"
            elif aqi <= 300:
                return "Very Unhealthy", "purple"
            else:
                return "Hazardous", "maroon"

        status, color = get_status(latest_aqi)

        st.markdown(
            f"<h2 style='color:{color}'>Air Quality Status: {status}</h2>",
            unsafe_allow_html=True
        )

        fig = px.line(
            forecast_df,
            x="timestamp",
            y="aqi",
            markers=True,
            title="Forecast Trend"
        )

        st.plotly_chart(fig, use_container_width=True)

    with st.expander("ℹ About AQI Categories"):
        st.markdown("""
        - **Good (0-50)**
        - **Moderate (51-100)**
        - **Unhealthy for Sensitive Groups (101-150)**
        - **Unhealthy (151-200)**
        - **Very Unhealthy (201-300)**
        - **Hazardous (301+)**
        """)

# ============================================================
# TAB 2 — MODEL METRICS
# ============================================================
with tab2:

    try:
        response = requests.get(f"{API_BASE_URL}/models/metrics")

        if response.status_code != 200:
            st.warning("Unable to fetch model metrics.")
            st.stop()

        data = response.json()

    except:
        st.warning("Unable to fetch model metrics.")
        st.stop()

    if "models" not in data:
        st.warning("No model data returned.")
        st.stop()

    models_df = pd.DataFrame(data["models"])

    if models_df.empty:
        st.warning("Model list empty.")
        st.stop()

    best_model = models_df.sort_values("rmse").iloc[0]

    st.success(
        f"🏆 Best Model: {best_model['model_name']} "
        f"(RMSE: {best_model['rmse']:.2f}, "
        f"R²: {best_model['r2']:.3f})"
    )

    st.dataframe(models_df)

    fig = px.bar(
        models_df,
        x="model_name",
        y="rmse",
        title="RMSE Comparison",
        color="rmse"
    )

    st.plotly_chart(fig, use_container_width=True)

# ============================================================
# TAB 3 — SHAP
# ============================================================
with tab3:

    try:
        response = requests.get(f"{API_BASE_URL}/forecast/shap")

        if response.status_code != 200:
            st.warning("SHAP endpoint error.")
            st.stop()

        shap_data = response.json()

    except:
        st.warning("Unable to fetch SHAP.")
        st.stop()

    if "contributions" not in shap_data:
        st.warning("SHAP data unavailable.")
        st.stop()

    shap_df = pd.DataFrame(shap_data["contributions"])

    shap_df = shap_df.rename(columns={"shap_value": "value"})

    shap_df = shap_df.reindex(
        shap_df["value"].abs().sort_values(ascending=False).index
    ).head(5)

    shap_df["color"] = shap_df["value"].apply(
        lambda x: "green" if x > 0 else "red"
    )

    fig_bar = go.Figure()

    fig_bar.add_trace(
        go.Bar(
            x=shap_df["value"],
            y=shap_df["feature"],
            orientation="h",
            marker_color=shap_df["color"]
        )
    )

    st.plotly_chart(fig_bar, use_container_width=True)

    base_value = shap_data["prediction"] - shap_df["value"].sum()

    fig_waterfall = go.Figure(go.Waterfall(
        orientation="h",
        measure=["absolute"] + ["relative"] * len(shap_df),
        y=["Base Value"] + shap_df["feature"].tolist(),
        x=[base_value] + shap_df["value"].tolist()
    ))

    st.plotly_chart(fig_waterfall, use_container_width=True)
