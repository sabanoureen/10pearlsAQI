import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

# ==========================================================
# PAGE CONFIG
# ==========================================================
st.set_page_config(
    page_title="AQI Predictor Dashboard",
    layout="wide"
)

API_BASE_URL = "https://10pearlsaqi-production-848d.up.railway.app"

# ==========================================================
# SIDEBAR
# ==========================================================
st.sidebar.header("⚙️ Configuration")

forecast_days = st.sidebar.selectbox(
    "Forecast Horizon (Days)",
    [1, 2, 3, 5, 7],
    index=2
)

st.sidebar.markdown("---")
st.sidebar.info(
    "Real-time AQI predictions using ML models deployed via FastAPI."
)

# API Health Check
try:
    health_response = requests.get(f"{API_BASE_URL}/")
    if health_response.status_code == 200:
        st.sidebar.success("API Connected ✅")
    else:
        st.sidebar.error("API Error ❌")
except:
    st.sidebar.error("API Not Reachable ❌")

# ==========================================================
# TITLE
# ==========================================================
st.title("🌍 AQI Predictor Dashboard")

tab1, tab2, tab3 = st.tabs(
    ["📊 Current & Forecast", "📈 Model Comparison", "🧠 SHAP Analysis"]
)

# ==========================================================
# TAB 1 — FORECAST
# ==========================================================
with tab1:

    if st.button("Get Predictions"):

        response = requests.get(
            f"{API_BASE_URL}/forecast/multi?days={forecast_days}"
        )

        if response.status_code != 200:
            st.error("API returned error.")
            st.stop()

        data = response.json()

        if "predictions" not in data:
            st.error("Invalid API response structure.")
            st.stop()

        df = pd.DataFrame(data["predictions"])

        if df.empty:
            st.warning("No forecast data returned.")
            st.stop()

        # --------------------------------------------------
        # Detect Prediction Column
        # --------------------------------------------------
        if "predicted_aqi" in df.columns:
            value_col = "predicted_aqi"
        elif "aqi" in df.columns:
            value_col = "aqi"
        else:
            st.error(f"Prediction column not found: {df.columns}")
            st.stop()

        # --------------------------------------------------
        # Detect Time Column
        # --------------------------------------------------
        if "datetime" in df.columns:
            time_col = "datetime"
        elif "date" in df.columns:
            time_col = "date"
        else:
            st.error(f"Time column not found: {df.columns}")
            st.stop()

        df[time_col] = pd.to_datetime(df[time_col])

        # --------------------------------------------------
        # Metrics
        # --------------------------------------------------
        col1, col2, col3 = st.columns(3)

        col1.metric("Latest AQI", round(df[value_col].iloc[0], 2))
        col2.metric("Max AQI", round(df[value_col].max(), 2))
        col3.metric("Average AQI", round(df[value_col].mean(), 2))

        # --------------------------------------------------
        # AQI Status
        # --------------------------------------------------
        latest_aqi = df[value_col].iloc[0]

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

        # --------------------------------------------------
        # Forecast Trend Chart
        # --------------------------------------------------
        fig = px.line(
            df,
            x=time_col,
            y=value_col,
            markers=True,
            title="Forecast Trend"
        )

        st.plotly_chart(fig, use_container_width=True)

    # AQI Categories
    with st.expander("ℹ About AQI Categories"):
        st.markdown("""
        - **Good (0-50)**: Air quality is satisfactory.
        - **Moderate (51-100)**: Acceptable for most people.
        - **Unhealthy for Sensitive Groups (101-150)**.
        - **Unhealthy (151-200)**.
        - **Very Unhealthy (201-300)**.
        - **Hazardous (301+)**.
        """)

# ==========================================================
# TAB 2 — MODEL COMPARISON
# ==========================================================
with tab2:

    response = requests.get(f"{API_BASE_URL}/models/metrics")

    if response.status_code != 200:
        st.error("Unable to fetch model metrics.")
        st.stop()

    data = response.json()

    if "models" not in data:
        st.error("Invalid model API response.")
        st.stop()

    df_models = pd.DataFrame(data["models"])

    if df_models.empty:
        st.warning("No model metrics available.")
        st.stop()

    # Best Model
    best = df_models.sort_values("rmse").iloc[0]

    st.success(
        f"🏆 Best Model: {best['model_name']} "
        f"(RMSE: {best['rmse']:.2f}, R²: {best['r2']:.3f})"
    )

    st.dataframe(df_models, use_container_width=True)

    fig = px.bar(
        df_models,
        x="model_name",
        y="rmse",
        color="rmse",
        title="Model RMSE Comparison"
    )

    st.plotly_chart(fig, use_container_width=True)

# ==========================================================
# TAB 3 — SHAP ANALYSIS
# ==========================================================
with tab3:

    response = requests.get(f"{API_BASE_URL}/forecast/shap")

    if response.status_code != 200:
        st.error("API returned error.")
        st.stop()

    data = response.json()

    if "contributions" not in data:
        st.error("SHAP contributions missing in API response.")
        st.stop()

    shap_df = pd.DataFrame(data["contributions"])

    shap_df = shap_df.rename(columns={"shap_value": "value"})

    shap_df = shap_df.reindex(
        shap_df["value"].abs().sort_values(ascending=False).index
    ).head(5)

    shap_df["color"] = shap_df["value"].apply(
        lambda x: "green" if x > 0 else "red"
    )

    st.subheader("Top 5 Feature Impact on AQI")

    # ---------------- BAR ----------------
    fig_bar = go.Figure()
    fig_bar.add_trace(go.Bar(
        x=shap_df["value"],
        y=shap_df["feature"],
        orientation="h",
        marker_color=shap_df["color"]
    ))

    fig_bar.update_layout(
        xaxis_title="SHAP Value (Impact)",
        yaxis_title="Feature"
    )

    st.plotly_chart(fig_bar, use_container_width=True)

    # ---------------- WATERFALL ----------------
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
