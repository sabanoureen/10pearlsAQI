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

API_BASE_URL = "https://10pearlsaqi-production-848d.up.railway.app"  # 🔁 CHANGE THIS

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
    health = requests.get(f"{API_BASE_URL}/health").json()
    st.sidebar.success("API Connected")
except:
    st.sidebar.error("API Not Connected")

# ==============================
# MAIN TITLE
# ==============================
st.title("🌍 AQI Predictor Dashboard")

# ==============================
# TABS
# ==============================
tab1, tab2, tab3 = st.tabs(
    ["📊 Current & Forecast", "📈 Model Comparison", "🧠 SHAP Analysis"]
)

# ============================================================
# TAB 1 — CURRENT AQI + FORECAST
# ============================================================
with tab1:

    if st.button("Get Predictions"):

        try:
            response = requests.get(
                f"{API_BASE_URL}/predict?days={forecast_days}"
            )
            data = response.json()

        except:
            st.error("Failed to fetch prediction from API.")
            st.stop()

        if "forecast" not in data:
            st.error("Invalid API response.")
            st.stop()

        forecast_df = pd.DataFrame(data["forecast"])

        col1, col2, col3 = st.columns(3)

        col1.metric("Latest AQI", round(forecast_df["aqi"].iloc[0], 2))
        col2.metric("Max AQI", round(forecast_df["aqi"].max(), 2))
        col3.metric("Average AQI", round(forecast_df["aqi"].mean(), 2))

        latest_aqi = forecast_df["aqi"].iloc[0]

        # AQI Status
        def get_aqi_status(aqi):
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

        status, color = get_aqi_status(latest_aqi)

        st.markdown(
            f"<h2 style='color:{color}'>Air Quality Status: {status}</h2>",
            unsafe_allow_html=True
        )

        # Forecast Trend
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
        - **Good (0-50)**: Air quality is satisfactory.
        - **Moderate (51-100)**: Acceptable for most people.
        - **Unhealthy for Sensitive Groups (101-150)**.
        - **Unhealthy (151-200)**.
        - **Very Unhealthy (201-300)**.
        - **Hazardous (301+)**.
        """)

# ============================================================
# TAB 2 — MODEL COMPARISON
# ============================================================
with tab2:

    try:
        response = requests.get(f"{API_BASE_URL}/models")
        models_data = response.json()
    except:
        st.warning("Unable to fetch model comparison data.")
        st.stop()

    if "models" not in models_data:
        st.warning("No model data returned.")
        st.stop()

    models_df = pd.DataFrame(models_data["models"])

    if models_df.empty:
        st.warning("Model list is empty.")
        st.stop()

    # Normalize columns
    models_df.columns = models_df.columns.str.lower()

    if "rmse" not in models_df.columns:
        st.error("RMSE column not found in API.")
        st.stop()

    # Best model
    best_model = models_df.sort_values("rmse").iloc[0]

    st.success(
        f"🏆 Best Model: {best_model['name']} "
        f"(RMSE: {best_model['rmse']:.2f}, "
        f"R²: {best_model.get('r2', 0):.3f})"
    )

    st.dataframe(models_df)

    # RMSE Bar Chart
    fig = px.bar(
        models_df,
        x="name",
        y="rmse",
        title="Model RMSE Comparison",
        color="rmse"
    )

    st.plotly_chart(fig, use_container_width=True)

# ============================================================
# TAB 3 — SHAP ANALYSIS
# ============================================================
with tab3:

    try:
        shap_response = requests.get(f"{API_BASE_URL}/shap")
        shap_data = shap_response.json()
    except:
        st.warning("Unable to fetch SHAP data.")
        st.stop()

    if "shap_values" not in shap_data:
        st.warning("SHAP values not available.")
        st.stop()

    shap_df = pd.DataFrame(shap_data["shap_values"])

    shap_df = shap_df.sort_values(
        by="value",
        key=lambda x: abs(x),
        ascending=False
    ).head(5)

    # Positive / Negative Coloring
    shap_df["color"] = shap_df["value"].apply(
        lambda x: "green" if x > 0 else "red"
    )

    # ==========================
    # SHAP Bar Chart
    # ==========================
    fig_bar = go.Figure()

    fig_bar.add_trace(
        go.Bar(
            x=shap_df["value"],
            y=shap_df["feature"],
            orientation="h",
            marker_color=shap_df["color"]
        )
    )

    fig_bar.update_layout(
        title="Top 5 Feature Impact on AQI",
        xaxis_title="SHAP Value (Impact)",
        yaxis_title="Feature"
    )

    st.plotly_chart(fig_bar, use_container_width=True)

    # ==========================
    # WATERFALL STYLE
    # ==========================
    base_value = shap_data.get("base_value", 0)

    fig_waterfall = go.Figure(go.Waterfall(
        name="SHAP Waterfall",
        orientation="h",
        measure=["absolute"] + ["relative"] * len(shap_df),
        y=["Base Value"] + shap_df["feature"].tolist(),
        x=[base_value] + shap_df["value"].tolist(),
        connector={"line": {"color": "rgb(63, 63, 63)"}},
    ))

    fig_waterfall.update_layout(
        title="Waterfall Explanation of AQI Prediction"
    )

    st.plotly_chart(fig_waterfall, use_container_width=True)

    final_prediction = base_value + shap_df["value"].sum()

    st.success(f"Prediction Explained: {final_prediction:.2f}")
