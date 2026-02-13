import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

# ==================================
# CONFIG
# ==================================
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

# ==================================
# TITLE
# ==================================
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
                f"{API_BASE_URL}/forecast/multi?days={forecast_days}"
            )

            if response.status_code != 200:
                st.error("API returned error.")
                st.stop()

            data = response.json()

        except Exception:
            st.error("Failed to connect to API.")
            st.stop()

        if "predictions" not in data:
            st.error("Invalid API response.")
            st.stop()

        df = pd.DataFrame(data["predictions"])

        # -----------------------------
        # FIX COLUMN NAMES
        # -----------------------------
        if "predicted_aqi" not in df.columns:
            st.error(f"Prediction column not found: {df.columns}")
            st.stop()

        df = df.rename(columns={
            "predicted_aqi": "aqi",
            "datetime": "date"
        })

        df["date"] = pd.to_datetime(df["date"])

        # -----------------------------
        # METRICS
        # -----------------------------
        col1, col2, col3 = st.columns(3)

        col1.metric("Latest AQI", round(df["aqi"].iloc[0], 2))
        col2.metric("Max AQI", round(df["aqi"].max(), 2))
        col3.metric("Average AQI", round(df["aqi"].mean(), 2))

        # -----------------------------
        # AQI STATUS
        # -----------------------------
        latest_aqi = df["aqi"].iloc[0]

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

        # -----------------------------
        # TREND GRAPH
        # -----------------------------
        fig = px.line(
            df,
            x="date",
            y="aqi",
            markers=True,
            title="Forecast Trend"
        )

        st.plotly_chart(fig, use_container_width=True)

# ======================================================
# TAB 2 — MODEL COMPARISON
# ======================================================
with tab2:

    try:
        response = requests.get(f"{API_BASE_URL}/models/metrics")

        if response.status_code != 200:
            st.warning("Unable to fetch model metrics.")
            st.stop()

        models = response.json().get("models", [])

        if not models:
            st.warning("No model metrics available.")
            st.stop()

        df = pd.DataFrame(models)

    except Exception:
        st.error("API connection failed.")
        st.stop()

    best = df.sort_values("rmse").iloc[0]

    st.success(
        f"🏆 Best Model: {best['model_name']} "
        f"(RMSE: {best['rmse']:.2f}, R²: {best['r2']:.3f})"
    )

    st.dataframe(df, use_container_width=True)

    fig = px.bar(df, x="model_name", y="rmse", color="rmse",
                 title="Model RMSE Comparison")

    st.plotly_chart(fig, use_container_width=True)

# ======================================================
# TAB 3 — SHAP ANALYSIS
# ======================================================
with tab3:

    try:
        response = requests.get(f"{API_BASE_URL}/forecast/shap")

        if response.status_code != 200:
            st.error("API returned error.")
            st.stop()

        data = response.json()

        if "contributions" not in data:
            st.warning("No SHAP data available.")
            st.stop()

        shap_df = pd.DataFrame(data["contributions"])

    except Exception:
        st.error("Failed to fetch SHAP data.")
        st.stop()

    shap_df = shap_df.rename(columns={"shap_value": "value"})

    shap_df = shap_df.reindex(
        shap_df["value"].abs().sort_values(ascending=False).index
    ).head(5)

    shap_df["color"] = shap_df["value"].apply(
        lambda x: "green" if x > 0 else "red"
    )

    # -----------------------------
    # BAR
    # -----------------------------
    fig_bar = go.Figure()

    fig_bar.add_trace(go.Bar(
        x=shap_df["value"],
        y=shap_df["feature"],
        orientation="h",
        marker_color=shap_df["color"]
    ))

    fig_bar.update_layout(title="Top 5 Feature Impact on AQI")

    st.plotly_chart(fig_bar, use_container_width=True)

    # -----------------------------
    # WATERFALL
    # -----------------------------
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
