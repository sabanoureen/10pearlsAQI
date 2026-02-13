import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go

# =========================================
# CONFIG
# =========================================
API_URL = "https://10pearlsaqi-production-848d.up.railway.app"

st.set_page_config(
    page_title="AQI Forecast Dashboard",
    page_icon="🌍",
    layout="wide"
)

# =========================================
# AQI STATUS FUNCTION
# =========================================
def get_aqi_status(aqi):
    if aqi <= 50:
        return "Good", "green"
    elif aqi <= 100:
        return "Moderate", "gold"
    elif aqi <= 150:
        return "Unhealthy (Sensitive)", "orange"
    elif aqi <= 200:
        return "Unhealthy", "red"
    else:
        return "Hazardous", "purple"


# =========================================
# HEADER
# =========================================
st.title("🌍 AQI Forecast Dashboard")
st.markdown("### Machine Learning powered air quality forecasting system")
st.divider()

# =========================================
# SIDEBAR
# =========================================
st.sidebar.header("⚙ Configuration")

horizon = st.sidebar.selectbox(
    "Forecast Horizon (Days)",
    [1, 3, 7],
    index=1
)

if st.sidebar.button("🔄 Generate Forecast"):
    r = requests.get(f"{API_URL}/forecast/multi?days={horizon}")
    if r.status_code == 200:
        st.success("Forecast generated successfully!")
    else:
        st.error("Failed to generate forecast")


# =========================================
# LOAD FORECAST
# =========================================
if st.button("📊 Load Forecast"):

    response = requests.get(f"{API_URL}/forecast/multi?days={horizon}")

    if response.status_code != 200:
        st.error("Failed to fetch forecast.")
        st.stop()

    results = response.json()

    df = pd.DataFrame(results["predictions"])
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values("datetime").reset_index(drop=True)

    latest_aqi = df["predicted_aqi"].iloc[-1]
    max_aqi = df["predicted_aqi"].max()
    avg_aqi = df["predicted_aqi"].mean()

    status, color = get_aqi_status(latest_aqi)

    # KPI
    col1, col2, col3 = st.columns(3)
    col1.metric("Latest AQI", round(latest_aqi, 2))
    col2.metric("Max AQI", round(max_aqi, 2))
    col3.metric("Average AQI", round(avg_aqi, 2))

    st.markdown(
        f"<h3 style='color:{color}'>Air Quality Status: {status}</h3>",
        unsafe_allow_html=True
    )

    # Model Info
    st.subheader("🤖 Model Information")
    colA, colB = st.columns(2)
    colA.info(f"Generated At: {results.get('generated_at', 'N/A')}")
    colB.info(f"Model Version: {results.get('model_version', 'production_v1')}")

    # Gauge
    st.subheader("🌡 AQI Gauge")

    gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=latest_aqi,
        title={'text': "Forecast AQI"},
        gauge={'axis': {'range': [0, 300]}}
    ))

    st.plotly_chart(gauge, use_container_width=True)

    # Forecast Graph
    st.subheader("📊 Forecast Trend")

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df["datetime"],
            y=df["predicted_aqi"],
            mode="lines+markers",
            name="Forecast AQI"
        )
    )

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("📋 Forecast Data")
    st.dataframe(df, use_container_width=True)

else:
    st.info("Click 'Load Forecast' to view predictions.")


# =========================================
# SHAP SECTION (SAFE PLACEHOLDER)
# =========================================
st.divider()
st.subheader("🧠 Model Explainability (SHAP)")

if st.button("Show SHAP Analysis", key="shap_button"):
    st.warning("SHAP endpoint not implemented yet.")
