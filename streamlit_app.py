import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import requests
import time

# ==========================================================
# CONFIG
# ==========================================================

st.set_page_config(
    page_title="Karachi AQI Forecast",
    page_icon="🌍",
    layout="wide"
)

st.title("🌍 Karachi AQI Forecast System")
st.markdown("AI-Powered Multi-Horizon Air Quality Prediction")
st.markdown("---")

# ==========================================================
# BACKEND API
# ==========================================================

FORECAST_URL = "https://web-production-382ce.up.railway.app/forecast"


@st.cache_data(ttl=120)
def fetch_forecast():
    """
    Handles Railway cold start automatically.
    Waits up to 60 seconds before failing.
    """

    max_wait = 60     # total wait time (seconds)
    interval = 5      # retry every 5 seconds
    waited = 0

    while waited < max_wait:
        try:
            response = requests.get(FORECAST_URL, timeout=30)
            response.raise_for_status()
            return response.json()

        except requests.exceptions.RequestException:
            time.sleep(interval)
            waited += interval

    return None


with st.spinner("🔄 Connecting to backend and generating forecast..."):
    results = fetch_forecast()

if not results:
    st.error("❌ Backend did not respond. Please try again in a moment.")
    st.stop()

# ==========================================================
# AQI CATEGORY LOGIC
# ==========================================================

def aqi_category(value):
    if value <= 50:
        return "Good", "green"
    elif value <= 100:
        return "Moderate", "yellow"
    elif value <= 150:
        return "Unhealthy for Sensitive Groups", "orange"
    elif value <= 200:
        return "Unhealthy", "red"
    else:
        return "Hazardous", "purple"

# ==========================================================
# GAUGE FUNCTION
# ==========================================================

def create_gauge(value, date_label):

    category, _ = aqi_category(value)

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        title={'text': f"{date_label}<br>{category}"},
        gauge={
            'axis': {'range': [0, 300]},
            'steps': [
                {'range': [0, 50], 'color': "green"},
                {'range': [50, 100], 'color': "yellow"},
                {'range': [100, 150], 'color': "orange"},
                {'range': [150, 200], 'color': "red"},
                {'range': [200, 300], 'color': "purple"},
            ],
        }
    ))

    fig.update_layout(height=350)
    st.plotly_chart(fig, use_container_width=True)

# ==========================================================
# FORECAST SECTION
# ==========================================================

st.markdown("## 📅 Multi-Day AQI Forecast")

col1, col2, col3 = st.columns(3)

with col1:
    create_gauge(
        results["1_day"]["value"],
        f"1 Day Forecast<br>📅 {results['1_day']['date']}"
    )

with col2:
    create_gauge(
        results["2_day"]["value"],
        f"2 Day Forecast<br>📅 {results['2_day']['date']}"
    )

with col3:
    create_gauge(
        results["3_day"]["value"],
        f"3 Day Forecast<br>📅 {results['3_day']['date']}"
    )

# ==========================================================
# TREND CHART
# ==========================================================

st.markdown("---")
st.markdown("## 📈 Forecast Trend")

forecast_dates = [
    results["1_day"]["date"],
    results["2_day"]["date"],
    results["3_day"]["date"]
]

forecast_values = [
    results["1_day"]["value"],
    results["2_day"]["value"],
    results["3_day"]["value"]
]

fig = go.Figure()

fig.add_trace(go.Scatter(
    x=forecast_dates,
    y=forecast_values,
    mode='lines+markers',
    name="AQI Forecast"
))

fig.update_layout(
    height=450,
    xaxis_title="Date",
    yaxis_title="AQI Value"
)

st.plotly_chart(fig, use_container_width=True)

# ==========================================================
# HEALTH ADVISORY
# ==========================================================

st.markdown("---")
st.markdown("## 🚨 Health Advisory")

max_aqi = max(forecast_values)

if max_aqi > 150:
    st.error("⚠️ Hazardous air quality expected. Avoid outdoor exposure.")
elif max_aqi > 100:
    st.warning("⚠️ Sensitive groups should limit outdoor activity.")
else:
    st.success("✅ Air quality within acceptable limits.")

# ==========================================================
# EXECUTIVE SUMMARY
# ==========================================================

st.markdown("---")
st.markdown("## 📌 Executive Summary")

st.info(f"""
• {results["1_day"]["date"]} → AQI: {results["1_day"]["value"]}  
• {results["2_day"]["date"]} → AQI: {results["2_day"]["value"]}  
• {results["3_day"]["date"]} → AQI: {results["3_day"]["value"]}  

Predictions generated via automated MLOps pipeline  
(Feature Pipeline → Model Registry → GridFS → Railway Production Deployment).
""")