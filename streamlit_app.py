import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import requests
from datetime import datetime

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
# BACKEND API URL (Railway)
# ==========================================================

API_URL = "https://web-production-382ce.up.railway.app/forecast"

# ==========================================================
# REFRESH BUTTON
# ==========================================================

if st.button("🔄 Refresh Live Forecast"):
    st.rerun()

# ==========================================================
# FETCH FORECAST FROM BACKEND
# ==========================================================

try:
    response = requests.get(API_URL, timeout=10)

    if response.status_code == 200:
        results = response.json()
    else:
        st.error("❌ Backend error. Please try again later.")
        st.stop()

except Exception as e:
    st.error("❌ Could not connect to backend.")
    st.stop()

# ==========================================================
# GAUGE FUNCTION
# ==========================================================

def create_gauge(value, title):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        title={'text': title},
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
        f"1 Day Forecast\n📅 {results['1_day']['date']}\n🤖 {results['1_day']['model']}"
    )

with col2:
    create_gauge(
        results["2_day"]["value"],
        f"2 Day Forecast\n📅 {results['2_day']['date']}\n🤖 {results['2_day']['model']}"
    )

with col3:
    create_gauge(
        results["3_day"]["value"],
        f"3 Day Forecast\n📅 {results['3_day']['date']}\n🤖 {results['3_day']['model']}"
    )

# ==========================================================
# TREND CHART (Forecast Only)
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

fig.update_layout(height=450)

st.plotly_chart(fig, use_container_width=True)

# ==========================================================
# MODEL COMPARISON (Static Benchmark Table)
# ==========================================================

st.markdown("---")
st.markdown("## 📊 Model Benchmark Comparison")

comparison_df = pd.DataFrame({
    "Horizon": ["H1 (24h)", "H2 (48h)", "H3 (72h)"],
    "Random Forest RMSE": [3.0669, 2.6788, 2.8933],
    "Gradient Boosting RMSE": [9.4081, 9.3588, 9.5649],
    "Ridge RMSE": [12.3637, 13.3617, 13.5450],
})

st.dataframe(comparison_df, use_container_width=True)

# ==========================================================
# BEST MODEL SECTION
# ==========================================================

st.markdown("---")
st.markdown("## 🏆 Production Model Selection")

st.success("""
Random Forest achieved the lowest RMSE
across all forecast horizons (24h, 48h, 72h).

Therefore, Random Forest is deployed as
the production model.
""")

# ==========================================================
# EXECUTIVE SUMMARY
# ==========================================================

st.markdown("---")
st.markdown("## 📌 Executive Summary")

st.info(f"""
• {results["1_day"]["date"]} → AQI: {results["1_day"]["value"]}  
• {results["2_day"]["date"]} → AQI: {results["2_day"]["value"]}  
• {results["3_day"]["date"]} → AQI: {results["3_day"]["value"]}  

Production model selected automatically via daily training pipeline.
""")
