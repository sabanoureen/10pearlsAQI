import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import requests

# ==========================================================
# CONFIG
# ==========================================================

st.set_page_config(
    page_title="Karachi AQI Forecast",
    page_icon="🌍",
    layout="wide"
)

# ==========================================================
# STYLING
# ==========================================================

st.markdown("""
<style>
.big-font {
    font-size:20px !important;
}
</style>
""", unsafe_allow_html=True)

# ==========================================================
# HEADER
# ==========================================================

st.title("🌍 Karachi AQI Forecast System")
st.markdown("AI-Powered Multi-Horizon Air Quality Prediction")
st.markdown("---")

# ==========================================================
# API URLS
# ==========================================================

BASE_URL = "https://web-production-382ce.up.railway.app"
FORECAST_URL = f"{BASE_URL}/forecast"
BEST_MODEL_URL = f"{BASE_URL}/models/best"
FEATURE_URL = f"{BASE_URL}/features/importance?horizon=1"
METRICS_URL = f"{BASE_URL}/models/metrics"

# ==========================================================
# FETCH FUNCTION
# ==========================================================

def safe_request(url, timeout=60):
    try:
        r = requests.get(url, timeout=timeout)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.HTTPError as e:
        st.error(f"HTTP Error: {e}")
    except requests.exceptions.ConnectionError:
        st.error("Connection error to backend.")
    except requests.exceptions.Timeout:
        st.error("Request timed out.")
    except Exception as e:
        st.error(f"Unexpected error: {e}")
    return None

# ==========================================================
# FETCH FORECAST
# ==========================================================

with st.spinner("🔄 Connecting to backend and generating forecast..."):
    results = safe_request(FORECAST_URL, timeout=120)

if results is None:
    if st.button("🔄 Retry Connection"):
        st.rerun()
    st.stop()

# ==========================================================
# AQI CATEGORY
# ==========================================================

def aqi_category(value):
    if value <= 50:
        return "Good"
    elif value <= 100:
        return "Moderate"
    elif value <= 150:
        return "Unhealthy for Sensitive Groups"
    elif value <= 200:
        return "Unhealthy"
    else:
        return "Hazardous"

# ==========================================================
# GAUGE FUNCTION
# ==========================================================

def create_gauge(value, date_label):

    category = aqi_category(value)

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

# ==========================================================
# BEST MODEL SECTION
# ==========================================================

st.markdown("---")
st.markdown("## 🏆 Best Production Model")

best_model_data = safe_request(BEST_MODEL_URL)

if best_model_data and "model" in best_model_data:
    best_model = best_model_data["model"]

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Model Name", best_model.get("model_name", "N/A"))
        st.metric("Algorithm", best_model.get("algorithm", "N/A"))

    with col2:
        st.metric("RMSE", round(best_model.get("rmse", 0), 2))
        st.metric("R² Score", round(best_model.get("r2", 0), 3))
else:
    st.warning("Best model details unavailable.")

# ==========================================================
# FEATURE IMPORTANCE
# ==========================================================

st.markdown("---")
st.markdown("## 📊 Top Feature Importance")

feature_data = safe_request(FEATURE_URL)

if feature_data and "features" in feature_data:

    df = pd.DataFrame(feature_data["features"])
    df = df.sort_values("importance", ascending=False).head(10)

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=df["importance"],
        y=df["feature"],
        orientation="h"
    ))

    fig.update_layout(
        height=500,
        yaxis=dict(autorange="reversed"),
        xaxis_title="Importance Score"
    )

    st.plotly_chart(fig, use_container_width=True)
else:
    st.warning("Feature importance unavailable.")

# ==========================================================
# MODEL COMPARISON
# ==========================================================

st.markdown("---")
st.markdown("## 📈 Model Comparison")

metrics_data = safe_request(METRICS_URL)

if metrics_data and "models" in metrics_data:
    df_models = pd.DataFrame(metrics_data["models"])
    st.dataframe(df_models, use_container_width=True)
else:
    st.warning("Model metrics unavailable.")

# ==========================================================
# SIDEBAR
# ==========================================================

st.sidebar.header("⚙ Dashboard Controls")
st.sidebar.write("Multi-horizon forecasting system")
st.sidebar.write("Production deployment on Railway")