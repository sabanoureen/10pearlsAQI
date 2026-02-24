import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import requests

# ==========================================================
# PAGE CONFIG
# ==========================================================

st.set_page_config(
    page_title="Karachi AQI Forecast",
    page_icon="🌍",
    layout="wide"
)

# ==========================================================
# DARK THEME STYLING
# ==========================================================

st.markdown("""
    <style>
        .stApp {
            background-color: #0e1117;
            color: white;
        }
        h1, h2, h3, h4 {
            color: white;
        }
        .stMetric {
            background-color: #161b22;
            padding: 20px;
            border-radius: 10px;
        }
        .css-1d391kg {
            background-color: #161b22;
        }
    </style>
""", unsafe_allow_html=True)

# ==========================================================
# SIDEBAR
# ==========================================================

st.sidebar.title("⚙ Dashboard Controls")
st.sidebar.markdown("""
### Karachi AQI Forecast System

• Multi-horizon forecasting  
• Production deployment on Railway  
• MongoDB Model Registry  
• GridFS Model Storage  
• Automated MLOps Pipeline  
""")

st.sidebar.markdown("---")
st.sidebar.success("System Status: Operational")

# ==========================================================
# HEADER
# ==========================================================

st.title("🌍 Karachi AQI Forecast System")
st.markdown("AI-Powered Multi-Horizon Air Quality Prediction")
st.markdown("---")

# ==========================================================
# BACKEND URLS
# ==========================================================

BASE_URL = "https://web-production-382ce.up.railway.app"

FORECAST_URL = f"{BASE_URL}/forecast"
BEST_MODEL_URL = f"{BASE_URL}/models/best"
FEATURE_URL = f"{BASE_URL}/features/importance?horizon=1"

# ==========================================================
# FETCH FORECAST
# ==========================================================

def fetch_data(url, timeout=30):
    try:
        response = requests.get(url, timeout=timeout)
        response.raise_for_status()
        return response.json()
    except:
        return None

with st.spinner("🔄 Connecting to backend..."):
    results = fetch_data(FORECAST_URL)

if results is None:
    st.error("Backend unavailable. Please try again.")
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
        return "Unhealthy (Sensitive)"
    elif value <= 200:
        return "Unhealthy"
    else:
        return "Hazardous"

# ==========================================================
# GAUGE FUNCTION
# ==========================================================

def create_gauge(value, date_label):

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        title={'text': f"{date_label}<br>{aqi_category(value)}"},
        gauge={
            'axis': {'range': [0, 300]},
            'steps': [
                {'range': [0, 50], 'color': "#00ff88"},
                {'range': [50, 100], 'color': "#ffee00"},
                {'range': [100, 150], 'color': "#ff9900"},
                {'range': [150, 200], 'color': "#ff3333"},
                {'range': [200, 300], 'color': "#9900cc"},
            ],
        }
    ))

    fig.update_layout(
        height=350,
        paper_bgcolor="#0e1117",
        font_color="white"
    )

    st.plotly_chart(fig, use_container_width=True)

# ==========================================================
# FORECAST SECTION
# ==========================================================

st.markdown("## 📅 Multi-Day AQI Forecast")

col1, col2, col3 = st.columns(3)

with col1:
    create_gauge(results["1_day"]["value"],
                 f"1 Day<br>{results['1_day']['date']}")

with col2:
    create_gauge(results["2_day"]["value"],
                 f"2 Day<br>{results['2_day']['date']}")

with col3:
    create_gauge(results["3_day"]["value"],
                 f"3 Day<br>{results['3_day']['date']}")

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
    line=dict(color="#00c3ff", width=3),
    marker=dict(size=8),
    name="AQI Forecast"
))

fig.update_layout(
    height=450,
    template="plotly_dark",
    xaxis_title="Date",
    yaxis_title="AQI Value",
    paper_bgcolor="#0e1117"
)

st.plotly_chart(fig, use_container_width=True)

# ==========================================================
# HEALTH ADVISORY
# ==========================================================

st.markdown("---")
st.markdown("## 🚨 Health Advisory")

max_aqi = max(forecast_values)

if max_aqi > 150:
    st.error("Hazardous air quality expected. Avoid outdoor exposure.")
elif max_aqi > 100:
    st.warning("Sensitive groups should limit outdoor activity.")
else:
    st.success("Air quality within acceptable limits.")

# ==========================================================
# BEST PRODUCTION MODEL
# ==========================================================

st.markdown("---")
st.markdown("## 🏆 Best Production Model")

best_model_data = fetch_data(BEST_MODEL_URL)

if best_model_data and "model" in best_model_data:

    best_model = best_model_data["model"]

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Model Name", best_model.get("model_name", "N/A"))

    with col2:
        st.metric("RMSE", round(best_model.get("rmse", 0), 2))
        st.metric("R² Score", round(best_model.get("r2", 0), 3))

    if best_model.get("r2", 0) > 0.8:
        st.success("Model performance is strong.")
    else:
        st.warning("Model performance could be improved.")

else:
    st.warning("Best model details unavailable.")

# ==========================================================
# FEATURE IMPORTANCE
# ==========================================================

st.markdown("---")
st.markdown("## 📊 Top Feature Importance")

feature_data = fetch_data(FEATURE_URL)

if feature_data and "features" in feature_data:

    df = pd.DataFrame(feature_data["features"])
    df = df.sort_values("importance", ascending=False).head(10)

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=df["importance"],
        y=df["feature"],
        orientation="h",
        marker_color="#00c3ff"
    ))

    fig.update_layout(
        title="Top 10 Features Affecting AQI",
        template="plotly_dark",
        height=500,
        yaxis=dict(autorange="reversed"),
        paper_bgcolor="#0e1117"
    )

    st.plotly_chart(fig, use_container_width=True)

else:
    st.warning("Feature importance unavailable.")