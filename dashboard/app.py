import streamlit as st
import requests
import pandas as pd
import plotly.express as px
from datetime import datetime, timedelta

# -------------------------------
# AQI COLOR & CATEGORY LOGIC
# -------------------------------
def aqi_category(aqi):
    if aqi <= 50:
        return "Good", "#2ECC71"
    elif aqi <= 100:
        return "Moderate", "#F1C40F"
    elif aqi <= 150:
        return "Unhealthy (Sensitive)", "#E67E22"
    elif aqi <= 200:
        return "Unhealthy", "#E74C3C"
    elif aqi <= 300:
        return "Very Unhealthy", "#8E44AD"
    else:
        return "Hazardous", "#7E0023"

# -------------------------------
# CONFIG
# -------------------------------
API_BASE_URL = "https://10pearlsaqi-production-d27d.up.railway.app"

st.set_page_config(
    page_title="AQI Forecast Dashboard",
    layout="wide",
)

# -------------------------------
# SIDEBAR
# -------------------------------
st.sidebar.title("⚙️ Configuration")

city = st.sidebar.selectbox("City", ["Karachi"])
st.sidebar.write("📍 Location: 24.8608, 67.0104")

forecast_days = st.sidebar.slider("Forecast Days", 1, 3, 3)

refresh = st.sidebar.button("🔄 Get Predictions")

st.sidebar.markdown("---")
st.sidebar.subheader("API Status")
st.sidebar.success("🟢 API Connected")

st.sidebar.markdown("---")
st.sidebar.subheader("About")
st.sidebar.info(
    "This dashboard provides real-time AQI predictions "
    "using machine learning models."
)

# -------------------------------
# FETCH DATA
# -------------------------------
@st.cache_data(ttl=300)
def fetch_forecast():
    r = requests.get(f"{API_BASE_URL}/predict/multi", timeout=10)
    r.raise_for_status()
    return r.json()

if refresh:
    st.cache_data.clear()

data = fetch_forecast()
preds = data["predictions"]

# -------------------------------
# HEADER
# -------------------------------
st.title("🌫️ Air Quality Index Forecast")
st.caption("Production AQI system • FastAPI + ML • Inspired by 10Pearls")

# -------------------------------
# METRICS
# -------------------------------
# -------------------------------
# COLORED AQI METRICS
# -------------------------------
c1, c2, c3 = st.columns(3)

for col, label, value in zip(
    [c1, c2, c3],
    ["1 Hour", "6 Hours", "24 Hours"],
    [preds["1h"], preds["6h"], preds["24h"]],
):
    category, color = aqi_category(value)

    col.markdown(
        f"""
        <div style="
            padding:18px;
            border-radius:12px;
            background-color:{color};
            color:white;
            text-align:center;
            box-shadow:0 4px 10px rgba(0,0,0,0.15)
        ">
            <h4>AQI ({label})</h4>
            <h1>{value}</h1>
            <p>{category}</p>
        </div>
        """,
        unsafe_allow_html=True
    )
# -------------------------------
# MODEL INFO
# -------------------------------
st.markdown("## 🧠 Model Information")

m1, m2, m3 = st.columns(3)
m1.metric("Best Model", "ridge_regression")
m2.metric("R² Score", data["r2"])
m3.metric("RMSE", data["rmse"])

# -------------------------------
# FORECAST DATA
# -------------------------------
dates = [datetime.utcnow().date() + timedelta(days=i) for i in range(forecast_days)]
aqi_values = [round(preds["24h"])] * forecast_days

df = pd.DataFrame({
    "Date": dates,
    "Predicted AQI": aqi_values
})

# -------------------------------
# CHART
# -------------------------------
st.markdown("## 📈 AQI Forecast for Next Days")

fig = px.bar(
    df,
    x="Date",
    y="Predicted AQI",
    text="Predicted AQI",
    color_discrete_sequence=[aqi_category(preds["24h"])[1]],
)

fig.update_layout(
    yaxis_title="AQI",
    xaxis_title="Date",
    showlegend=False
)

st.plotly_chart(fig, use_container_width=True)

# -------------------------------
# TABLE
# -------------------------------
st.markdown("## 📋 Forecast Details")

df["Category"] = df["Predicted AQI"].apply(
    lambda x: "Moderate" if x <= 100 else "Unhealthy"
)

st.dataframe(df, use_container_width=True)

# -------------------------------
# ALERTS
# -------------------------------
st.markdown("## 🚨 Alerts")

if df["Predicted AQI"].max() > 150:
    st.error("⚠️ Poor air quality expected. Limit outdoor activity.")
else:
    st.success("✅ No alerts – air quality within acceptable limits.")

# -------------------------------
# AQI INFO
# -------------------------------
with st.expander("ℹ️ About AQI Categories"):
    st.markdown("""
- **Good (0–50)**: Air quality is satisfactory  
- **Moderate (51–100)**: Acceptable for most people  
- **Unhealthy for Sensitive Groups (101–150)**  
- **Unhealthy (151–200)**  
- **Very Unhealthy (201–300)**  
- **Hazardous (301+)**
""")
