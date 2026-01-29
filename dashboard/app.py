import streamlit as st
import requests
import pandas as pd
from datetime import datetime

API_BASE_URL = "https://10pearlsaqi-production-d27d.up.railway.app"

st.set_page_config(
    page_title="AQI Forecast Dashboard",
    layout="centered"
)

st.title("🌫️ Air Quality Index Forecast")
st.caption("Production AQI system • FastAPI + ML • Inspired by 10Pearls")

# ---------------- Sidebar ----------------
st.sidebar.header("Settings")
city = st.sidebar.selectbox("City", ["Karachi"])
refresh = st.sidebar.button("🔄 Refresh Forecast")

@st.cache_data(ttl=300)
def fetch_forecast():
    r = requests.get(f"{API_BASE_URL}/predict/multi", timeout=10)
    r.raise_for_status()
    return r.json()

if refresh:
    st.cache_data.clear()

try:
    data = fetch_forecast()
    preds = data["predictions"]

    c1, c2, c3 = st.columns(3)
    c1.metric("AQI (1 Hour)", preds["1h"])
    c2.metric("AQI (6 Hours)", preds["6h"])
    c3.metric("AQI (24 Hours)", preds["24h"])

    df = pd.DataFrame({
        "Horizon (hours)": [1, 6, 24],
        "AQI": [preds["1h"], preds["6h"], preds["24h"]]
    })

    st.subheader("📈 AQI Forecast Trend")
    st.line_chart(df.set_index("Horizon (hours)"))

    with st.expander("ℹ️ Model Details"):
        st.write(f"Model: {data['model']}")
        st.write(f"RMSE: {data['rmse']}")
        st.write(f"R² Score: {data['r2']}")
        st.write(f"Last updated: {datetime.utcnow()} UTC")

except Exception as e:
    st.error("Failed to fetch data from API")
    st.code(str(e))