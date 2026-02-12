import streamlit as st
import requests
import pandas as pd
import plotly.express as px

API_URL = "https://10pearlsaqi-production-848d.up.railway.app"

st.set_page_config(page_title="AQI Forecast", layout="wide")

st.title("🌍 AQI Forecast Dashboard")
st.markdown("Machine Learning powered air quality forecasting")

st.sidebar.header("Configuration")
horizon = st.sidebar.selectbox("Forecast Horizon (Days)", [1, 3, 7], index=1)

# Generate new forecast
if st.sidebar.button("Generate New Forecast"):
    with st.spinner("Generating forecast..."):
        requests.get(f"{API_URL}/forecast/generate?horizon={horizon}")
    st.success("Forecast generated successfully!")

st.divider()

# Load latest forecast
response = None

if st.button("Load Latest Forecast"):
    response = requests.get(f"{API_URL}/forecast/latest?horizon={horizon}")

if response is None:
    st.info("Click 'Load Latest Forecast' to fetch data.")
    st.stop()

if response.status_code != 200:
    st.error("Failed to fetch forecast.")
    st.stop()

results = response.json()

if results.get("status") != "success":
    st.error("No forecast available.")
    st.stop()

df = pd.DataFrame(results["predictions"])
df["datetime"] = pd.to_datetime(df["datetime"])

st.subheader("Forecast Generated At")
st.write(results["generated_at"])

latest_aqi = df["predicted_aqi"].iloc[0]

col1, col2 = st.columns(2)
col1.metric("Latest AQI", round(latest_aqi, 2))
col2.metric("Max AQI", round(df["predicted_aqi"].max(), 2))

st.subheader("AQI Forecast Trend")

fig = px.line(
    df,
    x="datetime",
    y="predicted_aqi",
    title=f"{horizon}-Day AQI Forecast",
    markers=True
)

st.plotly_chart(fig, use_container_width=True)

st.subheader("Forecast Details")
st.dataframe(df)

if df["predicted_aqi"].max() > 150:
    st.error("⚠️ Poor air quality expected.")
else:
    st.success("✅ Air quality within acceptable limits.")
