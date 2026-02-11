import streamlit as st
import requests
import pandas as pd
import plotly.express as px

API_URL = "https://10pearlsaqi-production-848d.up.railway.app"

st.set_page_config(page_title="AQI 3-Day Forecast", layout="wide")

# ----------------------------------------
# HEADER
# ----------------------------------------
st.title("🌍 AQI 3-Day Forecast Dashboard")
st.markdown("Machine Learning powered air quality forecasting")

# ----------------------------------------
# SIDEBAR
# ----------------------------------------
st.sidebar.header("Configuration")
horizon = st.sidebar.selectbox("Forecast Horizon (Days)", [1, 3, 7], index=1)

# ----------------------------------------
# GENERATE BUTTON
# ----------------------------------------
if st.sidebar.button("Generate New Forecast"):
    with st.spinner("Generating forecast..."):
        requests.get(f"{API_URL}/forecast/generate?horizon={horizon}")
    st.success("Forecast generated successfully!")

# ----------------------------------------
# LOAD STORED FORECAST
# ----------------------------------------
if st.button("Load Latest Forecast"):

    response = requests.get(
        f"{API_URL}/forecast/latest?horizon={horizon}"
    )

    results = response.json()

    if results.get("status") != "success":
        st.error("No forecast available yet.")
        st.stop()

    df = pd.DataFrame(results["predictions"])

    if df.empty:
        st.error("Forecast data empty.")
        st.stop()

    df["datetime"] = pd.to_datetime(df["datetime"])

    # ----------------------------------------
    # INFO
    # ----------------------------------------
    st.subheader("Forecast Generated At")
    st.write(results["generated_at"])

    latest_aqi = df["predicted_aqi"].iloc[0]

    col1, col2 = st.columns(2)
    col1.metric("Latest AQI", round(latest_aqi, 2))
    col2.metric("Max AQI (Next Days)", round(df["predicted_aqi"].max(), 2))

    # ----------------------------------------
    # CHART
    # ----------------------------------------
    st.subheader("AQI Forecast Trend")

    fig = px.line(
        df,
        x="datetime",
        y="predicted_aqi",
        title=f"{horizon}-Day AQI Forecast",
        markers=True
    )

    st.plotly_chart(fig, use_container_width=True)

    # ----------------------------------------
    # TABLE
    # ----------------------------------------
    st.subheader("Forecast Details")
    st.dataframe(df)

    # ----------------------------------------
    # ALERTS
    # ----------------------------------------
    if df["predicted_aqi"].max() > 150:
        st.error("⚠️ Poor air quality expected.")
    else:
        st.success("✅ Air quality within acceptable limits.")
