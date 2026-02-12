import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go

API_URL = "https://10pearlsaqi-production-848d.up.railway.app"

st.set_page_config(
    page_title="AQI Forecast Dashboard",
    page_icon="🌍",
    layout="wide"
)

# -----------------------------------------
# AQI COLOR FUNCTION
# -----------------------------------------
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


# -----------------------------------------
# HEADER
# -----------------------------------------
st.title("🌍 AQI Forecast Dashboard")
st.markdown("### Machine Learning powered air quality forecasting system")

st.divider()

# -----------------------------------------
# SIDEBAR
# -----------------------------------------
st.sidebar.header("⚙ Configuration")

horizon = st.sidebar.selectbox(
    "Forecast Horizon (Days)",
    [1, 3, 7],
    index=1
)

if st.sidebar.button("🔄 Generate New Forecast"):
    with st.spinner("Generating forecast..."):
        requests.get(f"{API_URL}/forecast/generate?horizon={horizon}")
    st.success("Forecast generated successfully!")

# -----------------------------------------
# LOAD FORECAST
# -----------------------------------------
if st.button("📊 Load Latest Forecast"):

    response = requests.get(
        f"{API_URL}/forecast/latest?horizon={horizon}"
    )

    if response.status_code != 200:
        st.error("Failed to fetch forecast.")
        st.stop()

    results = response.json()

    if results.get("status") != "success":
        st.error("No forecast available.")
        st.stop()

    df = pd.DataFrame(results["predictions"])

    if df.empty:
        st.error("Forecast data empty.")
        st.stop()

    df["datetime"] = pd.to_datetime(df["datetime"])

    # -----------------------------------------
    # KPI SECTION
    # -----------------------------------------
    latest_aqi = df["predicted_aqi"].iloc[0]
    max_aqi = df["predicted_aqi"].max()
    avg_aqi = df["predicted_aqi"].mean()

    status, color = get_aqi_status(latest_aqi)

    st.subheader("📅 Forecast Generated At")
    st.info(results["generated_at"])

    col1, col2, col3 = st.columns(3)

    col1.metric("Latest AQI", round(latest_aqi, 2))
    col2.metric("Max AQI (Next Days)", round(max_aqi, 2))
    col3.metric("Average AQI", round(avg_aqi, 2))

    st.markdown(
        f"<h3 style='color:{color}'>Air Quality Status: {status}</h3>",
        unsafe_allow_html=True
    )

    st.divider()

    # -----------------------------------------
    # FORECAST CHART
    # -----------------------------------------
    st.subheader("📈 AQI Forecast Trend")

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=df["datetime"],
            y=df["predicted_aqi"],
            mode="lines+markers",
            line=dict(width=3),
            marker=dict(size=8)
        )
    )

    fig.update_layout(
        template="plotly_white",
        xaxis_title="Date",
        yaxis_title="Predicted AQI",
        height=450
    )

    st.plotly_chart(fig, use_container_width=True)

    # -----------------------------------------
    # TABLE
    # -----------------------------------------
    st.subheader("📋 Forecast Details")
    st.dataframe(df, use_container_width=True)

    # -----------------------------------------
    # ALERT BOX
    # -----------------------------------------
    if max_aqi > 150:
        st.error("⚠ Poor air quality expected in coming days.")
    elif max_aqi > 100:
        st.warning("⚠ Moderate air pollution expected.")
    else:
        st.success("✅ Air quality within acceptable limits.")

else:
    st.info("Click 'Load Latest Forecast' to view predictions.")
