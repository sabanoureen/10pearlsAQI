import streamlit as st
import requests
import pandas as pd
import plotly.express as px

API_URL = "https://10pearlsaqi-production-848d.up.railway.app"

st.set_page_config(page_title="AQI Predictor", layout="wide")

# ===============================
# HEADER
# ===============================
st.title("🌍 AQI Predictor Dashboard")

st.markdown("Real-time air quality forecasting using ML models.")

# ===============================
# SIDEBAR
# ===============================
st.sidebar.header("Configuration")
city = st.sidebar.text_input("City", "Karachi")

horizons = st.sidebar.multiselect(
    "Forecast Horizons (hours)",
    [1, 3, 5],
    default=[1, 3, 5]
)

# ===============================
# BUTTON
# ===============================
if st.button("Get Predictions"):

    response = requests.post(
        f"{API_URL}/predict/multi",
        json={"horizons": horizons}
    )

    results = response.json()

    # -----------------------------
    # CHECK API STATUS
    # -----------------------------
    if results.get("status") != "success":
        st.error("API error:")
        st.json(results)
        st.stop()

    predictions = results.get("predictions", {})

    if not predictions:
        st.error("No predictions returned.")
        st.stop()

    # -----------------------------
    # BUILD DATAFRAME
    # -----------------------------
    rows = []

    for key, value in predictions.items():
        if value["status"] == "success":
            rows.append({
                "horizon": value["horizon"],
                "predicted_aqi": value["predicted_aqi"]
            })

    if not rows:
        st.error("No successful predictions received.")
        st.json(results)
        st.stop()

    df = pd.DataFrame(rows).sort_values("horizon")

    # -----------------------------
    # MODEL INFO
    # -----------------------------
    st.subheader("Model Information")

    col1, col2 = st.columns(2)
    col1.metric("Model Used", list(predictions.values())[0]["model_name"])
    col2.metric("Latest AQI", round(df["predicted_aqi"].iloc[0], 2))

    # -----------------------------
    # CHART
    # -----------------------------
    st.subheader("AQI Forecast")

    fig = px.line(
        df,
        x="horizon",
        y="predicted_aqi",
        markers=True,
        title="Multi-Horizon AQI Forecast"
    )

    st.plotly_chart(fig, use_container_width=True)

    # -----------------------------
    # TABLE
    # -----------------------------
    st.subheader("Forecast Details")
    st.dataframe(df)

    # -----------------------------
    # ALERTS
    # -----------------------------
    if df["predicted_aqi"].max() > 150:
        st.error("⚠️ Poor air quality expected.")
    else:
        st.success("✅ Air quality within acceptable limits.")
