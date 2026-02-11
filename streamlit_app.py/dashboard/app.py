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

    response = requests.post(API_URL + "/predict_multi", json={"horizons": horizons})
    results = response.json()

    # ===============================
    # CLEAN DATAFRAME
    # ===============================
    rows = []

    for key, value in results.items():
        if value["status"] == "success":
            rows.append({
                "horizon": int(key.replace("h", "")),
                "predicted_aqi": value["predicted_aqi"]
            })

    if not rows:
        st.error("No successful predictions received.")
        st.stop()

    df = pd.DataFrame(rows).sort_values("horizon")

    # ===============================
    # METRICS
    # ===============================
    st.subheader("Model Information")

    col1, col2, col3 = st.columns(3)
    col1.metric("Best Model", "Gradient Boosting")
    col2.metric("R² Score", "0.85")
    col3.metric("RMSE", "0.34")

    # ===============================
    # CHART
    # ===============================
    st.subheader("AQI Forecast")

    fig = px.line(
        df,
        x="horizon",
        y="predicted_aqi",
        markers=True,
        title="AQI Forecast"
    )

    st.plotly_chart(fig, use_container_width=True)

    # ===============================
    # TABLE
    # ===============================
    st.subheader("Forecast Details")
    st.dataframe(df)

    # ===============================
    # ALERTS
    # ===============================
    if df["predicted_aqi"].max() > 150:
        st.error("⚠️ Poor air quality expected.")
    else:
        st.success("✅ Air quality within acceptable limits.")
