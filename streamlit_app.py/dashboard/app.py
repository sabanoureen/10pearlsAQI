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
        API_URL + "/predict_multi",
        json={"horizons": horizons}
    )

    try:
        results = response.json()
    except:
        st.error("Invalid response from API")
        st.stop()

    if not isinstance(results, dict):
        st.error("API did not return valid prediction data")
        st.write(results)
        st.stop()

    rows = []

    for key, value in results.items():
        if isinstance(value, dict) and value.get("status") == "success":
            rows.append({
                "horizon": int(key.replace("h", "")),
                "predicted_aqi": value["predicted_aqi"]
            })

    if not rows:
        st.error("No successful predictions received.")
        st.write(results)
        st.stop()

    df = pd.DataFrame(rows).sort_values("horizon")
