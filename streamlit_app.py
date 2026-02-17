import streamlit as st
import requests
import pandas as pd
import plotly.express as px

# ================================
# CONFIG
# ================================
API_URL = "https://alert-cat-production.up.railway.app"

st.set_page_config(page_title="AQI Dashboard", layout="wide")

# ================================
# SIDEBAR
# ================================
st.sidebar.title("⚙ Configuration")

horizon = st.sidebar.selectbox(
    "Forecast Horizon (Days)",
    [1, 3, 5],
    index=0
)

page = st.sidebar.radio(
    "Navigation",
    ["Forecast"]
)

# ================================
# TITLE
# ================================
st.title("🌍 AQI Forecast Dashboard")

# ============================================================
# FORECAST PAGE
# ============================================================
if page == "Forecast":

    st.header("📈 AQI Forecast")

    if st.button("Generate Forecast"):

        try:
            response = requests.get(
                f"{API_URL}/forecast/multi",
                params={"horizon": horizon},
                timeout=30
            )

            data = response.json()

            if data.get("status") != "success":
                st.error(data)
            else:
                prediction = data["prediction"]
                forecast_for = data["forecast_for"]

                st.success("Forecast generated")

                st.metric(
                    "Predicted AQI",
                    round(prediction, 2)
                )

                st.write(f"Forecast For: {forecast_for}")

        except Exception as e:
            st.error(f"API Error: {e}")
