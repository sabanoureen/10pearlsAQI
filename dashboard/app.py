# ===============================
# AQI DASHBOARD (STREAMLIT)
# ===============================

import os
import requests
import pandas as pd
import plotly.express as px
import streamlit as st

# -------------------------------
# PAGE CONFIG (MUST BE FIRST)
# -------------------------------
st.set_page_config(
    page_title="AQI Dashboard",
    layout="wide"
)

# -------------------------------
# CONFIG
# -------------------------------
API_BASE_URL = os.getenv(
    "API_BASE_URL",
    "https://10pearlsaqi-production-848d.up.railway.app"
)

TIMEOUT = 10

# -------------------------------
# SAFE API HELPERS
# -------------------------------
def safe_get(path, params=None):
    try:
        r = requests.get(
            f"{API_BASE_URL}{path}",
            params=params,
            timeout=TIMEOUT
        )
        if r.status_code == 200:
            return r.json()
        else:
            st.error(f"GET {path} failed ({r.status_code})")
    except Exception as e:
        st.error(f"GET {path} error: {e}")
    return None


def safe_post(path, payload):
    try:
        r = requests.post(
            f"{API_BASE_URL}{path}",
            json=payload,
            timeout=TIMEOUT
        )
        if r.status_code == 200:
            return r.json()
        else:
            st.error(f"POST {path} failed ({r.status_code})")
    except Exception as e:
        st.error(f"POST {path} error: {e}")
    return None


# -------------------------------
# HEADER
# -------------------------------
st.title("🌍 AQI Prediction Dashboard")
st.caption("Live predictions powered by FastAPI + MongoDB")

# -------------------------------
# API HEALTH
# -------------------------------
with st.expander("🔍 API Health Check", expanded=False):
    health = safe_get("/health")
    if health:
        st.success("API is healthy")
        st.json(health)

# ===============================
# SINGLE AQI PREDICTION
# ===============================
st.header("🔮 Single AQI Prediction")

horizon = st.slider(
    "Forecast horizon (days)",
    min_value=1,
    max_value=7,
    value=1
)

if st.button("Predict AQI", key="single"):
    result = safe_get("/predict", params={"horizon": horizon})

    if result:
        st.success("Prediction successful")
        st.json(result)

# ===============================
# MULTI AQI PREDICTION
# ===============================
st.header("📈 Multi-Horizon AQI Forecast")

selected_horizons = st.multiselect(
    "Select forecast horizons (days)",
    options=[1, 2, 3, 4, 5, 6, 7],
    default=[1, 3, 5]
)

if st.button("Run Multi Forecast", key="multi"):
    payload = {"horizons": selected_horizons}
    response = safe_post("/predict/multi", payload)

    if response and "predictions" in response:
        df = pd.DataFrame(response["predictions"])

        st.subheader("📊 Prediction Table")
        st.dataframe(df, use_container_width=True)

        st.subheader("📉 AQI Forecast Chart")
        fig = px.line(
            df,
            x="horizon",
            y="aqi",
            markers=True,
            title="AQI vs Forecast Horizon"
        )
        st.plotly_chart(fig, use_container_width=True)

# ===============================
# BEST MODEL VIEW
# ===============================
st.header("🏆 Best Model")

model_horizon = st.selectbox(
    "Select horizon",
    options=[1, 2, 3, 4, 5]
)

model_info = safe_get("/models/best", params={"horizon": model_horizon})

if model_info and "model" in model_info:
    st.success("Best model found")
    st.json(model_info["model"])

# -------------------------------
# FOOTER
# -------------------------------
st.markdown("---")
st.caption("🚀 Deployed on Railway | Powered by FastAPI & Streamlit")
