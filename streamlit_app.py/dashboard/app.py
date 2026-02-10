import os
import requests
import streamlit as st
import pandas as pd
import plotly.express as px

# -------------------------------
# Page Config (MUST be first)
# -------------------------------
st.set_page_config(
    page_title="AQI Prediction Dashboard",
    layout="wide"
)

# -------------------------------
# Config
# -------------------------------
API_BASE_URL = os.getenv(
    "API_BASE_URL",
    "https://10pearlsaqi-production-848d.up.railway.app"
)

TIMEOUT = 5  # lower timeout to fail fast

# -------------------------------
# Helpers
# -------------------------------
def safe_get(path, params=None):
    try:
        r = requests.get(
            f"{API_BASE_URL}{path}",
            params=params,
            timeout=TIMEOUT
        )
        r.raise_for_status()
        return r.json()
    except Exception as e:
        st.error(f"GET {path} failed: {e}")
        return None


def safe_post(path, payload):
    try:
        r = requests.post(
            f"{API_BASE_URL}{path}",
            json=payload,
            timeout=TIMEOUT
        )
        r.raise_for_status()
        return r.json()
    except Exception as e:
        st.error(f"POST {path} failed: {e}")
        return None

# -------------------------------
# UI
# -------------------------------
st.title("🌍 AQI Prediction Dashboard")
st.caption("Live predictions powered by FastAPI + MongoDB")

# -------------------------------
# Health Check
# -------------------------------
with st.expander("🔍 API Health Check", expanded=True):
    health = safe_get("/health")
    if health:
        st.success("API is healthy")
        st.json(health)

# -------------------------------
# Single Prediction
# -------------------------------
st.header("🔮 Single AQI Prediction")

horizon = st.slider(
    "Forecast horizon (days)",
    min_value=1,
    max_value=7,
    value=1
)

if st.button("Predict AQI"):
    result = safe_get("/predict", params={"horizon": horizon})
    if result:
        st.success("Prediction received")
        st.json(result)

# -------------------------------
# Multi-Horizon Prediction
# -------------------------------
st.header("📊 Multi-Horizon AQI Forecast")

horizons = st.multiselect(
    "Select forecast horizons (days)",
    options=[1, 3, 5, 7],
    default=[1, 3, 5]
)

if st.button("Run Multi Forecast"):
    result = safe_post(
        "/predict/multi",
        {"horizons": horizons}
    )
    if result:
        df = pd.DataFrame(result["predictions"])
        st.dataframe(df)

        fig = px.line(
            df,
            x="horizon",
            y="prediction",
            title="AQI Forecast"
        )
        st.plotly_chart(fig, use_container_width=True)

# -------------------------------
# Best Model
# -------------------------------
st.header("🏆 Best Model")

best_horizon = st.selectbox(
    "Select horizon",
    options=[1, 3, 5, 7]
)

best_model = safe_get(
    "/models/best",
    params={"horizon": best_horizon}
)

if best_model:
    st.json(best_model["model"])
