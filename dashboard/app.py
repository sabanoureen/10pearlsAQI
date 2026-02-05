import os
import requests
import pandas as pd
import streamlit as st
import plotly.express as px
from datetime import datetime

# -------------------------------
# Config
# -------------------------------
st.set_page_config(
    page_title="AQI Forecast Dashboard",
    layout="wide"
)

API_BASE_URL = os.getenv(
    "API_BASE_URL",
    "https://10pearlsaqi-production-d27.up.railway.app"
)

REQUEST_TIMEOUT = 30

# -------------------------------
# Helpers
# -------------------------------
def safe_get(path, params=None):
    try:
        r = requests.get(
            f"{API_BASE_URL}{path}",
            params=params,
            timeout=REQUEST_TIMEOUT
        )
        r.raise_for_status()
        return r.json()
    except Exception as e:
        st.error(f"API GET error: {e}")
        return None


def safe_post(path, payload):
    try:
        r = requests.post(
            f"{API_BASE_URL}{path}",
            json=payload,
            timeout=REQUEST_TIMEOUT
        )
        r.raise_for_status()
        return r.json()
    except Exception as e:
        st.error(f"API POST error: {e}")
        return None


# -------------------------------
# Sidebar
# -------------------------------
st.sidebar.title("⚙️ Controls")

forecast_days = st.sidebar.slider(
    "Forecast days",
    min_value=1,
    max_value=7,
    value=3
)

if st.sidebar.button("🔄 Refresh"):
    st.rerun()


# -------------------------------
# Header
# -------------------------------
st.title("🌍 AQI Forecast Dashboard")
st.caption("Production AQI predictions powered by ML ensemble models")


# -------------------------------
# Best Production Model
# -------------------------------
st.subheader("🏆 Best Production Model (1h horizon)")

best_model = safe_get(
    "/api/models/best",
    params={"horizon": 1}
)

if best_model and best_model.get("status") == "success":
    model = best_model["model"]
    c1, c2, c3 = st.columns(3)
    c1.metric("Model", model["model_name"])
    c2.metric("RMSE", round(model["rmse"], 2))
    c3.metric("R²", round(model["r2"], 3))
else:
    st.warning("No production model found")


# -------------------------------
# Current AQI Prediction
# -------------------------------
st.subheader("📈 Current AQI Prediction (1h)")

prediction = safe_get(
    "/api/predict",
    params={"horizon": 1}
)

if prediction and prediction.get("status") == "success":
    st.metric(
        "Predicted AQI",
        round(prediction["predicted_aqi"], 2)
    )
    st.caption(
        f"Model: **{prediction['model_name']}** | "
        f"Version: `{prediction.get('version','legacy')}`"
    )
else:
    st.warning("Prediction unavailable")


# -------------------------------
# Multi-day Forecast
# -------------------------------
st.subheader("📊 Multi-day AQI Forecast")

horizons = [h * 24 for h in range(1, forecast_days + 1)]

multi = safe_post(
    "/api/predict/multi",
    {"horizons": horizons}
)

if multi and multi.get("status") == "success":
    df = pd.DataFrame([
        {"Horizon (hrs)": k, "AQI": v}
        for k, v in multi["predictions"].items()
        if isinstance(v, (int, float))
    ])

    fig = px.line(
        df,
        x="Horizon (hrs)",
        y="AQI",
        markers=True,
        title="AQI Forecast Trend"
    )

    st.plotly_chart(fig, use_container_width=True)
else:
    st.warning("Multi-day forecast unavailable")


# -------------------------------
# Footer
# -------------------------------
st.divider()
st.caption(f"Last updated: {datetime.utcnow().isoformat()} UTC")
