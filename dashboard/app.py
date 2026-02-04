import streamlit as st
import requests
import pandas as pd
import plotly.express as px
from datetime import datetime

# -----------------------------------
# Config
# -----------------------------------
st.set_page_config(
    page_title="AQI Forecast Dashboard",
    layout="wide"
)

API_BASE_URL = st.secrets.get(
    "API_BASE_URL",
    "http://localhost:8000"
)

REQUEST_TIMEOUT = 45  # increased for Railway cold starts



# -----------------------------------
# Helpers
# -----------------------------------
def safe_get(url, params=None):
    try:
        r = requests.get(url, params=params, timeout=REQUEST_TIMEOUT)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        st.error(f"API error: {e}")
        return None


def safe_post(url, payload):
    try:
        r = requests.post(url, json=payload, timeout=REQUEST_TIMEOUT)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        st.error(f"API error: {e}")
        return None


# -----------------------------------
# Sidebar
# -----------------------------------
st.sidebar.title("⚙️ Settings")

forecast_days = st.sidebar.slider(
    "Forecast days",
    min_value=1,
    max_value=7,
    value=3
)

if st.sidebar.button("🔄 Refresh"):
    st.rerun()


# -----------------------------------
# Header
# -----------------------------------
st.title("🌍 AQI Forecast Dashboard")
st.caption("Real-time AQI predictions powered by ML Ensemble Models")


# -----------------------------------
# Best model (MongoDB)
# -----------------------------------
st.subheader("🏆 Best Production Model (1h)")

best_model = safe_get(
    f"{API_BASE_URL}/models/best",
    params={"horizon": 1}
)

if best_model and best_model.get("status") == "success":
    model = best_model["model"]
    col1, col2, col3 = st.columns(3)
    col1.metric("Model", model["model_name"])
    col2.metric("RMSE", round(model["rmse"], 2))
    col3.metric("R²", round(model["r2"], 3))
else:
    st.warning("No production model found")


# -----------------------------------
# Single prediction
# -----------------------------------
st.subheader("📈 Current AQI Prediction")

prediction = safe_get(
    f"{API_BASE_URL}/predict",
    params={"horizon": 1}
)

if prediction and prediction.get("status") == "success":
    st.metric(
        label="Predicted AQI (1h)",
        value=round(prediction["predicted_aqi"], 2)
    )
    st.caption(
        f"Model: **{prediction['model_name']}** | Version: `{prediction.get('version','legacy')}`"
    )
else:
    st.warning("Prediction unavailable")


# -----------------------------------
# Multi-day forecast
# -----------------------------------
st.subheader("📊 Multi-day AQI Forecast")

horizons = [h * 24 for h in range(1, forecast_days + 1)]

multi = safe_post(
    f"{API_BASE_URL}/predict/multi",
    payload={"horizons": horizons}
)

if multi and multi.get("status") == "success":
    rows = []
    for k, v in multi["predictions"].items():
        if isinstance(v, dict):
            continue
        rows.append({"Horizon": k, "AQI": v})

    df = pd.DataFrame(rows)

    fig = px.line(
        df,
        x="Horizon",
        y="AQI",
        markers=True,
        title="AQI Forecast Trend"
    )

    st.plotly_chart(fig, use_container_width=True)
else:
    st.warning("Multi-day forecast unavailable")


# -----------------------------------
# Footer
# -----------------------------------
st.divider()
st.caption(f"Last updated: {datetime.utcnow().isoformat()} UTC")
