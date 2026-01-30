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
    "http://localhost:8000"  # local fallback
)

REQUEST_TIMEOUT = 5  # seconds


# -----------------------------------
# Helper functions
# -----------------------------------
def safe_get(url, params=None):
    try:
        r = requests.get(url, params=params, timeout=REQUEST_TIMEOUT)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        st.error(f"API error: {e}")
        return None


def fetch_best_model(horizon: int):
    return safe_get(
        f"{API_BASE_URL}/models/best",
        params={"horizon": horizon}
    )


def fetch_prediction(horizon: int):
    return safe_get(
        f"{API_BASE_URL}/predict",
        params={"horizon": horizon}
    )


def fetch_multi_prediction(horizons):
    params = [("horizons", h) for h in horizons]
    return safe_get(
        f"{API_BASE_URL}/predict/multi",
        params=params
    )


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
st.caption("Real-time AQI predictions powered by ML")


# -----------------------------------
# Best model info
# -----------------------------------
st.subheader("🏆 Best Model (Horizon = 1h)")

best_model = fetch_best_model(horizon=1)

if best_model and best_model.get("status") == "ok":
    model = best_model["best_model"]
    st.success("Best model loaded")

    col1, col2, col3 = st.columns(3)
    col1.metric("Model", model["model_name"])
    col2.metric("RMSE", round(model["rmse"], 2))
    col3.metric("R²", round(model["r2"], 3))
else:
    st.warning("No best model available")


# -----------------------------------
# Single prediction
# -----------------------------------
st.subheader("📈 Current AQI Prediction")

prediction = fetch_prediction(horizon=1)

if prediction and prediction.get("status") == "ok":
    st.metric(
        label="Predicted AQI (1h)",
        value=prediction["predicted_aqi"]
    )
else:
    st.warning("Prediction unavailable")


# -----------------------------------
# Multi-day forecast
# -----------------------------------
st.subheader("📊 Multi-day Forecast")

horizons = [h * 24 for h in range(1, forecast_days + 1)]
multi = fetch_multi_prediction(horizons)

if multi and multi.get("status") == "success":
    data = []
    for k, v in multi["predictions"].items():
        if isinstance(v, dict):
            continue
        data.append({
            "Horizon": k,
            "AQI": v
        })

    df = pd.DataFrame(data)

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
