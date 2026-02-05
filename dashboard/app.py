import os
import requests
import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime

# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(
    page_title="AQI Forecast Dashboard",
    page_icon="🌍",
    layout="wide",
)

# =====================================================
# CONFIG
# =====================================================
import os

API_BASE_URL = os.getenv(
    "API_BASE_URL",
    "https://10pearlsaqi-production-d27.up.railway.app"
)


REQUEST_TIMEOUT = 20


# =====================================================
# SAFE API HELPERS
# =====================================================
def safe_get(path, params=None):
    try:
        r = requests.get(
            f"{API_BASE_URL}{path}",
            params=params,
            timeout=REQUEST_TIMEOUT,
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
            timeout=REQUEST_TIMEOUT,
        )
        r.raise_for_status()
        return r.json()
    except Exception as e:
        st.error(f"API POST error: {e}")
        return None


# =====================================================
# SIDEBAR
# =====================================================
st.sidebar.title("⚙️ Controls")

forecast_days = st.sidebar.slider(
    "Forecast days",
    min_value=1,
    max_value=7,
    value=3,
)

if st.sidebar.button("🔄 Refresh"):
    st.rerun()


# =====================================================
# HEADER
# =====================================================
st.title("🌍 AQI Forecast Dashboard")
st.caption("Production AQI predictions powered by ML ensemble models")


# =====================================================
# BEST MODEL
# =====================================================
st.subheader("🏆 Best Production Model (1h horizon)")

best_model = safe_get(
    "/models/best",
    params={"horizon": 1},
)

if best_model and best_model.get("status") == "success":
    model = best_model["model"]

    c1, c2, c3 = st.columns(3)
    c1.metric("Model", model["model_name"])
    c2.metric("RMSE", round(model["rmse"], 3))
    c3.metric("R²", round(model["r2"], 3))
else:
    st.warning("No production model found")


# =====================================================
# CURRENT PREDICTION
# =====================================================
st.subheader("📈 Current AQI Prediction (1h)")

prediction = safe_get(
    "/predict",
    params={"horizon": 1},
)

if prediction and prediction.get("status") == "success":
    st.metric(
        "Predicted AQI",
        round(prediction["predicted_aqi"], 2),
    )
    st.caption(
        f"Model: **{prediction['model_name']}** | "
        f"Version: `{prediction.get('version', 'latest')}`"
    )
else:
    st.warning("Prediction unavailable")


# =====================================================
# MULTI-DAY FORECAST
# =====================================================
st.subheader("📊 Multi-day AQI Forecast")

horizons = [d * 24 for d in range(1, forecast_days + 1)]

multi = safe_post(
    "/predict/multi",
    payload={"horizons": horizons},
)

if multi and multi.get("status") == "success":
    rows = []

    for h, v in multi["predictions"].items():
        if isinstance(v, (int, float)):
            rows.append(
                {
                    "Horizon (hours)": int(h),
                    "AQI": v,
                }
            )

    df = pd.DataFrame(rows).sort_values("Horizon (hours)")

    fig = px.line(
        df,
        x="Horizon (hours)",
        y="AQI",
        markers=True,
        title="AQI Forecast Trend",
    )

    st.plotly_chart(fig, use_container_width=True)
else:
    st.warning("Multi-day forecast unavailable")


# =====================================================
# FOOTER
# =====================================================
st.divider()
st.caption(f"Last updated: {datetime.utcnow().isoformat()} UTC")
