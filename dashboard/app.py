import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime

# -----------------------------------
# CONFIG
# -----------------------------------
API_BASE_URL = "https://10pearlsaqi-production-d27d.up.railway.app"
CITY = "Karachi"

st.set_page_config(
    page_title="AQI Forecast Dashboard",
    layout="wide",
)

# -----------------------------------
# HELPERS
# -----------------------------------
def fetch_best_model(horizon=1):
    r = requests.get(f"{API_BASE_URL}/models/best?horizon={horizon}", timeout=10)
    if r.status_code != 200:
        return None
    data = r.json()
    return data.get("best_model")


def fetch_prediction(horizon, version=None):
    params = {"horizon": horizon}
    if version:
        params["version"] = version
    r = requests.get(f"{API_BASE_URL}/predict", params=params, timeout=15)
    return r.json()


def fetch_multi_prediction(horizons):
    params = [("horizons", h) for h in horizons]
    r = requests.get(f"{API_BASE_URL}/predict/multi", params=params, timeout=20)
    return r.json()


def aqi_alert(aqi):
    if aqi <= 50:
        return "Good 🟢", "#2ECC71"
    elif aqi <= 100:
        return "Moderate 🟡", "#F1C40F"
    elif aqi <= 150:
        return "Unhealthy (Sensitive) 🟠", "#E67E22"
    elif aqi <= 200:
        return "Unhealthy 🔴", "#E74C3C"
    else:
        return "Very Unhealthy ⚠️", "#8E44AD"


# -----------------------------------
# SIDEBAR
# -----------------------------------
st.sidebar.title("⚙️ Settings")

forecast_days = st.sidebar.slider("Forecast days", 1, 7, 3)
refresh = st.sidebar.button("🔄 Refresh")

st.sidebar.markdown("---")
st.sidebar.subheader("Model Info")

best_model = fetch_best_model(horizon=1)

if best_model:
    st.sidebar.success(f"Model: {best_model['model_name']}")
    st.sidebar.write(f"Version: `{best_model['version']}`")
    st.sidebar.write(f"RMSE: **{best_model['rmse']:.2f}**")
    st.sidebar.write(f"R²: **{best_model['r2']:.3f}**")
else:
    st.sidebar.warning("Model info unavailable")

# -----------------------------------
# HEADER
# -----------------------------------
st.title("🌍 Air Quality Forecast Dashboard")
st.caption("Production ML system • Real-time AQI forecasting")

if refresh:
    st.cache_data.clear()

# -----------------------------------
# REAL-TIME PREDICTION
# -----------------------------------
st.subheader("📍 Current AQI Prediction")

current = fetch_prediction(horizon=1)

if current.get("status") == "success":
    aqi = current["predicted_aqi"]
    label, color = aqi_alert(aqi)

    st.markdown(
        f"""
        <div style="
            padding:25px;
            border-radius:15px;
            background:{color};
            color:white;
            text-align:center;
        ">
            <h2>{aqi}</h2>
            <h4>{label}</h4>
            <p>City: {CITY}</p>
            <p>Updated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}</p>
        </div>
        """,
        unsafe_allow_html=True
    )
else:
    st.error("Failed to fetch current AQI")

# -----------------------------------
# MULTI-DAY FORECAST
# -----------------------------------
st.subheader("📈 Multi-Day AQI Forecast")

horizons = [24 * i for i in range(1, forecast_days + 1)]
forecast = fetch_multi_prediction(horizons)

if forecast.get("status") == "success":
    data = []
    for h, val in forecast["predictions"].items():
        if isinstance(val, dict):
            continue
        day = int(h.replace("h", "")) // 24
        data.append({"Day": f"Day {day}", "AQI": val})

    df = pd.DataFrame(data)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df["Day"],
            y=df["AQI"],
            mode="lines+markers",
            line=dict(width=3),
        )
    )

    fig.update_layout(
        title="AQI Trend (Next Days)",
        xaxis_title="Day",
        yaxis_title="AQI",
        template="plotly_white"
    )

    st.plotly_chart(fig, use_container_width=True)
else:
    st.warning("Forecast data unavailable")

# -----------------------------------
# ALERT SYSTEM
# -----------------------------------
st.subheader("🚨 Health Advisory")

if current.get("status") == "success":
    label, _ = aqi_alert(current["predicted_aqi"])

    if "Unhealthy" in label:
        st.error("⚠️ Air quality is unhealthy. Reduce outdoor activities.")
    elif "Moderate" in label:
        st.warning("😷 Sensitive groups should take precautions.")
    else:
        st.success("✅ Air quality is safe.")

# -----------------------------------
# FOOTER
# -----------------------------------
st.markdown("---")
st.caption("Powered by FastAPI • MongoDB • GitHub Actions • Streamlit Cloud")
