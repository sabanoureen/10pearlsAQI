import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime

# =========================================================
# CONFIG
# =========================================================
st.set_page_config(page_title="AQI Predictor Dashboard", layout="wide")

API_BASE_URL = "https://10pearlsaqi-production-848d.up.railway.app"

# =========================================================
# SIDEBAR
# =========================================================
st.sidebar.title("🌍 Pearls AQI Predictor")

dashboard_mode = st.sidebar.radio(
    "Navigation",
    ["Forecast Dashboard", "Model Comparison", "SHAP Analysis"]
)

st.sidebar.markdown("### 📍 Location")
st.sidebar.info("Islamabad, Pakistan")

forecast_days = st.sidebar.selectbox(
    "Forecast Horizon (Days)",
    [1, 2, 3, 5, 7],
    index=2
)

if st.sidebar.button("🔄 Refresh Data"):
    st.rerun()

# =========================================================
# HELPER FUNCTIONS
# =========================================================

def get_aqi_category(aqi):
    if aqi <= 50:
        return "Good", "green"
    elif aqi <= 100:
        return "Moderate", "gold"
    elif aqi <= 150:
        return "Unhealthy for Sensitive Groups", "orange"
    elif aqi <= 200:
        return "Unhealthy", "red"
    elif aqi <= 300:
        return "Very Unhealthy", "purple"
    else:
        return "Hazardous", "maroon"


def create_gauge(value, title):

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        title={'text': title},
        gauge={
            'axis': {'range': [0, 500]},
            'steps': [
                {'range': [0, 50], 'color': "green"},
                {'range': [50, 100], 'color': "yellow"},
                {'range': [100, 150], 'color': "orange"},
                {'range': [150, 200], 'color': "red"},
                {'range': [200, 300], 'color': "purple"},
                {'range': [300, 500], 'color': "maroon"},
            ],
        }
    ))

    fig.update_layout(height=300)
    return fig


# =========================================================
# FORECAST DASHBOARD
# =========================================================
if dashboard_mode == "Forecast Dashboard":

    st.title("🌤 AQI Forecast Dashboard")

    # -----------------------------------------
    # FETCH FORECAST
    # -----------------------------------------
    response = requests.get(
        f"{API_BASE_URL}/forecast/multi?days={forecast_days}"
    )

    if response.status_code != 200:
        st.error("❌ API error while fetching forecast.")
        st.stop()

    data = response.json()

    if "predictions" not in data:
        st.error("❌ Invalid forecast response.")
        st.stop()

    df = pd.DataFrame(data["predictions"])

    # Fix column names automatically
    if "predicted_aqi" in df.columns:
        df.rename(columns={"predicted_aqi": "aqi"}, inplace=True)

    if "datetime" in df.columns:
        df.rename(columns={"datetime": "date"}, inplace=True)

    df["date"] = pd.to_datetime(df["date"])

    current_aqi = df["aqi"].iloc[0]
    predicted_aqi = df["aqi"].iloc[1] if len(df) > 1 else current_aqi

    # -----------------------------------------
    # METRICS
    # -----------------------------------------
    col1, col2, col3 = st.columns(3)

    col1.metric("Latest AQI", round(current_aqi, 2))
    col2.metric("Max AQI", round(df["aqi"].max(), 2))
    col3.metric("Average AQI", round(df["aqi"].mean(), 2))

    # -----------------------------------------
    # GAUGES
    # -----------------------------------------
    g1, g2 = st.columns(2)

    g1.plotly_chart(create_gauge(current_aqi, "Current AQI"), use_container_width=True)
    g2.plotly_chart(create_gauge(predicted_aqi, "Next Prediction"), use_container_width=True)

    # -----------------------------------------
    # HEALTH STATUS
    # -----------------------------------------
    status, color = get_aqi_category(current_aqi)

    st.markdown(
        f"<h2 style='color:{color}'>Air Quality Status: {status}</h2>",
        unsafe_allow_html=True
    )

    # -----------------------------------------
    # 3 DAY FORECAST BAR
    # -----------------------------------------
    st.subheader("📅 Multi-Day AQI Forecast")

    fig = px.bar(
        df,
        x="date",
        y="aqi",
        color="aqi",
        title=f"{forecast_days}-Day AQI Forecast",
        labels={"aqi": "Predicted AQI", "date": "Date"},
    )

    st.plotly_chart(fig, use_container_width=True)

# =========================================================
# MODEL COMPARISON
# =========================================================
elif dashboard_mode == "Model Comparison":

    st.title("📊 Model Comparison")

    response = requests.get(f"{API_BASE_URL}/models/metrics")

    if response.status_code != 200:
        st.error("Unable to fetch model metrics.")
        st.stop()

    models = response.json()["models"]

    if not models:
        st.warning("No model metrics available.")
        st.stop()

    df = pd.DataFrame(models)

    best = df.sort_values("rmse").iloc[0]

    st.success(
        f"🏆 Best Model: {best['model_name']} "
        f"(RMSE: {best['rmse']:.2f}, R²: {best['r2']:.3f})"
    )

    st.dataframe(df)

    fig = px.bar(
        df,
        x="model_name",
        y="rmse",
        color="rmse",
        title="RMSE Comparison"
    )

    st.plotly_chart(fig, use_container_width=True)

# =========================================================
# SHAP ANALYSIS
# =========================================================
elif dashboard_mode == "SHAP Analysis":

    st.title("🧠 SHAP Feature Analysis")

    response = requests.get(f"{API_BASE_URL}/forecast/shap")

    if response.status_code != 200:
        st.error("SHAP API error.")
        st.stop()

    data = response.json()

    if "contributions" not in data:
        st.warning("No SHAP data available.")
        st.stop()

    shap_df = pd.DataFrame(data["contributions"])

    shap_df = shap_df.rename(columns={"shap_value": "value"})

    shap_df = shap_df.reindex(
        shap_df["value"].abs().sort_values(ascending=False).index
    ).head(5)

    shap_df["color"] = shap_df["value"].apply(
        lambda x: "green" if x > 0 else "red"
    )

    # BAR
    fig_bar = go.Figure()

    fig_bar.add_trace(go.Bar(
        x=shap_df["value"],
        y=shap_df["feature"],
        orientation="h",
        marker_color=shap_df["color"]
    ))

    fig_bar.update_layout(title="Top 5 Feature Impact")

    st.plotly_chart(fig_bar, use_container_width=True)

    # WATERFALL
    base_value = data["prediction"] - shap_df["value"].sum()

    fig_waterfall = go.Figure(go.Waterfall(
        orientation="h",
        measure=["absolute"] + ["relative"] * len(shap_df),
        y=["Base Value"] + shap_df["feature"].tolist(),
        x=[base_value] + shap_df["value"].tolist(),
    ))

    fig_waterfall.update_layout(
        title="Waterfall Explanation of AQI Prediction"
    )

    st.plotly_chart(fig_waterfall, use_container_width=True)

    st.success(f"Prediction Explained: {data['prediction']:.2f}")
