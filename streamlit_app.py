import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go

# =========================================
# CONFIG
# =========================================
API_URL = "https://10pearlsaqi-production-848d.up.railway.app"

st.set_page_config(
    page_title="AQI Predictor Dashboard",
    page_icon="🌍",
    layout="wide"
)

# =========================================
# AQI STATUS FUNCTION
# =========================================
def get_aqi_status(aqi):
    if aqi <= 50:
        return "Good", "green"
    elif aqi <= 100:
        return "Moderate", "gold"
    elif aqi <= 150:
        return "Unhealthy (Sensitive)", "orange"
    elif aqi <= 200:
        return "Unhealthy", "red"
    else:
        return "Hazardous", "purple"


# =========================================
# SIDEBAR
# =========================================
st.sidebar.header("⚙ Configuration")

horizon = st.sidebar.selectbox(
    "Forecast Horizon (Days)",
    [1, 3, 7],
    index=1
)

# =========================================
# HEADER
# =========================================
st.title("🌍 AQI Predictor Dashboard")
st.markdown("### Machine Learning powered air quality forecasting system")
st.divider()

# =========================================
# LOAD FORECAST
# =========================================
if st.button("📊 Load Forecast"):

    response = requests.get(f"{API_URL}/forecast/multi?days={horizon}")

    if response.status_code != 200:
        st.error("Failed to fetch forecast.")
        st.stop()

    results = response.json()
    df = pd.DataFrame(results["predictions"])
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values("datetime")

    latest_aqi = df["predicted_aqi"].iloc[-1]
    max_aqi = df["predicted_aqi"].max()
    avg_aqi = df["predicted_aqi"].mean()

    status, color = get_aqi_status(latest_aqi)

    # ================= KPI =================
    col1, col2, col3 = st.columns(3)
    col1.metric("Latest AQI", round(latest_aqi, 2))
    col2.metric("Max AQI", round(max_aqi, 2))
    col3.metric("Average AQI", round(avg_aqi, 2))

    st.markdown(
        f"<h3 style='color:{color}'>Air Quality Status: {status}</h3>",
        unsafe_allow_html=True
    )

    # ================= FORECAST GRAPH =================
    st.subheader("📈 Forecast Trend")

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df["datetime"],
            y=df["predicted_aqi"],
            mode="lines+markers",
            name="Forecast AQI"
        )
    )

    fig.update_layout(
        template="plotly_white",
        height=450
    )

    st.plotly_chart(fig, use_container_width=True)

else:
    st.info("Click 'Load Forecast' to view predictions.")

# =====================================================
# SHAP SECTION
# =====================================================
st.divider()
st.subheader("🧠 Model Explainability (SHAP)")

if st.button("Show SHAP Analysis"):

    shap_res = requests.get(f"{API_URL}/forecast/shap")

    if shap_res.status_code != 200:
        st.error("SHAP analysis failed.")
        st.stop()

    shap_data = shap_res.json()

    shap_df = pd.DataFrame(shap_data["contributions"])

    # ================= TOP 5 FEATURES =================
    shap_df["abs_value"] = shap_df["shap_value"].abs()
    shap_df = shap_df.sort_values("abs_value", ascending=False).head(5)

    prediction = shap_data["prediction"]
    model_version = shap_data["model_version"]

    # ================= COLOR CODING =================
    shap_df["color"] = shap_df["shap_value"].apply(
        lambda x: "green" if x > 0 else "red"
    )

    # ================= WATERFALL =================
    st.subheader("📊 Waterfall Explanation of AQI Prediction")

    fig = go.Figure()

    # Base value (start from 0 for simplicity)
    base_value = 0
    cumulative = base_value

    fig.add_trace(
        go.Bar(
            x=[base_value],
            y=["Base Value"],
            orientation="h",
            marker_color="gray",
            name="Base"
        )
    )

    for _, row in shap_df.iterrows():
        fig.add_trace(
            go.Bar(
                x=[row["shap_value"]],
                y=[row["feature"]],
                orientation="h",
                marker_color=row["color"],
                name=row["feature"]
            )
        )
        cumulative += row["shap_value"]

    fig.update_layout(
        template="plotly_white",
        height=500,
        showlegend=False,
        xaxis_title="Impact on AQI"
    )

    st.plotly_chart(fig, use_container_width=True)

    # ================= PREDICTION BOX =================
    st.success(f"Prediction explained: {round(prediction,2)}")

    st.info(f"Model Version: {model_version}")
