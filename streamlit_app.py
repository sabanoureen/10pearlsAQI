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
# FUNCTIONS
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


def fetch_model_metrics():
    try:
        res = requests.get(f"{API_URL}/models/metrics")
        if res.status_code == 200:
            return res.json()
        return None
    except:
        return None


# =========================================
# SIDEBAR
# =========================================
st.sidebar.title("⚙ Configuration")

horizon = st.sidebar.selectbox(
    "Forecast Horizon (Days)",
    [1, 3, 7],
    index=1
)

st.sidebar.divider()

st.sidebar.subheader("About")

st.sidebar.info(
    "This dashboard provides real-time AQI predictions "
    "for the next few days using machine learning models."
)

st.sidebar.subheader("API Status")
st.sidebar.success("API Connected")

# =========================================
# TABS
# =========================================
tab1, tab2 = st.tabs(["📊 Forecast", "🤖 Model Comparison"])

# =========================================
# TAB 1 — FORECAST
# =========================================
with tab1:

    st.title("🌍 AQI Predictor Dashboard")

    if st.button("Get Predictions"):

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

        # KPIs
        col1, col2, col3 = st.columns(3)
        col1.metric("Latest AQI", round(latest_aqi, 2))
        col2.metric("Max AQI", round(max_aqi, 2))
        col3.metric("Average AQI", round(avg_aqi, 2))

        st.markdown(
            f"<h3 style='color:{color}'>Air Quality Status: {status}</h3>",
            unsafe_allow_html=True
        )

        # Forecast Chart
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

    # =========================================
    # AQI CATEGORY SECTION
    # =========================================
    st.divider()
    with st.expander("ℹ About AQI Categories"):

        st.markdown("""
        - **Good (0-50)**: Air quality is satisfactory.  
        - **Moderate (51-100)**: Acceptable for most people.  
        - **Unhealthy for Sensitive Groups (101-150)**: Sensitive groups may experience health effects.  
        - **Unhealthy (151-200)**: Everyone may begin to experience health effects.  
        - **Very Unhealthy (201-300)**: Health alert for everyone.  
        - **Hazardous (301+)**: Serious health warning for entire population.
        """)

# =========================================
# TAB 2 — MODEL COMPARISON
# =========================================
with tab2:

    st.title("🤖 Model Comparison")

    metrics_data = fetch_model_metrics()

    if metrics_data:

        models_df = pd.DataFrame(metrics_data["models"])

        # Highlight Best Model
        best_model = models_df.sort_values("rmse").iloc[0]

        st.success(
            f"Best Model: {best_model['model_name']} "
            f"(RMSE: {round(best_model['rmse'],2)}, "
            f"R²: {round(best_model['r2'],2)})"
        )

        col1, col2 = st.columns(2)
        col1.metric("Best R²", round(best_model["r2"], 3))
        col2.metric("Best RMSE", round(best_model["rmse"], 2))

        # RMSE Comparison Chart
        fig_rmse = go.Figure()
        fig_rmse.add_trace(
            go.Bar(
                x=models_df["model_name"],
                y=models_df["rmse"],
                name="RMSE"
            )
        )

        fig_rmse.update_layout(
            title="RMSE Comparison",
            template="plotly_white"
        )

        st.plotly_chart(fig_rmse, use_container_width=True)

        # R2 Comparison Chart
        fig_r2 = go.Figure()
        fig_r2.add_trace(
            go.Bar(
                x=models_df["model_name"],
                y=models_df["r2"],
                name="R² Score"
            )
        )

        fig_r2.update_layout(
            title="R² Comparison",
            template="plotly_white"
        )

        st.plotly_chart(fig_r2, use_container_width=True)

    else:
        st.warning("Model metrics endpoint not available.")
