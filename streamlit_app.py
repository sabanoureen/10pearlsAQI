import streamlit as st
import requests
import pandas as pd
import plotly.express as px

# ================================
# CONFIG
# ================================
API_URL = "https://10pearlsaqi-production-848d.up.railway.app"
st.set_page_config(page_title="AQI Dashboard", layout="wide")

# ================================
# SIDEBAR
# ================================
st.sidebar.title("⚙ Configuration")

horizon = st.sidebar.selectbox(
    "Forecast Horizon (Days)",
    [1, 3, 5],
    index=0
)

page = st.sidebar.radio(
    "Navigation",
    ["Forecast", "Model Comparison", "SHAP Explainability"]
)

# ================================
# TITLE
# ================================
st.title("🌍 AQI Forecast & Explainability Dashboard")

# ============================================================
# 1️⃣ FORECAST PAGE
# ============================================================
if page == "Forecast":

    st.header("📈 AQI Multi-Day Forecast")

    if st.button("Generate Forecast"):

        try:
            response = requests.get(
                f"{API_URL}/forecast/multi",
                params={"horizon": horizon}
            )

            data = response.json()

            if data.get("status") != "success":
                st.error(data.get("detail", "API Error"))
            else:
                df = pd.DataFrame(data["predictions"])
                df["datetime"] = pd.to_datetime(df["datetime"])

                fig = px.line(
                    df,
                    x="datetime",
                    y="predicted_aqi",
                    markers=True,
                    title=f"{horizon}-Day AQI Forecast"
                )

                st.plotly_chart(fig, use_container_width=True)

                st.metric(
                    "Latest Predicted AQI",
                    round(df["predicted_aqi"].iloc[-1], 2)
                )

        except Exception as e:
            st.error(f"API Error: {e}")

# ============================================================
# 2️⃣ MODEL COMPARISON
# ============================================================
elif page == "Model Comparison":

    st.header("📊 Model Performance")

    try:
        response = requests.get(f"{API_URL}/models/metrics")
        data = response.json()

        if data.get("status") != "success":
            st.error("API Error")
        else:
            df = pd.DataFrame(data["models"])

            st.dataframe(df, use_container_width=True)

            fig = px.bar(
                df,
                x="horizon",
                y="rmse",
                color="model_name",
                title="RMSE by Forecast Horizon"
            )

            st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"API Error: {e}")

# ============================================================
# 3️⃣ SHAP PAGE
# ============================================================
elif page == "SHAP Explainability":

    st.header("🧠 SHAP Explainability")

    if st.button("Generate SHAP Explanation"):

        try:
            response = requests.get(
                f"{API_URL}/forecast/shap",
                params={"horizon": horizon}
            )

            data = response.json()

            if data.get("status") != "success":
                st.error(data.get("detail", "API Error"))
            else:
                contributions = pd.DataFrame(data["contributions"])

                contributions = contributions.sort_values(
                    by="shap_value",
                    key=abs,
                    ascending=False
                )

                fig = px.bar(
                    contributions.head(15),
                    x="shap_value",
                    y="feature",
                    orientation="h",
                    title="Top SHAP Feature Contributions"
                )

                st.plotly_chart(fig, use_container_width=True)

                st.metric(
                    "Predicted AQI",
                    round(data["prediction"], 2)
                )

        except Exception as e:
            st.error(f"API Error: {e}")
