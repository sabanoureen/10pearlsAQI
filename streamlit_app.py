import streamlit as st
import requests
import pandas as pd
import plotly.express as px

# ==============================
# CONFIG
# ==============================
API_URL = "https://alert-cat-production.up.railway.app"

st.set_page_config(
    page_title="AQI Forecast Dashboard",
    layout="wide",
    page_icon="🌍"
)

# ==============================
# SIDEBAR
# ==============================
st.sidebar.title("⚙️ Configuration")

horizon = st.sidebar.selectbox(
    "Forecast Horizon (Days)",
    [1, 2, 3],
    index=0
)

page = st.sidebar.radio(
    "Navigation",
    [
        "📈 Forecast",
        "📊 Model Comparison",
        "🧠 SHAP Analysis",
        "🏆 Best Model",
        "⭐ Important Features"
    ]
)

st.title("🌍 AQI Forecast & Explainability Dashboard")

# =====================================================
# FORECAST
# =====================================================
if page == "📈 Forecast":

    st.header(f"{horizon}-Day AQI Forecast")

    if st.button("Generate Forecast"):

        try:
            r = requests.get(
                f"{API_URL}/forecast/multi",
                params={"horizon": horizon},
                timeout=20
            )
            data = r.json()

            if data["status"] != "success":
                st.error("API error")
            else:
                pred = data["prediction"]
                forecast_time = data["forecast_for"]

                st.success("Forecast generated")

                col1, col2 = st.columns(2)

                col1.metric("Predicted AQI", f"{pred:.2f}")
                col2.metric("Forecast Horizon", f"{horizon} day")

                st.caption(f"Forecast for: {forecast_time}")

        except Exception as e:
            st.error(f"API Error: {e}")

# =====================================================
# MODEL COMPARISON
# =====================================================
elif page == "📊 Model Comparison":

    st.header("Model Performance Comparison")

    try:
        r = requests.get(f"{API_URL}/models/metrics", timeout=20)
        data = r.json()

        if data["status"] != "success":
            st.error("No metrics available")
        else:
            df = pd.DataFrame(data["models"])

            st.dataframe(df, use_container_width=True)

            fig = px.bar(
                df,
                x="model_name",
                y="rmse",
                color="horizon",
                barmode="group",
                title="RMSE Comparison"
            )
            st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"API Error: {e}")

# =====================================================
# SHAP
# =====================================================
elif page == "🧠 SHAP Analysis":

    st.header("SHAP Feature Contributions")

    if st.button("Generate SHAP"):

        try:
            r = requests.get(
                f"{API_URL}/forecast/shap",
                params={"horizon": horizon},
                timeout=30
            )
            data = r.json()

            if data["status"] != "success":
                st.error("SHAP error")
            else:
                df = pd.DataFrame(data["contributions"])
                df = df.sort_values(
                    "shap_value",
                    key=abs,
                    ascending=False
                )

                fig = px.bar(
                    df.head(15),
                    x="shap_value",
                    y="feature",
                    orientation="h",
                    title="Top SHAP Features"
                )
                st.plotly_chart(fig, use_container_width=True)

                st.metric("Predicted AQI", f"{data['prediction']:.2f}")

        except Exception as e:
            st.error(f"API Error: {e}")

# =====================================================
# BEST MODEL
# =====================================================
elif page == "🏆 Best Model":

    st.header("Production Model")

    try:
        r = requests.get(f"{API_URL}/models/best", timeout=10)
        data = r.json()

        if data["status"] != "success":
            st.error("No production model")
        else:
            col1, col2, col3 = st.columns(3)

            col1.metric("Model", data["model_name"])
            col2.metric("RMSE", f"{data['rmse']:.2f}")
            col3.metric("MAE", f"{data['mae']:.2f}")

            st.caption(f"Horizon: {data['horizon']}")

    except Exception as e:
        st.error(f"API Error: {e}")

# =====================================================
# IMPORTANT FEATURES
# =====================================================
elif page == "⭐ Important Features":

    st.header("Feature Importance")

    try:
        r = requests.get(
            f"{API_URL}/models/features",
            params={"horizon": horizon},
            timeout=20
        )
        data = r.json()

        if data["status"] != "success":
            st.error("No feature importance")
        else:
            df = pd.DataFrame(data["features"])

            fig = px.bar(
                df.head(15),
                x="importance",
                y="feature",
                orientation="h",
                title="Top Important Features"
            )
            st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"API Error: {e}")
