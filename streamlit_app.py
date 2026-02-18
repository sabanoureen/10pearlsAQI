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


# ==============================
# HELPER FUNCTION
# ==============================
def call_api(endpoint, params=None, timeout=20):
    try:
        r = requests.get(
            f"{API_URL}{endpoint}",
            params=params,
            timeout=timeout
        )

        if r.status_code != 200:
            return None, f"API returned {r.status_code}"

        return r.json(), None

    except Exception as e:
        return None, str(e)


# =====================================================
# FORECAST
# =====================================================
if page == "📈 Forecast":

    st.header(f"{horizon}-Day AQI Forecast")

    if st.button("Generate Forecast"):

        data, error = call_api(
            "/forecast/multi",
            params={"horizon": horizon}
        )

        if error:
            st.error(f"API Error: {error}")
        else:
            pred = data.get("prediction")
            forecast_time = data.get("forecast_for")

            if pred is None:
                st.error("Invalid API response")
            else:
                st.success("Forecast generated")

                col1, col2 = st.columns(2)

                col1.metric("Predicted AQI", f"{pred:.2f}")
                col2.metric("Forecast Horizon", f"{horizon} day")

                if forecast_time:
                    st.caption(f"Forecast for: {forecast_time}")


# =====================================================
# MODEL COMPARISON
# =====================================================
elif page == "📊 Model Comparison":

    st.header("Model Performance Comparison")

    data, error = call_api("/models/metrics")

    if error:
        st.error(error)
    else:
        models = data.get("models", [])

        if not models:
            st.warning("No metrics available")
        else:
            df = pd.DataFrame(models)

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


# =====================================================
# SHAP
# =====================================================
elif page == "🧠 SHAP Analysis":

    st.header("SHAP Feature Contributions")

    if st.button("Generate SHAP"):

        data, error = call_api(
            "/forecast/shap",
            params={"horizon": horizon},
            timeout=30
        )

        if error:
            st.error(error)
        else:
            contributions = data.get("contributions", [])
            pred = data.get("prediction")

            if not contributions:
                st.warning("No SHAP data")
            else:
                df = pd.DataFrame(contributions)
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

                if pred is not None:
                    st.metric("Predicted AQI", f"{pred:.2f}")


# =====================================================
# BEST MODEL
# =====================================================
elif page == "🏆 Best Model":

    st.header("Production Model")

    data, error = call_api("/models/best")

    if error:
        st.error(error)
    else:
        model_name = data.get("model_name")
        rmse = data.get("rmse")
        mae = data.get("mae")
        horizon_val = data.get("horizon")

        if model_name is None:
            st.warning("No production model")
        else:
            col1, col2, col3 = st.columns(3)

            col1.metric("Model", model_name)
            col2.metric("RMSE", f"{rmse:.2f}")
            col3.metric("MAE", f"{mae:.2f}")

            if horizon_val:
                st.caption(f"Horizon: {horizon_val}")


# =====================================================
# IMPORTANT FEATURES
# =====================================================
elif page == "⭐ Important Features":

    st.header("Feature Importance")

    data, error = call_api(
        "/models/features",
        params={"horizon": horizon}
    )

    if error:
        st.error(error)
    else:
        features = data.get("features", [])

        if not features:
            st.warning("No feature importance")
        else:
            df = pd.DataFrame(features)

            fig = px.bar(
                df.head(15),
                x="importance",
                y="feature",
                orientation="h",
                title="Top Important Features"
            )
            st.plotly_chart(fig, use_container_width=True)
