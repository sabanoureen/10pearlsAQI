import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta
import joblib

from app.pipelines.inference_multi import predict_next_3_days
from app.db.mongo import get_db

# -------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------
st.set_page_config(
    page_title="Karachi AQI Forecast",
    page_icon="🌍",
    layout="wide"
)

st.title("🌍 Karachi AQI Forecast System")
st.markdown("AI-Powered Multi-Horizon Air Quality Prediction")
st.markdown("---")

# -------------------------------------------------
# REFRESH BUTTON
# -------------------------------------------------
if st.button("🔄 Refresh Live Forecast"):
    st.rerun()

# -------------------------------------------------
# GET PREDICTIONS
# -------------------------------------------------
results = predict_next_3_days()

# -------------------------------------------------
# GAUGE FUNCTION
# -------------------------------------------------
def create_gauge(value, title):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        title={'text': title},
        gauge={
            'axis': {'range': [0, 300]},
            'steps': [
                {'range': [0, 50], 'color': "green"},
                {'range': [50, 100], 'color': "yellow"},
                {'range': [100, 150], 'color': "orange"},
                {'range': [150, 200], 'color': "red"},
                {'range': [200, 300], 'color': "purple"},
            ],
        }
    ))
    fig.update_layout(height=350)
    st.plotly_chart(fig, use_container_width=True)

# -------------------------------------------------
# FORECAST SECTION
# -------------------------------------------------
st.markdown("## 📅 Multi-Day AQI Forecast")

col1, col2, col3 = st.columns(3)

with col1:
    create_gauge(
        results["1_day"]["value"],
        f"1 Day Forecast\n📅 {results['1_day']['date']}\n🤖 {results['1_day']['model']}"
    )

with col2:
    create_gauge(
        results["2_day"]["value"],
        f"2 Day Forecast\n📅 {results['2_day']['date']}\n🤖 {results['2_day']['model']}"
    )

with col3:
    create_gauge(
        results["3_day"]["value"],
        f"3 Day Forecast\n📅 {results['3_day']['date']}\n🤖 {results['3_day']['model']}"
    )

# -------------------------------------------------
# HISTORICAL + FORECAST TREND
# -------------------------------------------------
st.markdown("---")
st.markdown("## 📈 Historical + Forecast Trend")

db = get_db()
data = list(db["historical_hourly_data"].find({}, {"_id": 0}))
df = pd.DataFrame(data)

if not df.empty:
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values("datetime")

    fig = go.Figure()

    # Historical
    fig.add_trace(go.Scatter(
        x=df["datetime"],
        y=df["pm2_5"],
        mode='lines',
        name="Historical PM2.5"
    ))

    # Forecast
    forecast_dates = [
        results["1_day"]["date"],
        results["2_day"]["date"],
        results["3_day"]["date"]
    ]

    forecast_values = [
        results["1_day"]["value"],
        results["2_day"]["value"],
        results["3_day"]["value"]
    ]

    fig.add_trace(go.Scatter(
        x=forecast_dates,
        y=forecast_values,
        mode='markers+lines',
        name="Forecast"
    ))

    fig.update_layout(height=450)
    st.plotly_chart(fig, use_container_width=True)

# -------------------------------------------------
# MODEL COMPARISON TABLE
# -------------------------------------------------
st.markdown("---")
st.markdown("## 📊 Model Comparison (9 Models)")

comparison_df = pd.DataFrame({
    "Horizon": ["H1 (24h)", "H2 (48h)", "H3 (72h)"],
    "Random Forest RMSE": [3.0669, 2.6788, 2.8933],
    "Gradient Boosting RMSE": [9.4081, 9.3588, 9.5649],
    "Ridge RMSE": [12.3637, 13.3617, 13.5450],
})

st.dataframe(comparison_df, use_container_width=True)

# -------------------------------------------------
# BEST MODEL SECTION
# -------------------------------------------------
st.markdown("---")
st.markdown("## 🏆 Best Model Selection")

st.success("""
Random Forest achieved the lowest RMSE
across all forecast horizons (24h, 48h, 72h).

Therefore, Random Forest is selected as
the production deployment model.
""")

# -------------------------------------------------
# FEATURE IMPORTANCE
# -------------------------------------------------
st.markdown("---")
st.markdown("## 🔎 Feature Importance (1-Day Model)")

model = joblib.load("models/rf_h1.joblib" if False else "models/rf_h1.joblib")
# If your file name is rf_h1.joblib adjust accordingly:
# Example:
# model = joblib.load("models/rf_h1.joblib")

model = joblib.load("models/rf_h1.joblib") if False else joblib.load("models/rf_h1.joblib")

# If your actual file name is:
# models/rf_h1.joblib OR models/rf_model_h1.joblib
# Replace above accordingly.

model = joblib.load("models/rf_h1.joblib") if False else joblib.load("models/rf_h1.joblib")
importances = model.feature_importances_
features = model.feature_names_in_

importance_df = pd.DataFrame({
    "Feature": features,
    "Importance": importances
}).sort_values("Importance", ascending=False)

fig2 = go.Figure(go.Bar(
    x=importance_df["Importance"],
    y=importance_df["Feature"],
    orientation='h'
))

fig2.update_layout(height=400)
st.plotly_chart(fig2, use_container_width=True)

# -------------------------------------------------
# EXECUTIVE SUMMARY
# -------------------------------------------------
st.markdown("---")
st.markdown("## 📌 Executive Summary")

st.info(f"""
• {results["1_day"]["date"]} → AQI: {results["1_day"]["value"]}  
• {results["2_day"]["date"]} → AQI: {results["2_day"]["value"]}  
• {results["3_day"]["date"]} → AQI: {results["3_day"]["value"]}  

Random Forest selected as production model
due to lowest RMSE across all horizons.
""")
# -------------------------------------------------
# MODEL REGISTRY DASHBOARD
# -------------------------------------------------

from app.db.mongo import get_model_registry
from app.pipelines.model_rollback import rollback_model

st.markdown("---")
st.markdown("## 🗂 Model Registry & Monitoring")

registry = get_model_registry()

models = list(registry.find({}, {"_id": 0}))

if models:

    df_models = pd.DataFrame(models)
    df_models = df_models.sort_values("created_at", ascending=False)

    st.dataframe(df_models, use_container_width=True)

    # -------------------------
    # RMSE Trend Chart
    # -------------------------
    st.markdown("### 📉 RMSE Trend Over Time")

    if "created_at" in df_models.columns:
        fig = go.Figure()

        for horizon in df_models["horizon"].unique():
            horizon_df = df_models[df_models["horizon"] == horizon]

            fig.add_trace(go.Scatter(
                x=horizon_df["created_at"],
                y=horizon_df["rmse"],
                mode='lines+markers',
                name=f"H{horizon}"
            ))

        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

    # -------------------------
    # Production Model Highlight
    # -------------------------
    st.markdown("### 🏆 Current Production Models")

    production_models = df_models[df_models["is_best"] == True]

    if not production_models.empty:
        st.success("Active Production Models:")

        for _, row in production_models.iterrows():
            st.write(
                f"Horizon: {row['horizon']} | "
                f"Model: {row['model_name']} | "
                f"RMSE: {round(row['rmse'], 4)} | "
                f"Run ID: {row['run_id']}"
            )

    # -------------------------
    # Rollback Section (Optional Advanced)
    # -------------------------
    st.markdown("### 🔁 Manual Rollback")

    selected_horizon = st.selectbox(
        "Select Horizon",
        sorted(df_models["horizon"].unique())
    )

    selected_run_id = st.selectbox(
        "Select Run ID to Rollback",
        df_models[df_models["horizon"] == selected_horizon]["run_id"]
    )

    if st.button("Rollback Model"):
        rollback_model(selected_horizon, selected_run_id)
        st.success("Rollback completed. Refresh page.")

else:
    st.warning("No models found in registry.")
