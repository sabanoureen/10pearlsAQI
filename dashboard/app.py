# -------------------------------
# Best Production Model
# -------------------------------
st.subheader("🏆 Best Production Model (1h horizon)")

best_model = safe_get("/models/best", params={"horizon": 1})

if best_model:
    c1, c2, c3 = st.columns(3)
    c1.metric("Model", best_model["model_name"])
    c2.metric("RMSE", round(best_model["rmse"], 2))
    c3.metric("R²", round(best_model["r2"], 3))
else:
    st.warning("No production model found")


# -------------------------------
# Current AQI Prediction
# -------------------------------
st.subheader("📈 Current AQI Prediction (1h)")

prediction = safe_get("/predict", params={"horizon": 1})

if prediction:
    st.metric("Predicted AQI", round(prediction["predicted_aqi"], 2))
    st.caption(
        f"Model: **{prediction['model_name']}** | "
        f"Version: `{prediction.get('version','legacy')}`"
    )
else:
    st.warning("Prediction unavailable")


# -------------------------------
# Multi-day Forecast
# -------------------------------
st.subheader("📊 Multi-day AQI Forecast")

horizons = [h * 24 for h in range(1, forecast_days + 1)]

multi = safe_post("/predict/multi", {"horizons": horizons})

if multi:
    df = pd.DataFrame([
        {"Horizon (hrs)": h, "AQI": v}
        for h, v in zip(horizons, multi["predictions"])
    ])

    fig = px.line(df, x="Horizon (hrs)", y="AQI", markers=True)
    st.plotly_chart(fig, use_container_width=True)
else:
    st.warning("Multi-day forecast unavailable")
