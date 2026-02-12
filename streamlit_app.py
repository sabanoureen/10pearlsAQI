# -----------------------------------------
# LOAD FORECAST
# -----------------------------------------
if st.button("📊 Load Forecast"):

    response = requests.get(f"{API_URL}/forecast/latest?horizon={horizon}")

    if response.status_code != 200:
        st.error("Failed to fetch forecast.")
        st.stop()

    results = response.json()

    if results.get("status") != "success":
        st.error("No forecast available.")
        st.stop()

    df = pd.DataFrame(results["predictions"])

    if df.empty:
        st.error("Forecast data empty.")
        st.stop()

    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values("datetime").reset_index(drop=True)

    latest_aqi = df["predicted_aqi"].iloc[-1]
    max_aqi = df["predicted_aqi"].max()
    avg_aqi = df["predicted_aqi"].mean()

    status, color = get_aqi_status(latest_aqi)

    # -----------------------------------------
    # KPI SECTION
    # -----------------------------------------
    col1, col2, col3 = st.columns(3)

    col1.metric("Latest AQI", round(latest_aqi, 2))
    col2.metric("Max AQI", round(max_aqi, 2))
    col3.metric("Average AQI", round(avg_aqi, 2))

    st.markdown(
        f"<h3 style='color:{color}'>Air Quality Status: {status}</h3>",
        unsafe_allow_html=True
    )

    # -----------------------------------------
    # MODEL INFORMATION
    # -----------------------------------------
    st.subheader("🤖 Model Information")

    colA, colB = st.columns(2)

    colA.info(f"Generated At: {results.get('generated_at', 'N/A')}")
    colB.info(f"Model Version: {results.get('model_version', 'Unknown')}")

    # -----------------------------------------
    # AQI GAUGE
    # -----------------------------------------
    st.subheader("🌡 AQI Gauge")

    gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=latest_aqi,
        title={'text': "Forecast AQI"},
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

    st.plotly_chart(gauge, use_container_width=True)
