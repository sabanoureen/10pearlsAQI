from model_registry import register_model

# ---------------------------
# +1h BEST MODEL (ACTIVE)
# ---------------------------
register_model(
    model_name="ridge_regression",
    horizon=1,
    rmse=11.26,
    r2=0.736,
    features=[
        "pm2_5", "pm10", "carbon_monoxide",
        "nitrogen_dioxide", "sulphur_dioxide",
        "ozone", "hour", "day_of_week",
        "hour_sin", "hour_cos"
    ],
    status="active"
)

# ---------------------------
# +24h BEST MODEL (ACTIVE)
# ---------------------------
register_model(
    model_name="ridge_regression",
    horizon=24,
    rmse=12.44,
    r2=0.744,
    features=[
        "pm2_5", "pm10", "carbon_monoxide",
        "nitrogen_dioxide", "sulphur_dioxide",
        "ozone", "hour", "day_of_week",
        "hour_sin", "hour_cos"
    ],
    status="active"
)