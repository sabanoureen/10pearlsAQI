import streamlit as st
import requests

# =====================================
# CONFIG
# =====================================

API_BASE_URL = "https://10pearlsaqi-production-d27d.up.railway.app"
# For local testing:
# API_BASE_URL = "http://127.0.0.1:8000"

st.set_page_config(
    page_title="AQI Feature Store Monitor",
    layout="wide",
)

# =====================================
# HELPERS
# =====================================

def fetch_feature_freshness():
    try:
        r = requests.get(
            f"{API_BASE_URL}/features/freshness",
            timeout=8
        )
        r.raise_for_status()
        return r.json()
    except Exception as e:
        st.error(f"❌ Feature freshness error: {e}")
        return None


def fetch_best_model(horizon: int = 1):
    try:
        r = requests.get(
            f"{API_BASE_URL}/models/best?horizon={horizon}",
            timeout=8
        )
        r.raise_for_status()
        data = r.json()

        if data.get("status") != "ok":
            return None

        # 🔑 API returns "model"
        return data.get("model")

    except Exception as e:
        st.error(f"❌ Best model fetch error: {e}")
        return None


def freshness_status(age_minutes):
    if age_minutes is None:
        return "Unknown", "#95A5A6"

    if age_minutes <= 30:
        return "Live", "#2ECC71"
    elif age_minutes <= 60:
        return "Delayed", "#F1C40F"
    else:
        return "Stale", "#E74C3C"


# =====================================
# SIDEBAR
# =====================================

st.sidebar.title("⚙️ Configuration")

city = st.sidebar.selectbox("City", ["Karachi"])
st.sidebar.write("📍 Location: 24.8608, 67.0104")

refresh = st.sidebar.button("🔄 Refresh Data")

st.sidebar.markdown("---")
st.sidebar.subheader("API Status")
st.sidebar.success("🟢 API Connected")

st.sidebar.markdown("---")
st.sidebar.subheader("About")
st.sidebar.info(
    "This dashboard monitors feature freshness and "
    "displays the automatically selected production model."
)

# =====================================
# HEADER
# =====================================

st.title("🧪 AQI Feature Store Monitor")
st.caption("Production ML system • Feature freshness & model registry")

# =====================================
# FEATURE STORE STATUS
# =====================================

if refresh:
    st.cache_data.clear()

@st.cache_data(ttl=60)
def cached_feature_freshness():
    return fetch_feature_freshness()

fresh = cached_feature_freshness()

if fresh and fresh.get("status") == "ok":
    label, color = freshness_status(fresh["age_minutes"])

    st.markdown(
        f"""
        <div style="
            padding:24px;
            border-radius:16px;
            background-color:{color};
            color:white;
            text-align:center;
            box-shadow:0 6px 14px rgba(0,0,0,0.18)
        ">
            <h3>Feature Store Status</h3>
            <h1>{label}</h1>
            <p><b>City:</b> {fresh['city']}</p>
            <p><b>Age:</b> {fresh['age_minutes']} minutes</p>
            <p><b>Last Updated:</b><br>{fresh['updated_at']}</p>
        </div>
        """,
        unsafe_allow_html=True
    )
else:
    st.warning("⚠️ Feature freshness unavailable")

# =====================================
# BEST MODEL DISPLAY
# =====================================

st.markdown("---")
st.subheader("🧠 Best Model (Auto-Selected)")

best_model = fetch_best_model(horizon=1)

if best_model:
    st.success(
        f"""
        **Model:** {best_model['model_name']}  
        **RMSE:** {best_model['rmse']:.2f}  
        **R²:** {best_model['r2']:.3f}  
        **Status:** Production
        """
    )

    st.caption(
        f"Predictions generated using **{best_model['model_name']}** "
        f"(RMSE={best_model['rmse']:.2f})"
    )
else:
    st.warning("Best model information not available")
