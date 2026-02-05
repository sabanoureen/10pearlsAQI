import streamlit as st
import requests
import pandas as pd
import plotly.express as px

# -------------------------------
# Config
# -------------------------------
API_BASE_URL = "https://10pearlsaqi-production-848d.up.railway.app"
forecast_days = 5  # ← you were missing this

st.set_page_config(page_title="AQI Dashboard", layout="wide")

# -------------------------------
# Safe API helpers
# -------------------------------
def safe_get(path, params=None):
    try:
        r = requests.get(f"{API_BASE_URL}{path}", params=params, timeout=10)
        if r.status_code == 200:
            return r.json()
    except Exception as e:
        st.error(f"GET {path} failed: {e}")
    return None


def safe_post(path, payload):
    try:
        r = requests.post(
            f"{API_BASE_URL}{path}",
            json=payload,
            timeout=10,
        )
        if r.status_code == 200:
            return r.json()
    except Exception as e:
        st.error(f"POST {path} failed: {e}")
    return None
