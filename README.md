# 🌫️ 10Pearls AQI – Air Quality Index Prediction System

**Live Dashboard:** https://10pearls-aqi-predictor.streamlit.app/  
**API Base URL:** https://10pearlsaqi-production-d27d.up.railway.app/  
**FastAPI Docs:** https://10pearlsaqi-production-d27d.up.railway.app/docs  

---

## 📋 Table of Contents
- Project Overview
- System Architecture
- Key Features
- Technology Stack
- Live Deployment
- API Documentation
- Streamlit Dashboard
- Project Structure
- MLOps Concepts
- Performance Metrics
- Local Setup
- Verification & Testing
- Author

---

## 🎯 Project Overview

**10Pearls AQI** is an end-to-end **Air Quality Index (AQI) prediction and monitoring system** designed for **production-style ML workflows**.

The system predicts AQI for **Karachi, Pakistan** using trained machine learning models and exposes predictions via a **FastAPI backend**, with a **MongoDB-backed feature store** and a **Streamlit monitoring dashboard**.

This project focuses on **real-world MLOps concepts** rather than only model accuracy.

---

## 🏗️ System Architecture

```mermaid
flowchart LR
    A[Weather & AQI Data Sources] --> B[Feature Engineering Pipeline]
    B --> C[Final Feature Table]

    C --> D[ML Models - Ridge Regression]
    D --> E[FastAPI Inference Service]

    E --> F[Single Horizon Prediction]
    E --> G[Multi-Horizon Prediction]

    E --> H[MongoDB Feature Store]
    H --> I[Feature Freshness API]

    I --> J[Streamlit Monitoring Dashboard]

  ✨ Key Features

✅ Multi-horizon AQI prediction (1h, 6h, 24h)

✅ Production FastAPI service

✅ MongoDB Feature Store

✅ Feature freshness monitoring

✅ Streamlit dashboard for system health

✅ Horizon-specific ML models

✅ Deployed on Railway

✅ Clean modular ML pipelines

🛠️ Technology Stack
Backend

Python

FastAPI

Uvicorn

Pydantic

Machine Learning

Scikit-learn

Ridge Regression

Feature filtering per prediction horizon

Data & Storage

MongoDB Atlas

Feature store with timestamps

Model metadata storage

Frontend

Streamlit

HTML-styled health cards

Deployment

Railway (FastAPI backend)

Streamlit Cloud (dashboard)

🌐 Live Deployment
🔹 FastAPI

Base URL:
https://10pearlsaqi-production-d27d.up.railway.app

Swagger Docs:
/docs

🔹 Streamlit Dashboard

URL:
https://10pearls-aqi-predictor.streamlit.app/

The dashboard acts as a monitoring layer, not just visualization.

📡 API Documentation
1️⃣ Single Horizon Prediction

GET /predict?horizon=24

Response

{
  "status": "ok",
  "city": "Karachi",
  "horizon_hours": 24,
  "predicted_aqi": 162.4,
  "model": "ridge_regression",
  "timestamp": "2026-01-29T18:42:10Z"
}
2️⃣ Multi-Horizon Prediction

GET /predict/multi?horizons=1&horizons=6&horizons=24

Response

{
  "status": "success",
  "city": "Karachi",
  "predictions": {
    "1h": 98.1,
    "6h": 132.4,
    "24h": 168.7
  },
  "model": "ridge_regression",
  "rmse": 11.26,
  "r2": 0.736,
  "timestamp": "2026-01-29T18:44:02Z"
}
3️⃣ Feature Freshness Monitoring

GET /features/freshness

Response

{
  "status": "ok",
  "city": "Karachi",
  "updated_at": "2026-01-29T12:38:51Z",
  "age_minutes": 52.36
}
📊 Streamlit Dashboard

The Streamlit dashboard provides:

🧪 Feature store freshness status

🕒 Last feature update timestamp

🚦 Live / Delayed / Stale indicators

🩺 System health monitoring

This mirrors real-world ML monitoring practices used in production systems.

📁 Project Structure
10pearlsAQI/
│
├── api/
│   └── main.py               # FastAPI application
│
├── pipelines/
│   ├── inference.py          # Prediction logic
│   ├── final_feature_table.py
│   ├── horizon_feature_filter.py
│
├── db/
│   └── mongo.py              # MongoDB feature store
│
├── dashboard/
│   └── app.py                # Streamlit dashboard
│
├── models/
│   └── ridge_h*/             # Horizon-specific models
│
├── Dockerfile
├── requirements.txt
└── README.md
⚙️ MLOps Concepts Demonstrated

Feature Store

Feature Freshness Monitoring

Model Registry

Multi-Horizon Forecasting

Production Inference APIs

Monitoring & Observability

📈 Performance Metrics

Model: Ridge Regression

RMSE: 11.26

R²: 0.736

Prediction Horizons: 1h, 6h, 24h

🧪 Local Setup
git clone <your-repo-url>
cd 10pearlsAQI
pip install -r requirements.txt
Run FastAPI
uvicorn api.main:app --reload
Run Streamlit Dashboard
cd dashboard
streamlit run app.py
✅ Verification & Testing
curl https://10pearlsaqi-production-d27d.up.railway.app/predict?horizon=24
curl https://10pearlsaqi-production-d27d.up.railway.app/predict/multi?horizons=1&horizons=6&horizons=24
curl https://10pearlsaqi-production-d27d.up.railway.app/features/freshness
👤 Author

Saba Noureen
Data Science & Machine Learning