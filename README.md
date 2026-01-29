# 🌫️ 10Pearls AQI Prediction System

A **production-grade Air Quality Index (AQI) prediction platform** built with  
**FastAPI + Machine Learning + MongoDB + Streamlit**, deployed on **Railway**.

This project demonstrates **end-to-end MLOps thinking**:
data ingestion → feature engineering → model inference → feature store → monitoring → dashboard.

---

## 🚀 Live Links

- **API (FastAPI Docs)**  
  👉 https://10pearlsaqi-production-d27d.up.railway.app/docs

- **Feature Freshness Endpoint**  
  👉 https://10pearlsaqi-production-d27d.up.railway.app/features/freshness

---

## 🧠 Key Features

✅ Multi-horizon AQI prediction (1h, 6h, 24h)  
✅ Online **feature store** (MongoDB)  
✅ Feature freshness monitoring API  
✅ Streamlit monitoring dashboard  
✅ Production-ready FastAPI service  
✅ Modular ML pipelines  
✅ Model registry & horizon-specific models  

---

## 🏗️ System Architecture

```mermaid
flowchart LR
    A[External AQI Sources] --> B[Data Ingestion Pipeline]
    B --> C[Feature Engineering]
    C --> D[Final Feature Table]

    D --> E[ML Models (Ridge Regression)]
    E --> F[FastAPI Inference Service]

    F --> G[Single AQI Prediction]
    F --> H[Multi-Horizon Prediction]

    F --> I[MongoDB Feature Store]
    I --> J[Feature Freshness API]

    J --> K[Streamlit Dashboard]

   ## 📡 API Endpoints

### 🔹 Single Horizon Prediction
**GET** `/predict?horizon=24`

**Response**
```json
{
  "status": "ok",
  "city": "Karachi",
  "horizon_hours": 24,
  "predicted_aqi": 162.4,
  "model": "ridge_regression",
  "timestamp": "2026-01-29T18:42:10Z"
}


###🔹 Multi-Horizon Prediction
**GET** /predict/multi?horizons=1&horizons=6&horizons=24
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

🔹 Feature Freshness Monitoring
GET /features/freshness
Response
{
  "status": "ok",
  "city": "Karachi",
  "updated_at": "2026-01-29T12:38:51Z",
  "age_minutes": 52.36
}


---

### 📊 Streamlit Dashboard

```md
## 📊 Streamlit Dashboard

The Streamlit dashboard acts as a **monitoring layer**, not just visualization.

It provides:
- Feature store freshness status (**Live / Delayed / Stale**)
- Last feature update timestamps
- System health indicators

This mirrors real-world ML monitoring practices used in production systems.

## 🗂️ Project Structure


10pearlsAQI/
│
├── api/
│   └── main.py               # FastAPI app
│
├── pipelines/
│   ├── inference.py          # Prediction logic
│   ├── final_feature_table.py
│   ├── horizon_feature_filter.py
│   └── ...
│
├── db/
│   └── mongo.py              # Feature store (MongoDB)
│
├── dashboard/
│   └── app.py                # Streamlit monitoring dashboard
│
├── models/
│   └── ridge_h*/             # Horizon-specific models
│
├── Dockerfile
├── requirements.txt
└── README.md

## 🧪 Tech Stack

**Backend:** FastAPI  
**Machine Learning:** Scikit-learn (Ridge Regression)  
**Database:** MongoDB (Feature Store)  
**Frontend:** Streamlit  
**Deployment:** Railway  

### MLOps Concepts
- Feature Store
- Feature Freshness Monitoring
- Model Registry
- Multi-Horizon Forecasting

## 👤 Author

**Saba Noureen**  
Data Science & Machine Learning