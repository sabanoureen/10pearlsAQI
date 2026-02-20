🌍 Karachi AQI Forecast System

AI-Powered Multi-Horizon Air Quality Prediction
Production-Grade MLOps System for Multi-Horizon Time Series Forecasting

🔗 Live Streamlit App:
https://10pearlsaqi-4gpbzmccfuqust2wshwqt.streamlit.app
https://10pearlsaqi-4gpbzmccfuqust2wshwqt.streamlit.app/

🔗 Production Backend API (Railway):
https://web-production-382ce.up.railway.app

🚀 Project Overview

This project is a production-grade, multi-horizon Air Quality Index (AQI) forecasting system built using:

FastAPI (Backend API)

Streamlit (Frontend Dashboard)

MongoDB Atlas (Feature Store + Model Registry)

Railway (Dockerized backend deployment)

Streamlit Cloud (Frontend deployment)

Scikit-learn (Model training)

Random Forest (Production model)

The system predicts AQI for:

✅ 24 hours ahead

✅ 48 hours ahead

✅ 72 hours ahead

🧠 Problem Statement

Karachi suffers from fluctuating air quality levels. Accurate short-term forecasting enables:

Public health advisories

Government response planning

Environmental monitoring

Preventive risk mitigation

This project implements a multi-horizon forecasting pipeline to predict AQI up to 3 days ahead using machine learning.

🏗️ System Architecture
MongoDB Atlas
   │
   ├── Historical Data
   ├── Feature Store
   └── Model Registry
          │
          ▼
Training Pipeline (Scikit-learn)
          │
          ▼
Production Model (Random Forest)
          │
          ▼
FastAPI (Dockerized)
          │
          ▼
Railway Deployment
          │
          ▼
Streamlit Frontend

🔬 Methodology
1️⃣ Data Engineering

Historical hourly AQI dataset

Feature engineering:

Lag features

Rolling averages

Time-based features

Stored in MongoDB Atlas

2️⃣ Multi-Horizon Modeling Strategy

Instead of recursive forecasting, this system uses:

✔ Separate model per horizon
✔ Horizon-specific training
✔ Independent optimization

Models trained:

Random Forest

Gradient Boosting

Ridge Regression

Each model trained separately for:

H1 → 24h

H2 → 48h

H3 → 72h

3️⃣ Model Selection

Evaluation metric:

RMSE (Root Mean Squared Error)

Results:

| Horizon | Random Forest | Gradient Boosting | Ridge |
| ------- | ------------- | ----------------- | ----- |
| 24h     | 3.06          | 9.40              | 12.36 |
| 48h     | 2.67          | 9.35              | 13.36 |
| 72h     | 2.89          | 9.56              | 13.54 |

🏆 Random Forest selected as production model

🧩 Backend (FastAPI)
Endpoints

| Endpoint               | Description           |
| ---------------------- | --------------------- |
| `/`                    | Health check          |
| `/forecast`            | Multi-day forecast    |
| `/models/metrics`      | Model registry        |
| `/models/best`         | Best production model |
| `/features/importance` | Feature importance    |
| `/forecast/shap`       | SHAP explanations     |

🐳 Docker Configuration

Backend runs inside Docker on Railway:

FROM python:3.11-slim

WORKDIR /app

COPY . .

RUN pip install --upgrade pip
RUN pip install -r requirements_api.txt

CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT}"]

Railway dynamically injects the PORT environment variable.

☁️ Deployment
🚂 Backend → Railway

Dockerized FastAPI

MongoDB Atlas connection via environment variables

Auto-redeploy on GitHub push

🎈 Frontend → Streamlit Cloud

Connects to Railway API

Uses secrets manager for:

MongoDB URI

API URL

📊 Streamlit Dashboard Features

✔ Multi-day AQI gauge charts
✔ Forecast trend visualization
✔ Model benchmark comparison
✔ Executive summary
✔ Live refresh button

🔐 Environment Variables
Railway
MONGODB_URI=your_mongodb_connection_string

Streamlit Secrets
MONGODB_URI="..."
API_URL="https://web-production-382ce.up.railway.app"

🧠 Advanced Features

Model registry system

Automated best-model selection

SHAP explainability

Production-ready multi-model pipeline

Separate training & inference architecture

⚙️ Tech Stack

Python 3.11

FastAPI

Streamlit

MongoDB Atlas

Railway

Docker

Scikit-learn

Plotly

📌 Challenges Solved

Docker port configuration on Railway

MongoDB Atlas connection handling

Multi-horizon forecast logic

Model registry architecture

Environment variable management

Streamlit–Railway communication

Production deployment debugging

🎯 Future Improvements

CI/CD with GitHub Actions

Automated daily retraining

Redis caching

Real-time data ingestion

Alert system (SMS / Email)

Container scaling

Monitoring & logging dashboard

👩‍💻 Author

Saba Noureen
MS Data Science
Machine Learning & AI Systems

⭐ If you like this project

Please star the repository and share!


