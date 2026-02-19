from fastapi import FastAPI, HTTPException
from datetime import datetime, timedelta
from pathlib import Path
import numpy as np
import joblib
import os

from app.db.mongo import get_db, get_model_registry

from fastapi import FastAPI
from app.pipelines.inference_multi import predict_next_3_days

app = FastAPI()

@app.get("/")
def root():
    return {"message": "AQI Backend Running"}

@app.get("/forecast")
def forecast():
    return predict_next_3_days()
