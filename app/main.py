from fastapi import FastAPI

app = FastAPI(title="AQI Prediction API")


@app.get("/health")
def health():
    try:
        get_model_registry().find_one()
        return {
            "status": "ok",
            "time": datetime.utcnow().isoformat()
        }
    except PyMongoError as e:
        raise HTTPException(status_code=500, detail=str(e))
