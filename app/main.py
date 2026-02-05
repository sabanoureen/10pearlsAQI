from fastapi import FastAPI

app = FastAPI(title="AQI Prediction API")


@app.get("/health")
def health():
    return {"status": "ok"}
