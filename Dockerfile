FROM python:3.11-slim

WORKDIR /app

COPY requirements_backend.txt .
RUN pip install --no-cache-dir -r requirements_backend.txt

COPY app ./app
COPY models ./models
COPY model_registry.json .

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

