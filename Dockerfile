FROM python:3.11-slim

WORKDIR /app

COPY requirements_backend.txt .
RUN pip install -r requirements_backend.txt

COPY app /app/app
COPY scripts /app/scripts
COPY models /app/models   # 👈 ADD THIS

CMD ["uvicorn", "app.api.main:app", "--host", "0.0.0.0", "--port", "8000"]

