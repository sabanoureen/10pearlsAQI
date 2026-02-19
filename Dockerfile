FROM python:3.11-slim

WORKDIR /app

COPY . .

RUN pip install --upgrade pip
RUN pip install -r requirements_api.txt

CMD uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}
