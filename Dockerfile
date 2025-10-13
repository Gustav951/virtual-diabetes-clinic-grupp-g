FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Install curl (for healthcheck)
RUN apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY model.joblib .
COPY src/ ./src/

ARG VERSION=dev
ENV MODEL_VERSION=${VERSION}
ENV MODEL_PATH=/app/model.joblib

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=3s --start-period=10s CMD curl -fsS http://localhost:8000/health || exit 1

CMD ["uvicorn", "src.predict_service:app", "--host", "0.0.0.0", "--port", "8000"]
