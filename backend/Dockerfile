# Python 3.11 slim base — matches the training environment.
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# System deps: opencv-python-headless needs libgl/libglib; psycopg2-binary is
# pure-Python-wheels but keeps libpq around defensively.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        libpq5 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy backend requirements first for layer-caching.
COPY backend/requirements.txt /app/backend/requirements.txt
RUN pip install --upgrade pip \
    && pip install -r /app/backend/requirements.txt

# Copy the shared training-time code and config, then the backend service.
COPY config.py /app/config.py
COPY src /app/src
COPY models /app/models
COPY backend /app/backend

ENV PYTHONPATH=/app:/app/backend \
    MODEL_DIR=/app/models

EXPOSE 8000

# Railway injects $PORT; default to 8000 for local runs.
CMD ["sh", "-c", "uvicorn main:app --app-dir /app/backend --host 0.0.0.0 --port ${PORT:-8000}"]
