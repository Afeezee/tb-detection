# Single-container monorepo build for TB Detection.
#
# Stage 1 (frontend-build): Node 20 → `next build` with output: "export" →
# produces frontend/out/, a fully static Next.js site.
#
# Stage 2 (runtime): Python 3.11 → FastAPI + torch, copies src/, config.py,
# models/, backend/, and the static frontend from stage 1. Serves the API
# under /api/* and the frontend at / on a single port.

# ---------------------------------------------------------------------------
# Stage 1 — build the Next.js static export.
# ---------------------------------------------------------------------------
FROM node:20-slim AS frontend-build

WORKDIR /frontend

# Copy manifest first so npm install is layer-cached.
COPY frontend/package.json ./
# Lockfile is optional; use npm install to be robust to it being absent.
RUN npm install --no-audit --no-fund

# Copy the rest of the frontend source and build.
COPY frontend/ ./
RUN npm run build

# next.config.js sets output: "export" → static site lives in ./out/

# ---------------------------------------------------------------------------
# Stage 2 — Python runtime.
# ---------------------------------------------------------------------------
FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONPATH=/app:/app/backend \
    MODEL_DIR=/app/models \
    FRONTEND_STATIC_DIR=/app/frontend_static

# opencv-python-headless needs libgl / libglib; libpq5 keeps psycopg2 happy.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        libpq5 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Python deps first for cache efficiency.
COPY backend/requirements.txt /app/backend/requirements.txt
RUN pip install --upgrade pip && pip install -r /app/backend/requirements.txt

# App source.
COPY config.py /app/config.py
COPY src /app/src
COPY models /app/models
COPY backend /app/backend

# Static frontend from stage 1.
COPY --from=frontend-build /frontend/out /app/frontend_static

EXPOSE 8000

# uvicorn under sh -c so ${PORT} expands (Railway injects PORT).
CMD ["sh", "-c", "uvicorn main:app --app-dir /app/backend --host 0.0.0.0 --port ${PORT:-8000}"]
