"""FastAPI entrypoint for the TB detection service.

Single-container deployment:
- API is mounted under /api/*
- The Next.js static export (frontend/out) is mounted at / so this container
  serves both the UI and the API on the same port.

Routes
------
POST /api/predict?model=densenet121|hybrid  — classify an uploaded chest X-ray
GET  /api/history?limit=50                  — recent predictions from Neon
GET  /api/metrics                           — hardcoded benchmark table
GET  /api/health                            — Railway health check
"""
from __future__ import annotations

import logging
import os
import sys
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, Optional

from dotenv import load_dotenv
from fastapi import APIRouter, FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import config  # noqa: E402
from src import db  # noqa: E402

from benchmarks import get_benchmarks  # noqa: E402
from inference import InferenceService, SUPPORTED_MODELS  # noqa: E402
from schemas import (  # noqa: E402
    BenchmarkResponse,
    HealthResponse,
    HistoryResponse,
    HistoryRow,
    PredictionResponse,
)

logger = logging.getLogger("tb_backend")
logging.basicConfig(level=logging.INFO)


DEFAULT_MODEL: str = os.getenv("DEFAULT_MODEL", "densenet121")
MODEL_DIR = Path(os.getenv("MODEL_DIR", str(config.MODELS_DIR))).resolve()
STATIC_DIR = Path(os.getenv("FRONTEND_STATIC_DIR", "/app/frontend_static"))


@asynccontextmanager
async def lifespan(app: FastAPI):
    service = InferenceService(models_dir=MODEL_DIR)
    available = service.registry.available()
    if not available:
        logger.warning(
            "No checkpoints found in %s. /api/predict will return 503 until you add "
            "densenet121_best.pt or hybrid_best.pt.",
            MODEL_DIR,
        )
    else:
        try:
            service.registry.get(DEFAULT_MODEL if DEFAULT_MODEL in available else available[0])
        except Exception as exc:  # pragma: no cover
            logger.exception("Could not warm default model: %s", exc)
    app.state.inference = service
    yield


app = FastAPI(
    title="TB Detection API",
    description="FastAPI wrapper around the DenseNet121 baseline and the Hybrid CNN+ViT novelty model.",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS is still permissive so a locally-run frontend against a remote backend
# works during development. In production the frontend is same-origin.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

api = APIRouter(prefix="/api")


@api.get("/health", response_model=HealthResponse, tags=["system"])
def health():
    service: InferenceService = app.state.inference
    return HealthResponse(
        status="ok",
        default_model=DEFAULT_MODEL if DEFAULT_MODEL in SUPPORTED_MODELS else "densenet121",
        loaded_models=list(service.registry._cache.keys()),  # noqa: SLF001
        device=str(service.device),
    )


@api.get("/metrics", response_model=BenchmarkResponse, tags=["system"])
def metrics():
    return get_benchmarks()


@api.post("/predict", response_model=PredictionResponse, tags=["inference"])
async def predict(
    file: UploadFile = File(..., description="Chest X-ray, PNG or JPEG"),
    model: Literal["densenet121", "hybrid"] = Query(
        DEFAULT_MODEL, description="Model to run inference with"
    ),
    patient_ref: Optional[str] = Form(None),
    clinician_notes: Optional[str] = Form(None),
    persist: bool = Form(True, description="Save result to Neon Postgres"),
):
    if file.content_type not in {"image/png", "image/jpeg", "image/jpg"}:
        raise HTTPException(415, f"Unsupported content type: {file.content_type}")

    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(400, "Empty upload.")

    service: InferenceService = app.state.inference
    try:
        result = service.predict(image_bytes, model)
    except FileNotFoundError as exc:
        raise HTTPException(503, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc

    row_id: Optional[int] = None
    created_at = datetime.now(timezone.utc)

    if persist:
        try:
            row_id = db.insert_prediction(
                image_filename=file.filename or "upload.png",
                prediction=result["label"],
                confidence=result["confidence"],
                model_name=model,
                patient_ref=patient_ref or None,
                gradcam_path=None,
                clinician_notes=clinician_notes or None,
            )
        except RuntimeError as exc:
            logger.warning("Skipping Neon persistence: %s", exc)
        except Exception as exc:  # pragma: no cover
            logger.exception("Failed to persist prediction: %s", exc)

    return PredictionResponse(
        id=row_id,
        label=result["label"],
        label_index=result["label_index"],
        confidence=result["confidence"],
        probabilities=result["probabilities"],
        model_used=model,
        gradcam_base64=result["gradcam_base64"],
        image_filename=file.filename or "upload.png",
        patient_ref=patient_ref,
        created_at=created_at,
    )


@api.get("/history", response_model=HistoryResponse, tags=["records"])
def history(limit: int = Query(50, ge=1, le=500)):
    try:
        rows = db.fetch_history(limit=limit)
    except RuntimeError as exc:
        raise HTTPException(503, str(exc)) from exc

    items = [HistoryRow(**dict(row)) for row in rows]
    return HistoryResponse(count=len(items), items=items)


app.include_router(api)


# ---------------------------------------------------------------------------
# Static frontend — mounted at "/" AFTER the API router so /api/* still resolves.
# ---------------------------------------------------------------------------
if STATIC_DIR.exists():
    # SPA-style: unknown paths fall back to index.html so trailing-slash routes
    # like /predict/ resolve to the static export produced by `next build`.
    app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")
else:
    logger.warning("Static frontend directory %s not found — UI will 404.", STATIC_DIR)

    @app.get("/")
    def _no_static():
        return {"detail": "Frontend not built into this image."}
