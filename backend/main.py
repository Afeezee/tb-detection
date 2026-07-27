"""FastAPI entrypoint for the TB detection service.

Routes
------
POST /predict?model=densenet121|hybrid  — classify an uploaded chest X-ray
GET  /history?limit=50                  — recent predictions from Neon Postgres
GET  /metrics                           — hardcoded benchmark table (both models,
                                          internal + external TBX11K)
GET  /health                            — Railway health check

The service imports the training-time modules from src/ so preprocessing,
architectures, and Grad-CAM stay bit-for-bit identical to what was validated
during training.
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
from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware

# .env at the project root (one directory up from backend/)
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


def _parse_cors_origins() -> list[str]:
    """Read allowed origins from FRONTEND_URL / CORS_ORIGINS (comma-separated).

    Falls back to '*' for local development. Railway sets FRONTEND_URL to the
    deployed frontend domain after the first deploy.
    """
    raw = os.getenv("CORS_ORIGINS") or os.getenv("FRONTEND_URL") or ""
    origins = [o.strip() for o in raw.split(",") if o.strip()]
    if not origins:
        return ["*"]
    # Always allow localhost dev regardless of prod list.
    for local in ("http://localhost:3000", "http://127.0.0.1:3000"):
        if local not in origins:
            origins.append(local)
    return origins


@asynccontextmanager
async def lifespan(app: FastAPI):
    service = InferenceService(models_dir=MODEL_DIR)
    available = service.registry.available()
    if not available:
        logger.warning(
            "No checkpoints found in %s. /predict will return 503 until you add "
            "densenet121_best.pt or hybrid_best.pt.",
            MODEL_DIR,
        )
    else:
        # Warm the default so the first request is not slow.
        try:
            service.registry.get(DEFAULT_MODEL if DEFAULT_MODEL in available else available[0])
        except Exception as exc:  # pragma: no cover - warm-up is best-effort
            logger.exception("Could not warm default model: %s", exc)
    app.state.inference = service
    yield


app = FastAPI(
    title="TB Detection API",
    description="FastAPI wrapper around the DenseNet121 baseline and the Hybrid CNN+ViT novelty model.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=_parse_cors_origins(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse, tags=["system"])
def health():
    service: InferenceService = app.state.inference
    return HealthResponse(
        status="ok",
        default_model=DEFAULT_MODEL if DEFAULT_MODEL in SUPPORTED_MODELS else "densenet121",
        loaded_models=list(service.registry._cache.keys()),  # noqa: SLF001
        device=str(service.device),
    )


@app.get("/metrics", response_model=BenchmarkResponse, tags=["system"])
def metrics():
    return get_benchmarks()


@app.post("/predict", response_model=PredictionResponse, tags=["inference"])
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
                gradcam_path=None,  # base64 lives in the response; we do not write to disk here
                clinician_notes=clinician_notes or None,
            )
        except RuntimeError as exc:
            logger.warning("Skipping Neon persistence: %s", exc)
        except Exception as exc:  # pragma: no cover - db failures should not break inference
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


@app.get("/history", response_model=HistoryResponse, tags=["records"])
def history(limit: int = Query(50, ge=1, le=500)):
    try:
        rows = db.fetch_history(limit=limit)
    except RuntimeError as exc:
        raise HTTPException(503, str(exc)) from exc

    items = [HistoryRow(**dict(row)) for row in rows]
    return HistoryResponse(count=len(items), items=items)
