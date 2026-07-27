from datetime import datetime
from typing import Literal, Optional

from pydantic import BaseModel, Field


ModelName = Literal["densenet121", "hybrid"]


class PredictionResponse(BaseModel):
    id: Optional[int] = Field(None, description="Row id in the tb_predictions table, if persisted")
    label: Literal["Normal", "TB-positive"]
    label_index: Literal[0, 1]
    confidence: float = Field(..., ge=0.0, le=1.0)
    probabilities: dict[str, float]
    model_used: ModelName
    gradcam_base64: str = Field(..., description="PNG-encoded Grad-CAM overlay, base64 string")
    image_filename: str
    patient_ref: Optional[str] = None
    created_at: datetime


class HistoryRow(BaseModel):
    id: int
    patient_ref: Optional[str]
    image_filename: str
    prediction: str
    confidence: float
    model_name: str
    gradcam_path: Optional[str]
    clinician_notes: Optional[str]
    created_at: datetime


class HistoryResponse(BaseModel):
    count: int
    items: list[HistoryRow]


class ModelBenchmark(BaseModel):
    model: str
    training_regime: Literal["single-source", "multi-source"]
    test_set: Literal["internal", "external_tbx11k"]
    sensitivity: float
    specificity: float
    f1: float
    auc_roc: float
    notes: Optional[str] = None


class BenchmarkResponse(BaseModel):
    rows: list[ModelBenchmark]
    generated_at: datetime


class HealthResponse(BaseModel):
    status: Literal["ok"]
    default_model: ModelName
    loaded_models: list[ModelName]
    device: str
