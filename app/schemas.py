"""Request and response schemas for AI detector service (architecture.md section 4)."""
from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field


VerdictType = Literal["likely_ai", "likely_real", "inconclusive", "ai_generated_verified"]
ConfidenceType = Literal["high", "medium", "low"]


class C2PASignal(BaseModel):
    present: bool = False
    ai_declared: Optional[bool] = None


class ModelSignal(BaseModel):
    model: str
    probability: float = Field(ge=0.0, le=1.0)


class SignalsResponse(BaseModel):
    c2pa: C2PASignal
    classifier: ModelSignal
    clip_probe: ModelSignal


class DetectionResponse(BaseModel):
    request_id: str
    verdict: VerdictType
    ai_probability: float = Field(ge=0.0, le=1.0)
    confidence: ConfidenceType
    signals: SignalsResponse
    warnings: List[str] = Field(default_factory=list)
    model_version: str
    benchmark_id: Optional[str] = None
    latency_ms: float


class ErrorDetail(BaseModel):
    code: str
    message: str


class ErrorResponse(BaseModel):
    error: ErrorDetail


class VersionResponse(BaseModel):
    model_version: str
    benchmark_id: Optional[str] = None
    thresholds: Dict[str, float]
    fusion: Dict[str, Any]
    models: Dict[str, str]
