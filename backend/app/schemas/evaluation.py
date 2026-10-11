import uuid
from datetime import datetime

from pydantic import BaseModel

from app.models.enums import EvaluationType


class EvaluationRequest(BaseModel):
    experiment_id: uuid.UUID
    evaluation_type: EvaluationType
    model_version_id: uuid.UUID | None = None
    dataset_version_id: uuid.UUID
    max_samples: int | None = 20
    metrics: list[str] = ["rouge", "bleu", "exact_match"]


class EvaluationResponse(BaseModel):
    id: uuid.UUID
    experiment_id: uuid.UUID
    evaluation_type: EvaluationType
    status: str
    aggregate_scores: dict
    sample_predictions: list
    latency_ms_avg: float | None
    created_at: datetime
    completed_at: datetime | None

    class Config:
        from_attributes = True


class ExperimentResponse(BaseModel):
    id: uuid.UUID
    name: str
    base_model_id: str
    fine_tuning_method: str | None
    hyperparameters: dict
    hardware: dict
    duration_seconds: float | None
    status: str
    report_path: str | None
    created_at: datetime

    class Config:
        from_attributes = True


class PlaygroundRequest(BaseModel):
    model_version_id: uuid.UUID | None = None
    base_model_id: str | None = None
    prompt: str
    temperature: float = 0.7
    top_p: float = 0.9
    max_tokens: int = 128


class PlaygroundResponse(BaseModel):
    response: str
    input_tokens: int
    output_tokens: int
    latency_ms: float
