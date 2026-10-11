import uuid
from datetime import datetime

from pydantic import BaseModel, Field

from app.models.enums import JobStatus, TrainingMethod


class TrainingConfigRequest(BaseModel):
    method: TrainingMethod = TrainingMethod.LORA
    learning_rate: float = 2e-4
    batch_size: int = 2
    gradient_accumulation_steps: int = 4
    epochs: int = 1
    max_seq_length: int = 512
    warmup_ratio: float = 0.03
    weight_decay: float = 0.0
    optimizer: str = "adamw_torch"
    scheduler: str = "cosine"
    seed: int = 42

    lora_rank: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_target_modules: list[str] = Field(default_factory=lambda: ["q_proj", "v_proj"])

    use_4bit: bool = False
    quant_type: str = "nf4"
    compute_dtype: str = "float16"
    double_quant: bool = True


class TrainingJobCreateRequest(BaseModel):
    dataset_version_id: uuid.UUID
    base_model_id: str
    experiment_name: str
    config: TrainingConfigRequest
    max_train_samples: int | None = None  # useful for CPU demo runs


class TrainingJobResponse(BaseModel):
    id: uuid.UUID
    status: JobStatus
    base_model_id: str
    current_epoch: float
    current_step: int
    total_steps: int
    output_dir: str | None
    error_message: str | None
    hardware_info: dict
    started_at: datetime | None
    completed_at: datetime | None
    created_at: datetime

    class Config:
        from_attributes = True


class TrainingMetricResponse(BaseModel):
    step: int
    epoch: float
    train_loss: float | None
    eval_loss: float | None
    learning_rate: float | None
    recorded_at: datetime

    class Config:
        from_attributes = True
