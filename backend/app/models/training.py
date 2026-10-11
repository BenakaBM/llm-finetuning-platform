import uuid
from datetime import datetime

from sqlalchemy import DateTime, Enum, Float, ForeignKey, Integer, JSON, String
from sqlalchemy.orm import Mapped, mapped_column, relationship
from app.db.types import GUID

from app.db.session import Base
from app.models.enums import JobStatus, TrainingMethod


class TrainingConfig(Base):
    __tablename__ = "training_configs"

    id: Mapped[uuid.UUID] = mapped_column(GUID(), primary_key=True, default=uuid.uuid4)
    method: Mapped[TrainingMethod] = mapped_column(Enum(TrainingMethod), nullable=False)
    learning_rate: Mapped[float] = mapped_column(Float, default=2e-4)
    batch_size: Mapped[int] = mapped_column(Integer, default=2)
    gradient_accumulation_steps: Mapped[int] = mapped_column(Integer, default=4)
    epochs: Mapped[int] = mapped_column(Integer, default=1)
    max_seq_length: Mapped[int] = mapped_column(Integer, default=512)
    warmup_ratio: Mapped[float] = mapped_column(Float, default=0.03)
    weight_decay: Mapped[float] = mapped_column(Float, default=0.0)
    optimizer: Mapped[str] = mapped_column(String(50), default="adamw_torch")
    scheduler: Mapped[str] = mapped_column(String(50), default="cosine")
    seed: Mapped[int] = mapped_column(Integer, default=42)

    lora_rank: Mapped[int] = mapped_column(Integer, default=8)
    lora_alpha: Mapped[int] = mapped_column(Integer, default=16)
    lora_dropout: Mapped[float] = mapped_column(Float, default=0.05)
    lora_target_modules: Mapped[list] = mapped_column(JSON, default=list)

    use_4bit: Mapped[bool] = mapped_column(default=False)
    quant_type: Mapped[str] = mapped_column(String(20), default="nf4")
    compute_dtype: Mapped[str] = mapped_column(String(20), default="float16")
    double_quant: Mapped[bool] = mapped_column(default=True)

    extra: Mapped[dict] = mapped_column(JSON, default=dict)

    jobs = relationship("TrainingJob", back_populates="config")


class TrainingJob(Base):
    __tablename__ = "training_jobs"

    id: Mapped[uuid.UUID] = mapped_column(GUID(), primary_key=True, default=uuid.uuid4)
    owner_id: Mapped[uuid.UUID] = mapped_column(GUID(), ForeignKey("users.id"), nullable=False, index=True)
    dataset_version_id: Mapped[uuid.UUID] = mapped_column(
        GUID(), ForeignKey("dataset_versions.id"), nullable=False
    )
    base_model_id: Mapped[str] = mapped_column(String(255), nullable=False)
    config_id: Mapped[uuid.UUID] = mapped_column(GUID(), ForeignKey("training_configs.id"), nullable=False)
    status: Mapped[JobStatus] = mapped_column(Enum(JobStatus), default=JobStatus.PENDING, index=True)
    celery_task_id: Mapped[str | None] = mapped_column(String(255), nullable=True)
    current_epoch: Mapped[float] = mapped_column(Float, default=0.0)
    current_step: Mapped[int] = mapped_column(Integer, default=0)
    total_steps: Mapped[int] = mapped_column(Integer, default=0)
    output_dir: Mapped[str | None] = mapped_column(String(500), nullable=True)
    error_message: Mapped[str | None] = mapped_column(String(2000), nullable=True)
    hardware_info: Mapped[dict] = mapped_column(JSON, default=dict)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=datetime.utcnow)

    owner = relationship("User", back_populates="training_jobs")
    config = relationship("TrainingConfig", back_populates="jobs")
    metrics = relationship("TrainingMetric", back_populates="job", cascade="all, delete-orphan")
    experiment = relationship("Experiment", back_populates="training_job", uselist=False)


class TrainingMetric(Base):
    __tablename__ = "training_metrics"

    id: Mapped[uuid.UUID] = mapped_column(GUID(), primary_key=True, default=uuid.uuid4)
    job_id: Mapped[uuid.UUID] = mapped_column(GUID(), ForeignKey("training_jobs.id"), nullable=False, index=True)
    step: Mapped[int] = mapped_column(Integer, nullable=False)
    epoch: Mapped[float] = mapped_column(Float, nullable=False)
    train_loss: Mapped[float | None] = mapped_column(Float, nullable=True)
    eval_loss: Mapped[float | None] = mapped_column(Float, nullable=True)
    learning_rate: Mapped[float | None] = mapped_column(Float, nullable=True)
    recorded_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=datetime.utcnow)

    job = relationship("TrainingJob", back_populates="metrics")