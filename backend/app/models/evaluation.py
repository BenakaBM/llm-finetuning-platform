import uuid
from datetime import datetime

from sqlalchemy import DateTime, Enum, Float, ForeignKey, JSON, String
from sqlalchemy.orm import Mapped, mapped_column, relationship
from app.db.types import GUID

from app.db.session import Base
from app.models.enums import EvaluationType


class Evaluation(Base):
    __tablename__ = "evaluations"

    id: Mapped[uuid.UUID] = mapped_column(GUID(), primary_key=True, default=uuid.uuid4)
    experiment_id: Mapped[uuid.UUID] = mapped_column(GUID(), ForeignKey("experiments.id"), nullable=False, index=True)
    evaluation_type: Mapped[EvaluationType] = mapped_column(Enum(EvaluationType), nullable=False)
    model_version_id: Mapped[uuid.UUID | None] = mapped_column(GUID(), ForeignKey("model_versions.id"), nullable=True)
    dataset_version_id: Mapped[uuid.UUID] = mapped_column(GUID(), ForeignKey("dataset_versions.id"), nullable=False)
    status: Mapped[str] = mapped_column(String(20), default="PENDING")
    aggregate_scores: Mapped[dict] = mapped_column(JSON, default=dict)
    sample_predictions: Mapped[list] = mapped_column(JSON, default=list)
    latency_ms_avg: Mapped[float | None] = mapped_column(Float, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=datetime.utcnow)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    experiment = relationship("Experiment", back_populates="evaluations")
    results = relationship("EvaluationResult", back_populates="evaluation", cascade="all, delete-orphan")


class EvaluationResult(Base):
    """Per-example evaluation result, enabling drill-down / auditing."""

    __tablename__ = "evaluation_results"

    id: Mapped[uuid.UUID] = mapped_column(GUID(), primary_key=True, default=uuid.uuid4)
    evaluation_id: Mapped[uuid.UUID] = mapped_column(GUID(), ForeignKey("evaluations.id"), nullable=False, index=True)
    example_index: Mapped[int] = mapped_column(default=0)
    prompt: Mapped[str] = mapped_column(String(4000), default="")
    reference: Mapped[str] = mapped_column(String(4000), default="")
    prediction: Mapped[str] = mapped_column(String(4000), default="")
    scores: Mapped[dict] = mapped_column(JSON, default=dict)

    evaluation = relationship("Evaluation", back_populates="results")


class Experiment(Base):
    __tablename__ = "experiments"

    id: Mapped[uuid.UUID] = mapped_column(GUID(), primary_key=True, default=uuid.uuid4)
    owner_id: Mapped[uuid.UUID] = mapped_column(GUID(), ForeignKey("users.id"), nullable=False, index=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    training_job_id: Mapped[uuid.UUID | None] = mapped_column(GUID(), ForeignKey("training_jobs.id"), nullable=True, unique=True)
    dataset_version_id: Mapped[uuid.UUID] = mapped_column(GUID(), ForeignKey("dataset_versions.id"), nullable=False)
    base_model_id: Mapped[str] = mapped_column(String(255), nullable=False)
    fine_tuning_method: Mapped[str | None] = mapped_column(String(20), nullable=True)
    hyperparameters: Mapped[dict] = mapped_column(JSON, default=dict)
    hardware: Mapped[dict] = mapped_column(JSON, default=dict)
    duration_seconds: Mapped[float | None] = mapped_column(Float, nullable=True)
    seed: Mapped[int | None] = mapped_column(nullable=True)
    report_path: Mapped[str | None] = mapped_column(String(500), nullable=True)
    status: Mapped[str] = mapped_column(String(20), default="RUNNING")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=datetime.utcnow)

    owner = relationship("User")
    training_job = relationship("TrainingJob", back_populates="experiment")
    evaluations = relationship("Evaluation", back_populates="experiment", cascade="all, delete-orphan")
    artifacts = relationship("Artifact", back_populates="experiment", cascade="all, delete-orphan")


class Artifact(Base):
    __tablename__ = "artifacts"

    id: Mapped[uuid.UUID] = mapped_column(GUID(), primary_key=True, default=uuid.uuid4)
    experiment_id: Mapped[uuid.UUID] = mapped_column(GUID(), ForeignKey("experiments.id"), nullable=False, index=True)
    artifact_type: Mapped[str] = mapped_column(String(50), nullable=False)  # model, checkpoint, report, log, config
    path: Mapped[str] = mapped_column(String(500), nullable=False)
    size_bytes: Mapped[int | None] = mapped_column(nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=datetime.utcnow)

    experiment = relationship("Experiment", back_populates="artifacts")


class AuditLog(Base):
    __tablename__ = "audit_logs"

    id: Mapped[uuid.UUID] = mapped_column(GUID(), primary_key=True, default=uuid.uuid4)
    user_id: Mapped[uuid.UUID | None] = mapped_column(GUID(), ForeignKey("users.id"), nullable=True)
    action: Mapped[str] = mapped_column(String(100), nullable=False)
    detail: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=datetime.utcnow)
