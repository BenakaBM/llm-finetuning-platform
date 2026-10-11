import uuid
from datetime import datetime

from sqlalchemy import DateTime, ForeignKey, Integer, JSON, String
from sqlalchemy.orm import Mapped, mapped_column, relationship
from app.db.types import GUID

from app.db.session import Base


class Model(Base):
    """A logical model entry, e.g. 'TinyLlama-1.1B-Chat' base model."""

    __tablename__ = "models"

    id: Mapped[uuid.UUID] = mapped_column(GUID(), primary_key=True, default=uuid.uuid4)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    base_model_id: Mapped[str] = mapped_column(String(255), nullable=False)  # HF hub id
    architecture: Mapped[str | None] = mapped_column(String(100), nullable=True)
    parameter_count: Mapped[str | None] = mapped_column(String(50), nullable=True)
    context_length: Mapped[int | None] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=datetime.utcnow)

    versions = relationship("ModelVersion", back_populates="model", cascade="all, delete-orphan")


class ModelVersion(Base):
    """A concrete artifact of a model: the base weights, or a fine-tuned checkpoint."""

    __tablename__ = "model_versions"

    id: Mapped[uuid.UUID] = mapped_column(GUID(), primary_key=True, default=uuid.uuid4)
    model_id: Mapped[uuid.UUID] = mapped_column(GUID(), ForeignKey("models.id"), nullable=False, index=True)
    version_label: Mapped[str] = mapped_column(String(100), nullable=False)
    training_method: Mapped[str | None] = mapped_column(String(20), nullable=True)  # LORA / QLORA / None
    quantization: Mapped[str | None] = mapped_column(String(50), nullable=True)
    training_job_id: Mapped[uuid.UUID | None] = mapped_column(
        GUID(), ForeignKey("training_jobs.id"), nullable=True
    )
    artifact_path: Mapped[str | None] = mapped_column(String(500), nullable=True)
    evaluation_scores: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=datetime.utcnow)

    model = relationship("Model", back_populates="versions")
