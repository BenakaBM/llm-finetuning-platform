from __future__ import annotations

import uuid
from pathlib import Path

from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.core.errors import NotFoundError, ResourceError
from app.ml.training.hardware import check_resource_feasibility, detect_hardware, estimate_memory_requirement_gb
from app.models.dataset import DatasetVersion
from app.models.enums import JobStatus
from app.models.training import TrainingConfig, TrainingJob
from app.models.user import User
from app.schemas.training import TrainingJobCreateRequest

settings = get_settings()


def create_training_job(db: Session, user: User, payload: TrainingJobCreateRequest) -> TrainingJob:
    dataset_version = db.get(DatasetVersion, payload.dataset_version_id)
    if not dataset_version:
        raise NotFoundError("Dataset version not found")

    hardware = detect_hardware()

    # Rough parameter-count estimate for resource validation purposes only;
    # actual per-model values are looked up from the model registry when available.
    estimated_params_b = 1.1 if "tiny" not in payload.base_model_id.lower() else 0.01
    required_gb = estimate_memory_requirement_gb(
        estimated_params_b,
        use_4bit=payload.config.use_4bit,
        batch_size=payload.config.batch_size,
        max_seq_length=payload.config.max_seq_length,
    )
    feasibility = check_resource_feasibility(required_gb, hardware)
    if not feasibility["feasible"]:
        raise ResourceError(feasibility["warning"])

    config = TrainingConfig(**payload.config.model_dump())
    db.add(config)
    db.flush()

    job_id = uuid.uuid4()
    output_dir = str(Path(settings.STORAGE_PATH, "checkpoints", str(job_id)))

    job = TrainingJob(
        id=job_id,
        owner_id=user.id,
        dataset_version_id=dataset_version.id,
        base_model_id=payload.base_model_id,
        config_id=config.id,
        status=JobStatus.PENDING,
        output_dir=output_dir,
        hardware_info=hardware,
    )
    db.add(job)
    db.commit()
    db.refresh(job)
    return job


def cancel_training_job(db: Session, job: TrainingJob) -> TrainingJob:
    if job.status in (JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED):
        return job
    job.status = JobStatus.CANCELLED
    db.commit()
    db.refresh(job)
    return job


def delete_training_job(db: Session, job: TrainingJob) -> None:
    """Permanently deletes a training job row (and its metrics, via
    cascade). If the job is still active, attempts to forcibly terminate
    the underlying Celery task first - this matters for jobs stuck in
    RUNNING because the worker process was killed by the OS (e.g. an
    out-of-memory kill) and never got to update its own status."""
    still_active = job.status in (JobStatus.PENDING, JobStatus.PREPARING, JobStatus.RUNNING)
    if still_active and job.celery_task_id:
        try:
            from app.workers.celery_app import celery_app

            celery_app.control.revoke(job.celery_task_id, terminate=True, signal="SIGKILL")
        except Exception:  # noqa: BLE001
            # Best-effort: if the broker is unreachable or the worker is
            # already gone, we still want the DB row (and stuck-job UX)
            # cleaned up rather than blocking the delete.
            pass

    # Detach (don't cascade-delete) any Experiment created from this job,
    # so evaluation history isn't silently destroyed just because the
    # originating job record is being removed.
    from app.models.evaluation import Experiment

    experiment = db.query(Experiment).filter(Experiment.training_job_id == job.id).first()
    if experiment:
        experiment.training_job_id = None
        db.add(experiment)

    # Best-effort cleanup of on-disk checkpoint artifacts for jobs that
    # never completed (nothing useful was ever saved to reuse).
    if still_active and job.output_dir:
        import shutil

        shutil.rmtree(job.output_dir, ignore_errors=True)

    db.delete(job)
    db.commit()