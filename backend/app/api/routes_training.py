import asyncio
import json

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from app.api.deps import get_current_user
from app.core.errors import NotFoundError
from app.db.session import get_db
from app.ml.training.hardware import detect_hardware
from app.models.training import TrainingJob, TrainingMetric
from app.models.user import User
from app.schemas.training import TrainingJobCreateRequest, TrainingJobResponse, TrainingMetricResponse
from app.services.training_service import cancel_training_job, create_training_job, delete_training_job

router = APIRouter(prefix="/api/training", tags=["training"])


@router.post("/jobs", response_model=TrainingJobResponse, status_code=201)
def start_training_job(
    payload: TrainingJobCreateRequest,
    max_train_samples: int | None = None,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    job = create_training_job(db, user, payload)

    from app.workers.tasks import run_training_job_task

    async_result = run_training_job_task.delay(str(job.id), max_train_samples or payload.max_train_samples)
    job.celery_task_id = async_result.id
    db.commit()
    db.refresh(job)
    return job


@router.get("/jobs", response_model=list[TrainingJobResponse])
def list_training_jobs(db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    return (
        db.query(TrainingJob)
        .filter(TrainingJob.owner_id == user.id)
        .order_by(TrainingJob.created_at.desc())
        .all()
    )


@router.get("/jobs/{job_id}", response_model=TrainingJobResponse)
def get_training_job(job_id: str, db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    job = db.get(TrainingJob, job_id)
    if not job or job.owner_id != user.id:
        raise NotFoundError("Training job not found")
    return job


@router.get("/jobs/{job_id}/metrics", response_model=list[TrainingMetricResponse])
def get_training_metrics(job_id: str, db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    job = db.get(TrainingJob, job_id)
    if not job or job.owner_id != user.id:
        raise NotFoundError("Training job not found")
    return (
        db.query(TrainingMetric)
        .filter(TrainingMetric.job_id == job.id)
        .order_by(TrainingMetric.step.asc())
        .all()
    )


@router.get("/jobs/{job_id}/stream")
async def stream_training_progress(job_id: str, db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    """Server-Sent Events stream of live training status/metrics, polled from
    the database (which the worker updates in near-real-time as it trains)."""
    job = db.get(TrainingJob, job_id)
    if not job or job.owner_id != user.id:
        raise NotFoundError("Training job not found")

    async def event_generator():
        last_step = -1
        while True:
            db.expire_all()
            current = db.get(TrainingJob, job_id)
            if current is None:
                break
            if current.current_step != last_step:
                last_step = current.current_step
                latest_metric = (
                    db.query(TrainingMetric)
                    .filter(TrainingMetric.job_id == current.id)
                    .order_by(TrainingMetric.step.desc())
                    .first()
                )
                payload = {
                    "status": current.status.value,
                    "current_step": current.current_step,
                    "current_epoch": current.current_epoch,
                    "train_loss": latest_metric.train_loss if latest_metric else None,
                    "eval_loss": latest_metric.eval_loss if latest_metric else None,
                }
                yield f"data: {json.dumps(payload)}\n\n"
            if current.status.value in ("COMPLETED", "FAILED", "CANCELLED"):
                break
            await asyncio.sleep(2)

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@router.post("/jobs/{job_id}/cancel", response_model=TrainingJobResponse)
def cancel_job(job_id: str, db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    job = db.get(TrainingJob, job_id)
    if not job or job.owner_id != user.id:
        raise NotFoundError("Training job not found")
    return cancel_training_job(db, job)


@router.delete("/jobs/{job_id}", status_code=204)
def delete_job(job_id: str, db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    """Permanently removes a training job. If it's still running (or stuck
    - e.g. the worker process was killed by the OS due to an out-of-memory
    condition and never got to update its status), this also attempts to
    forcibly terminate the underlying Celery task, unlike /cancel which
    only asks a still-alive worker to stop cooperatively."""
    job = db.get(TrainingJob, job_id)
    if not job or job.owner_id != user.id:
        raise NotFoundError("Training job not found")
    delete_training_job(db, job)