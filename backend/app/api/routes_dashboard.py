from fastapi import APIRouter, Depends
from sqlalchemy import func
from sqlalchemy.orm import Session

from app.api.deps import get_current_user
from app.db.session import get_db
from app.models.dataset import Dataset
from app.models.evaluation import Evaluation, Experiment
from app.models.enums import EvaluationType, JobStatus
from app.models.model import Model
from app.models.training import TrainingJob
from app.models.user import User

router = APIRouter(prefix="/api/dashboard", tags=["dashboard"])


@router.get("/summary")
def dashboard_summary(db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    dataset_count = db.query(func.count(Dataset.id)).filter(Dataset.owner_id == user.id).scalar()
    model_count = db.query(func.count(Model.id)).scalar()
    active_jobs = (
        db.query(func.count(TrainingJob.id))
        .filter(TrainingJob.owner_id == user.id, TrainingJob.status.in_([JobStatus.RUNNING, JobStatus.PREPARING, JobStatus.PENDING]))
        .scalar()
    )
    experiment_count = db.query(func.count(Experiment.id)).filter(Experiment.owner_id == user.id).scalar()

    recent_experiments = (
        db.query(Experiment)
        .filter(Experiment.owner_id == user.id)
        .order_by(Experiment.created_at.desc())
        .limit(5)
        .all()
    )
    recent_jobs = (
        db.query(TrainingJob)
        .filter(TrainingJob.owner_id == user.id)
        .order_by(TrainingJob.created_at.desc())
        .limit(5)
        .all()
    )

    best_eval = (
        db.query(Evaluation)
        .join(Experiment, Evaluation.experiment_id == Experiment.id)
        .filter(Experiment.owner_id == user.id, Evaluation.evaluation_type == EvaluationType.FINE_TUNED)
        .order_by(Evaluation.created_at.desc())
        .first()
    )

    trend = [
        {"created_at": e.created_at, "scores": e.aggregate_scores}
        for e in (
            db.query(Evaluation)
            .join(Experiment, Evaluation.experiment_id == Experiment.id)
            .filter(Experiment.owner_id == user.id, Evaluation.evaluation_type == EvaluationType.FINE_TUNED)
            .order_by(Evaluation.created_at.asc())
            .all()
        )
    ]

    return {
        "counts": {
            "datasets": dataset_count,
            "models": model_count,
            "active_jobs": active_jobs,
            "experiments": experiment_count,
        },
        "recent_experiments": [
            {"id": e.id, "name": e.name, "status": e.status, "created_at": e.created_at} for e in recent_experiments
        ],
        "recent_jobs": [
            {"id": j.id, "status": j.status, "base_model_id": j.base_model_id, "created_at": j.created_at}
            for j in recent_jobs
        ],
        "best_evaluation": best_eval.aggregate_scores if best_eval else None,
        "evaluation_trend": trend,
    }
