from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.api.deps import get_current_user
from app.core.errors import NotFoundError
from app.db.session import get_db
from app.models.dataset import DatasetVersion
from app.models.evaluation import Evaluation, Experiment
from app.models.user import User
from app.schemas.evaluation import EvaluationRequest, EvaluationResponse

router = APIRouter(prefix="/api/evaluations", tags=["evaluations"])


@router.post("", response_model=EvaluationResponse, status_code=201)
def create_evaluation(payload: EvaluationRequest, db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    from app.workers.tasks import run_evaluation_task

    experiment = db.get(Experiment, payload.experiment_id)
    if not experiment:
        raise NotFoundError("Experiment not found")

    # Synchronous kick-off for direct API use; the training pipeline itself
    # triggers evaluation automatically via the async task.
    run_evaluation_task(str(experiment.id), str(payload.model_version_id) if payload.model_version_id else None)
    latest = (
        db.query(Evaluation)
        .filter(Evaluation.experiment_id == experiment.id)
        .order_by(Evaluation.created_at.desc())
        .first()
    )
    return latest


@router.get("", response_model=list[EvaluationResponse])
def list_evaluations(experiment_id: str | None = None, db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    query = db.query(Evaluation)
    if experiment_id:
        query = query.filter(Evaluation.experiment_id == experiment_id)
    return query.order_by(Evaluation.created_at.desc()).all()


@router.get("/{evaluation_id}", response_model=EvaluationResponse)
def get_evaluation(evaluation_id: str, db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    evaluation = db.get(Evaluation, evaluation_id)
    if not evaluation:
        raise NotFoundError("Evaluation not found")
    return evaluation
