from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.api.deps import get_current_user
from app.core.errors import NotFoundError
from app.db.session import get_db
from app.models.evaluation import Evaluation, Experiment
from app.models.enums import EvaluationType
from app.models.training import TrainingMetric
from app.models.user import User
from app.schemas.evaluation import ExperimentResponse

router = APIRouter(prefix="/api/experiments", tags=["experiments"])


@router.get("", response_model=list[ExperimentResponse])
def list_experiments(db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    return (
        db.query(Experiment)
        .filter(Experiment.owner_id == user.id)
        .order_by(Experiment.created_at.desc())
        .all()
    )


@router.get("/{experiment_id}")
def get_experiment_detail(experiment_id: str, db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    experiment = db.get(Experiment, experiment_id)
    if not experiment or experiment.owner_id != user.id:
        raise NotFoundError("Experiment not found")

    evaluations = (
        db.query(Evaluation).filter(Evaluation.experiment_id == experiment.id).all()
    )
    base_eval = next((e for e in evaluations if e.evaluation_type == EvaluationType.BASE), None)
    ft_eval = next((e for e in evaluations if e.evaluation_type == EvaluationType.FINE_TUNED), None)

    comparison = []
    if base_eval and ft_eval:
        from app.services.evaluation_service import compute_comparison

        comparison = compute_comparison(base_eval, ft_eval)

    metrics = []
    if experiment.training_job_id:
        metrics = (
            db.query(TrainingMetric)
            .filter(TrainingMetric.job_id == experiment.training_job_id)
            .order_by(TrainingMetric.step.asc())
            .all()
        )

    return {
        "experiment": {
            "id": experiment.id,
            "name": experiment.name,
            "base_model_id": experiment.base_model_id,
            "fine_tuning_method": experiment.fine_tuning_method,
            "hyperparameters": experiment.hyperparameters,
            "hardware": experiment.hardware,
            "duration_seconds": experiment.duration_seconds,
            "status": experiment.status,
            "report_path": experiment.report_path,
            "created_at": experiment.created_at,
        },
        "training_curves": [
            {
                "step": m.step,
                "epoch": m.epoch,
                "train_loss": m.train_loss,
                "eval_loss": m.eval_loss,
                "learning_rate": m.learning_rate,
            }
            for m in metrics
        ],
        "base_evaluation": base_eval.aggregate_scores if base_eval else None,
        "fine_tuned_evaluation": ft_eval.aggregate_scores if ft_eval else None,
        "comparison": comparison,
        "sample_predictions": ft_eval.sample_predictions if ft_eval else [],
    }


@router.post("/compare")
def compare_experiments(experiment_ids: list[str], db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    if len(experiment_ids) < 2 or len(experiment_ids) > 4:
        from app.core.errors import ValidationAppError

        raise ValidationAppError("Provide between 2 and 4 experiment IDs to compare.")

    rows = []
    for exp_id in experiment_ids:
        experiment = db.get(Experiment, exp_id)
        if not experiment:
            continue
        ft_eval = (
            db.query(Evaluation)
            .filter(Evaluation.experiment_id == experiment.id, Evaluation.evaluation_type == EvaluationType.FINE_TUNED)
            .first()
        )
        rows.append(
            {
                "experiment_id": experiment.id,
                "name": experiment.name,
                "base_model_id": experiment.base_model_id,
                "method": experiment.fine_tuning_method,
                "duration_seconds": experiment.duration_seconds,
                "scores": ft_eval.aggregate_scores if ft_eval else {},
                "latency_ms_avg": ft_eval.latency_ms_avg if ft_eval else None,
            }
        )

    best = None
    if rows:
        def score_key(r):
            scores = r["scores"] or {}
            return sum(v for v in scores.values() if isinstance(v, (int, float)))

        best = max(rows, key=score_key)["experiment_id"]

    return {"rows": rows, "best_experiment_id": best}
