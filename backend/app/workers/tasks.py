from __future__ import annotations

import uuid
from datetime import datetime

from app.core.config import get_settings
from app.core.logging import configure_logging, get_logger
from app.db.session import SessionLocal
from app.ml.training.hardware import detect_hardware
from app.ml.training.trainer import TrainingConfigDTO, run_training
from app.models.dataset import DatasetVersion
from app.models.enums import EvaluationType, JobStatus
from app.models.evaluation import Experiment
from app.models.model import Model, ModelVersion
from app.models.training import TrainingJob, TrainingMetric
from app.services.evaluation_service import compute_comparison, run_evaluation
from app.services.report_service import generate_report
from app.workers.celery_app import celery_app

configure_logging()
logger = get_logger(__name__)
settings = get_settings()


@celery_app.task(name="training.run_job")
def run_training_job_task(job_id: str, max_train_samples: int | None = None) -> None:
    db = SessionLocal()
    try:
        job: TrainingJob | None = db.get(TrainingJob, uuid.UUID(job_id))
        if not job:
            logger.error("Training job %s not found", job_id)
            return

        job.status = JobStatus.PREPARING
        job.started_at = datetime.utcnow()
        job.hardware_info = detect_hardware()
        db.commit()

        dataset_version = db.get(DatasetVersion, job.dataset_version_id)
        config = job.config

        experiment = Experiment(
            owner_id=job.owner_id,
            name=f"exp-{job.id}",
            training_job_id=job.id,
            dataset_version_id=dataset_version.id,
            base_model_id=job.base_model_id,
            fine_tuning_method=config.method.value,
            hyperparameters={
                "learning_rate": config.learning_rate,
                "batch_size": config.batch_size,
                "gradient_accumulation_steps": config.gradient_accumulation_steps,
                "epochs": config.epochs,
                "max_seq_length": config.max_seq_length,
                "lora_rank": config.lora_rank,
                "lora_alpha": config.lora_alpha,
                "lora_dropout": config.lora_dropout,
                "use_4bit": config.use_4bit,
            },
            hardware=job.hardware_info,
            seed=config.seed,
            status="RUNNING",
        )
        db.add(experiment)
        db.commit()
        db.refresh(experiment)

        def on_metric(payload: dict) -> None:
            metric = TrainingMetric(job_id=job.id, **payload)
            db.add(metric)
            job.current_step = payload["step"]
            job.current_epoch = payload["epoch"]
            db.commit()

        def should_stop() -> bool:
            db.refresh(job)
            return job.status == JobStatus.CANCELLED

        job.status = JobStatus.RUNNING
        db.commit()

        config_dto = TrainingConfigDTO(
            method=config.method.value,
            learning_rate=config.learning_rate,
            batch_size=config.batch_size,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            epochs=config.epochs,
            max_seq_length=config.max_seq_length,
            warmup_ratio=config.warmup_ratio,
            weight_decay=config.weight_decay,
            optimizer=config.optimizer,
            scheduler=config.scheduler,
            seed=config.seed,
            lora_rank=config.lora_rank,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            lora_target_modules=config.lora_target_modules or ["q_proj", "v_proj"],
            use_4bit=config.use_4bit,
            quant_type=config.quant_type,
            compute_dtype=config.compute_dtype,
            double_quant=config.double_quant,
        )

        result = run_training(
            base_model_id=job.base_model_id,
            train_path=dataset_version.train_path,
            val_path=dataset_version.validation_path,
            output_dir=job.output_dir,
            config=config_dto,
            cache_dir=settings.MODEL_CACHE,
            max_train_samples=max_train_samples,
            on_metric=on_metric,
            should_stop=should_stop,
        )

        if job.status == JobStatus.CANCELLED:
            experiment.status = "CANCELLED"
            db.commit()
            return

        job.status = JobStatus.COMPLETED
        job.completed_at = datetime.utcnow()
        job.total_steps = result["global_step"]
        db.commit()

        experiment.duration_seconds = result["duration_seconds"]
        experiment.status = "COMPLETED"
        db.commit()

        model_entry = Model(
            name=f"{job.base_model_id.split('/')[-1]}-finetuned",
            base_model_id=job.base_model_id,
        )
        db.add(model_entry)
        db.flush()
        model_version = ModelVersion(
            model_id=model_entry.id,
            version_label=f"job-{job.id}",
            training_method=config.method.value,
            training_job_id=job.id,
            artifact_path=result["output_dir"],
        )
        db.add(model_version)
        db.commit()

        run_evaluation_task.delay(str(experiment.id), str(model_version.id))

    except Exception as exc:  # noqa: BLE001
        logger.exception("Training job %s failed", job_id)
        db.rollback()
        job = db.get(TrainingJob, uuid.UUID(job_id))
        if job:
            job.status = JobStatus.FAILED
            job.error_message = str(exc)[:2000]
            db.commit()
    finally:
        db.close()


@celery_app.task(name="evaluation.run")
def run_evaluation_task(experiment_id: str, model_version_id: str, max_samples: int = 10) -> None:
    db = SessionLocal()
    try:
        experiment = db.get(Experiment, uuid.UUID(experiment_id))
        model_version = db.get(ModelVersion, uuid.UUID(model_version_id))
        dataset_version = db.get(DatasetVersion, experiment.dataset_version_id)

        base_eval = run_evaluation(
            db, experiment, EvaluationType.BASE, dataset_version,
            experiment.base_model_id, None, max_samples, ["rouge", "bleu", "exact_match"],
        )
        ft_eval = run_evaluation(
            db, experiment, EvaluationType.FINE_TUNED, dataset_version,
            experiment.base_model_id, model_version.artifact_path, max_samples,
            ["rouge", "bleu", "exact_match"],
        )

        model_version.evaluation_scores = ft_eval.aggregate_scores
        db.commit()

        comparison = compute_comparison(base_eval, ft_eval)
        report_path = generate_report(experiment, comparison, ft_eval.sample_predictions)
        experiment.report_path = report_path
        db.commit()

        from app.models.evaluation import Artifact

        db.add(Artifact(experiment_id=experiment.id, artifact_type="report", path=report_path))
        db.add(Artifact(experiment_id=experiment.id, artifact_type="model", path=model_version.artifact_path))
        db.commit()

    except Exception:  # noqa: BLE001
        logger.exception("Evaluation for experiment %s failed", experiment_id)
        db.rollback()
    finally:
        db.close()
