from __future__ import annotations

import json
import time
import uuid

from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.ml.datasets.loader import format_prompt_only
from app.ml.evaluation.metrics import run_metrics
from app.ml.inference.generate import generate, load_adapter_model, load_base_model
from app.models.dataset import DatasetVersion
from app.models.evaluation import Evaluation, EvaluationResult, Experiment
from app.models.enums import EvaluationType

settings = get_settings()


def _load_test_records(dataset_version: DatasetVersion, max_samples: int) -> list[dict]:
    records = []
    with open(dataset_version.test_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= max_samples:
                break
            line = line.strip()
            if line:
                records.append(json.loads(line))
    if not records:
        # fall back to validation split if test split ended up empty
        with open(dataset_version.validation_path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i >= max_samples:
                    break
                line = line.strip()
                if line:
                    records.append(json.loads(line))
    return records


def run_evaluation(
    db: Session,
    experiment: Experiment,
    evaluation_type: EvaluationType,
    dataset_version: DatasetVersion,
    base_model_id: str,
    adapter_path: str | None,
    max_samples: int,
    metric_names: list[str],
) -> Evaluation:
    evaluation = Evaluation(
        experiment_id=experiment.id,
        evaluation_type=evaluation_type,
        dataset_version_id=dataset_version.id,
        status="RUNNING",
    )
    db.add(evaluation)
    db.commit()
    db.refresh(evaluation)

    records = _load_test_records(dataset_version, max_samples)

    if evaluation_type == EvaluationType.FINE_TUNED and adapter_path:
        model, tokenizer = load_adapter_model(base_model_id, adapter_path)
    else:
        model, tokenizer = load_base_model(base_model_id)

    predictions, references, latencies = [], [], []
    sample_predictions = []

    for idx, record in enumerate(records):
        prompt = format_prompt_only(record)
        result = generate(model, tokenizer, prompt, temperature=0.0, max_new_tokens=96)
        predictions.append(result["response"])
        references.append(record["output"])
        latencies.append(result["latency_ms"])

        row = EvaluationResult(
            evaluation_id=evaluation.id,
            example_index=idx,
            prompt=prompt[:4000],
            reference=record["output"][:4000],
            prediction=result["response"][:4000],
            scores={},
        )
        db.add(row)
        if idx < 5:
            sample_predictions.append(
                {"prompt": prompt, "reference": record["output"], "prediction": result["response"]}
            )

    aggregate_scores = run_metrics(metric_names, predictions, references)

    evaluation.status = "COMPLETED"
    evaluation.aggregate_scores = aggregate_scores
    evaluation.sample_predictions = sample_predictions
    evaluation.latency_ms_avg = round(sum(latencies) / len(latencies), 2) if latencies else None
    from datetime import datetime

    evaluation.completed_at = datetime.utcnow()

    db.commit()
    db.refresh(evaluation)
    return evaluation


def compute_comparison(base_eval: Evaluation, fine_tuned_eval: Evaluation) -> list[dict]:
    rows = []
    for metric, base_value in base_eval.aggregate_scores.items():
        ft_value = fine_tuned_eval.aggregate_scores.get(metric)
        if base_value is None or ft_value is None:
            continue
        improvement = None
        if base_value:
            improvement = round(((ft_value - base_value) / abs(base_value)) * 100, 1)
        rows.append(
            {
                "metric": metric,
                "base": base_value,
                "fine_tuned": ft_value,
                "improvement_pct": improvement,
            }
        )
    return rows
