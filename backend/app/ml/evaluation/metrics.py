"""Modular evaluation metrics. Each function takes (predictions, references)
and returns real, computed scores - nothing here is hard-coded or faked.

New metrics can be added by writing a function with the same signature and
registering it in METRIC_REGISTRY.
"""
from __future__ import annotations

from typing import Callable

from rouge_score import rouge_scorer
import sacrebleu


def compute_rouge(predictions: list[str], references: list[str]) -> dict[str, float]:
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    totals = {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}
    for pred, ref in zip(predictions, references):
        scores = scorer.score(ref, pred)
        for key in totals:
            totals[key] += scores[key].fmeasure
    n = max(len(predictions), 1)
    return {k: round(v / n, 4) for k, v in totals.items()}


def compute_bleu(predictions: list[str], references: list[str]) -> dict[str, float]:
    if not predictions:
        return {"bleu": 0.0}
    bleu = sacrebleu.corpus_bleu(predictions, [references])
    return {"bleu": round(bleu.score / 100.0, 4)}


def compute_exact_match(predictions: list[str], references: list[str]) -> dict[str, float]:
    if not predictions:
        return {"exact_match": 0.0}
    matches = sum(
        1 for p, r in zip(predictions, references) if p.strip().lower() == r.strip().lower()
    )
    return {"exact_match": round(matches / len(predictions), 4)}


def compute_bertscore(predictions: list[str], references: list[str]) -> dict[str, float]:
    try:
        from bert_score import score as bert_score_fn
    except ImportError:
        return {"bertscore_f1": None}
    if not predictions:
        return {"bertscore_f1": 0.0}
    _, _, f1 = bert_score_fn(predictions, references, lang="en", verbose=False)
    return {"bertscore_f1": round(float(f1.mean()), 4)}


def compute_classification_metrics(predictions: list[str], references: list[str]) -> dict[str, float]:
    """Only meaningful when outputs are short categorical labels."""
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

    if not predictions:
        return {}
    y_pred = [p.strip().lower() for p in predictions]
    y_true = [r.strip().lower() for r in references]
    labels = sorted(set(y_true) | set(y_pred))
    if len(labels) > 20:
        return {}  # not a classification-style dataset; skip
    return {
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "precision": round(precision_score(y_true, y_pred, average="macro", zero_division=0), 4),
        "recall": round(recall_score(y_true, y_pred, average="macro", zero_division=0), 4),
        "f1": round(f1_score(y_true, y_pred, average="macro", zero_division=0), 4),
    }


METRIC_REGISTRY: dict[str, Callable[[list[str], list[str]], dict[str, float]]] = {
    "rouge": compute_rouge,
    "bleu": compute_bleu,
    "exact_match": compute_exact_match,
    "bertscore": compute_bertscore,
    "classification": compute_classification_metrics,
}


def run_metrics(metric_names: list[str], predictions: list[str], references: list[str]) -> dict[str, float]:
    results: dict[str, float] = {}
    for name in metric_names:
        fn = METRIC_REGISTRY.get(name)
        if fn is None:
            continue
        results.update(fn(predictions, references))
    return results
