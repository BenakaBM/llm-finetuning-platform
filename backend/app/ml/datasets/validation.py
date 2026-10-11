"""Real dataset validation, quality scoring, and statistics computation.

No numbers here are fabricated - everything is computed directly from the
normalized records that were actually loaded from the uploaded file.
"""
from __future__ import annotations

import hashlib
import random
from typing import Any


def _tokenize_len(text: str, tokenizer=None) -> int:
    if tokenizer is not None:
        return len(tokenizer.encode(text, add_special_tokens=False))
    # Cheap deterministic fallback when no tokenizer is loaded (keeps
    # dataset validation fast and independent of downloading a model).
    return max(1, len(text.split()))


def compute_statistics(records: list[dict[str, str]], tokenizer=None) -> dict[str, Any]:
    from app.ml.datasets.loader import format_as_prompt

    if not records:
        return {
            "records": 0, "columns": [], "missing_values": 0, "duplicates": 0,
            "average_tokens": 0, "maximum_tokens": 0,
        }

    lengths = []
    seen_hashes = set()
    duplicates = 0
    missing_values = 0

    for r in records:
        text = format_as_prompt(r)
        lengths.append(_tokenize_len(text, tokenizer))

        h = hashlib.sha256(text.encode("utf-8")).hexdigest()
        if h in seen_hashes:
            duplicates += 1
        seen_hashes.add(h)

        if not r.get("instruction") or not r.get("output"):
            missing_values += 1

    return {
        "records": len(records),
        "columns": ["instruction", "input", "output"],
        "missing_values": missing_values,
        "duplicates": duplicates,
        "average_tokens": round(sum(lengths) / len(lengths), 2),
        "maximum_tokens": max(lengths),
        "minimum_tokens": min(lengths),
    }


def compute_quality_report(
    records: list[dict[str, str]],
    stats: dict[str, Any],
    dropped_records: int,
    max_seq_length: int = 512,
) -> dict[str, Any]:
    total = max(stats["records"] + dropped_records, 1)

    duplicate_pct = stats["duplicates"] / total * 100
    empty_output_pct = sum(1 for r in records if not r.get("output", "").strip()) / total * 100
    dropped_pct = dropped_records / total * 100

    from app.ml.datasets.loader import format_as_prompt

    long_seq_count = 0
    for r in records:
        if _tokenize_len(format_as_prompt(r)) > max_seq_length:
            long_seq_count += 1
    long_seq_pct = long_seq_count / total * 100

    warnings = []
    if duplicate_pct > 0:
        warnings.append(f"{duplicate_pct:.1f}% duplicate examples")
    if empty_output_pct > 0:
        warnings.append(f"{empty_output_pct:.1f}% empty outputs")
    if long_seq_pct > 0:
        warnings.append(f"{long_seq_pct:.1f}% sequences longer than {max_seq_length} tokens (will be truncated)")
    if dropped_pct > 0:
        warnings.append(f"{dropped_pct:.1f}% records dropped due to missing required fields")

    # Simple, transparent scoring rubric - deductions are proportional to
    # measured issues, capped at 0.
    score = 100.0
    score -= min(duplicate_pct * 2, 20)
    score -= min(empty_output_pct * 3, 30)
    score -= min(long_seq_pct * 1, 15)
    score -= min(dropped_pct * 2, 25)
    score = max(round(score), 0)

    return {"score": score, "warnings": warnings}


def validate_records(records: list[dict[str, str]]) -> list[str]:
    """Structural validation. Returns a list of human-readable error strings
    (empty list means valid)."""
    errors = []
    if len(records) == 0:
        errors.append("Dataset contains zero valid records after normalization.")
    if len(records) < 2:
        errors.append("Dataset must contain at least 2 records to create train/val splits.")
    return errors


def split_records(
    records: list[dict[str, str]],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> tuple[list, list, list]:
    rng = random.Random(seed)
    shuffled = records[:]
    rng.shuffle(shuffled)

    n = len(shuffled)
    n_train = max(1, int(n * train_ratio))
    n_val = max(1 if n - n_train > 1 else 0, int(n * val_ratio))
    n_val = min(n_val, max(0, n - n_train - 1)) if n - n_train > 1 else 0

    train = shuffled[:n_train]
    val = shuffled[n_train:n_train + n_val]
    test = shuffled[n_train + n_val:]

    # Guarantee non-empty splits when there's enough data
    if not val and len(test) > 1:
        val = [test.pop()]
    if not test and len(train) > 2:
        test = [train.pop()]

    return train, val, test
