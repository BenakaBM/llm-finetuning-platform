"""Load raw dataset files (json/jsonl/csv/parquet) and normalize them into a
single internal record shape:

    {"instruction": str, "input": str, "output": str}

Both the "instruction" format and the "conversational" ({"messages": [...]})
format are supported and normalized to this shape so the rest of the
pipeline (tokenization, training, evaluation) only has to deal with one
representation.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from app.models.enums import DatasetFormat


# Common wrapper keys real-world exports use for a top-level records list,
# e.g. {"data": [...]}, {"examples": [...]}, {"records": [...]}.
_LIST_WRAPPER_KEYS = ("data", "examples", "records", "items", "rows", "dataset")


def _read_raw_records(path: Path, file_format: str) -> list[Any]:
    """Returns the raw, un-normalized list of entries from the file. Entries
    are NOT guaranteed to be dicts here - malformed/unexpected shapes are
    filtered out and counted in `load_and_normalize`, not here, so callers
    always get a clear count of what was dropped and why."""
    if file_format == "jsonl":
        records: list[Any] = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    records.append(None)  # tracked as a dropped/malformed row
        return records

    if file_format == "json":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, list):
            return data

        if isinstance(data, dict):
            # Look for a known wrapper key holding the actual list of records.
            for key in _LIST_WRAPPER_KEYS:
                if key in data and isinstance(data[key], list):
                    return data[key]
            # A single record posted directly at the top level, e.g.
            # {"instruction": "...", "output": "..."} or {"messages": [...]}.
            if "instruction" in data or "output" in data or "messages" in data:
                return [data]
            raise ValueError(
                "Could not find a list of records in the uploaded JSON file. "
                "Expected either a top-level JSON array, or an object with a "
                f"list under one of: {', '.join(_LIST_WRAPPER_KEYS)}."
            )

        raise ValueError("Uploaded JSON file must contain an array or object of records.")

    if file_format == "csv":
        df = pd.read_csv(path)
        df = df.where(pd.notnull(df), "")  # empty cells -> "" instead of NaN
        return df.to_dict(orient="records")

    if file_format == "parquet":
        df = pd.read_parquet(path)
        df = df.where(pd.notnull(df), "")
        return df.to_dict(orient="records")

    raise ValueError(f"Unsupported file format: {file_format}")


def detect_dataset_format(records: list[Any]) -> DatasetFormat:
    """Detects format by scanning leading records for the first one that is
    actually an object (dict), so a single malformed/blank entry at index 0
    can't crash detection or silently misdetect the format."""
    if not records:
        raise ValueError("Dataset contains no records.")

    sample = next((r for r in records[:20] if isinstance(r, dict)), None)
    if sample is None:
        raise ValueError(
            "Could not detect dataset format: no valid JSON objects were found "
            "in the first 20 rows. Each record must be an object like "
            '{"instruction": "...", "output": "..."} or {"messages": [...]}.'
        )

    if "messages" in sample:
        return DatasetFormat.CONVERSATIONAL
    if "instruction" in sample or "output" in sample:
        return DatasetFormat.INSTRUCTION
    raise ValueError(
        "Could not detect dataset format. Expected an 'instruction'/'output' "
        f"field or a 'messages' field. Found fields: {list(sample.keys())}"
    )


def normalize_record(record: Any, fmt: DatasetFormat) -> dict[str, str] | None:
    """Convert one raw record into the internal {instruction, input, output} shape.
    Returns None if the record cannot be normalized (missing required fields,
    or the record isn't a JSON object at all)."""
    if not isinstance(record, dict):
        return None

    if fmt == DatasetFormat.INSTRUCTION:
        instruction = str(record.get("instruction", "")).strip()
        input_text = str(record.get("input", "")).strip()
        output = str(record.get("output", "")).strip()
        if not instruction or not output:
            return None
        return {"instruction": instruction, "input": input_text, "output": output}

    if fmt == DatasetFormat.CONVERSATIONAL:
        messages = record.get("messages", [])
        if not isinstance(messages, list):
            return None
        messages = [m for m in messages if isinstance(m, dict)]
        user_turns = [m.get("content", "") for m in messages if m.get("role") == "user"]
        assistant_turns = [m.get("content", "") for m in messages if m.get("role") == "assistant"]
        if not user_turns or not assistant_turns:
            return None
        instruction = str(user_turns[0]).strip()
        output = str(assistant_turns[-1]).strip()
        if not instruction or not output:
            return None
        return {"instruction": instruction, "input": "", "output": output}

    raise ValueError(f"Unsupported dataset format: {fmt}")


def load_and_normalize(path: str, file_format: str) -> tuple[DatasetFormat, list[dict[str, str]], int]:
    """Returns (detected_format, normalized_records, num_dropped_records).

    Malformed or unexpected rows (not a JSON object, missing required
    fields, unparsable JSONL lines) are dropped and counted rather than
    raising - the caller sees an accurate dropped count in the dataset's
    quality report instead of the whole upload failing because of a
    handful of bad rows."""
    raw_records = _read_raw_records(Path(path), file_format)
    fmt = detect_dataset_format(raw_records)
    normalized = []
    dropped = 0
    for r in raw_records:
        n = normalize_record(r, fmt)
        if n is None:
            dropped += 1
        else:
            normalized.append(n)
    return fmt, normalized, dropped


def format_as_prompt(record: dict[str, str]) -> str:
    """Render a normalized record as a single training string."""
    if record.get("input"):
        return (
            f"### Instruction:\n{record['instruction']}\n\n"
            f"### Input:\n{record['input']}\n\n"
            f"### Response:\n{record['output']}"
        )
    return f"### Instruction:\n{record['instruction']}\n\n### Response:\n{record['output']}"


def format_prompt_only(record: dict[str, str]) -> str:
    """Prompt portion only (no answer) - used at inference/eval time."""
    if record.get("input"):
        return (
            f"### Instruction:\n{record['instruction']}\n\n"
            f"### Input:\n{record['input']}\n\n### Response:\n"
        )
    return f"### Instruction:\n{record['instruction']}\n\n### Response:\n"
