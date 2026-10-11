from __future__ import annotations

import json
import uuid
from pathlib import Path

from fastapi import UploadFile
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.core.errors import ValidationAppError
from app.ml.datasets.loader import load_and_normalize
from app.ml.datasets.validation import (
    compute_quality_report,
    compute_statistics,
    split_records,
    validate_records,
)
from app.models.dataset import Dataset, DatasetVersion
from app.models.user import User

settings = get_settings()


def _safe_filename(filename: str) -> str:
    name = Path(filename).name
    return "".join(c for c in name if c.isalnum() or c in (".", "_", "-")) or "upload"


async def upload_dataset(db: Session, user: User, file: UploadFile, name: str, description: str | None) -> Dataset:
    ext = Path(file.filename or "").suffix.lower()
    if ext not in settings.ALLOWED_DATASET_EXTENSIONS:
        raise ValidationAppError(
            f"Unsupported file extension '{ext}'. Allowed: {settings.ALLOWED_DATASET_EXTENSIONS}"
        )

    contents = await file.read()
    size_mb = len(contents) / (1024 * 1024)
    if size_mb > settings.MAX_UPLOAD_MB:
        raise ValidationAppError(f"File too large ({size_mb:.1f} MB). Max is {settings.MAX_UPLOAD_MB} MB.")

    dataset_id = uuid.uuid4()
    dataset_dir = Path(settings.STORAGE_PATH, "datasets", str(dataset_id))
    dataset_dir.mkdir(parents=True, exist_ok=True)
    safe_name = _safe_filename(file.filename or "dataset")
    raw_path = dataset_dir / f"raw{ext}"
    raw_path.write_bytes(contents)

    file_format = ext.lstrip(".")
    try:
        dataset_format, normalized, dropped = load_and_normalize(str(raw_path), file_format)
    except Exception as exc:  # noqa: BLE001
        raise ValidationAppError(f"Failed to parse dataset: {exc}") from exc

    errors = validate_records(normalized)
    if errors:
        raise ValidationAppError("; ".join(errors))

    dataset = Dataset(
        id=dataset_id,
        owner_id=user.id,
        name=name,
        description=description,
        file_format=file_format,
        dataset_format=dataset_format,
        storage_path=str(raw_path),
        record_count=len(normalized),
    )
    db.add(dataset)

    version = _create_version(db, dataset, normalized, dropped, version_number=1)
    db.commit()
    db.refresh(dataset)
    db.refresh(version)
    return dataset


def _create_version(
    db: Session, dataset: Dataset, normalized: list[dict], dropped: int, version_number: int
) -> DatasetVersion:
    stats = compute_statistics(normalized)
    quality = compute_quality_report(normalized, stats, dropped)
    train, val, test = split_records(normalized)

    version_dir = Path(settings.STORAGE_PATH, "datasets", str(dataset.id), f"v{version_number}")
    version_dir.mkdir(parents=True, exist_ok=True)

    paths = {}
    for split_name, split_data in (("train", train), ("validation", val), ("test", test)):
        p = version_dir / f"{split_name}.jsonl"
        with open(p, "w", encoding="utf-8") as f:
            for r in split_data:
                f.write(json.dumps(r) + "\n")
        paths[split_name] = str(p)

    version = DatasetVersion(
        dataset_id=dataset.id,
        version_number=version_number,
        train_path=paths["train"],
        validation_path=paths["validation"],
        test_path=paths["test"],
        statistics=stats,
        quality_report=quality,
    )
    db.add(version)
    return version


def preview_dataset(version: DatasetVersion, limit: int = 20) -> dict:
    rows = []
    with open(version.train_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= limit:
                break
            rows.append(json.loads(line))
    return {"columns": ["instruction", "input", "output"], "rows": rows, "total_records": len(rows)}
