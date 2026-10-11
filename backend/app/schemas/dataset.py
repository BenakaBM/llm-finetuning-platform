import uuid
from datetime import datetime

from pydantic import BaseModel

from app.models.enums import DatasetFormat


class DatasetResponse(BaseModel):
    id: uuid.UUID
    name: str
    description: str | None
    file_format: str
    dataset_format: DatasetFormat
    record_count: int
    created_at: datetime

    class Config:
        from_attributes = True


class DatasetVersionResponse(BaseModel):
    id: uuid.UUID
    dataset_id: uuid.UUID
    version_number: int
    statistics: dict
    quality_report: dict
    created_at: datetime

    class Config:
        from_attributes = True


class DatasetPreviewResponse(BaseModel):
    columns: list[str]
    rows: list[dict]
    total_records: int
