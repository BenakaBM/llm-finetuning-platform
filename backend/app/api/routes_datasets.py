from fastapi import APIRouter, Depends, File, Form, UploadFile
from sqlalchemy.orm import Session

from app.api.deps import get_current_user
from app.core.errors import NotFoundError
from app.db.session import get_db
from app.models.dataset import Dataset, DatasetVersion
from app.models.user import User
from app.schemas.dataset import DatasetPreviewResponse, DatasetResponse, DatasetVersionResponse
from app.services.dataset_service import preview_dataset, upload_dataset

router = APIRouter(prefix="/api/datasets", tags=["datasets"])


@router.post("", response_model=DatasetResponse, status_code=201)
async def create_dataset(
    file: UploadFile = File(...),
    name: str = Form(...),
    description: str | None = Form(None),
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    return await upload_dataset(db, user, file, name, description)


@router.get("", response_model=list[DatasetResponse])
def list_datasets(db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    return db.query(Dataset).filter(Dataset.owner_id == user.id).order_by(Dataset.created_at.desc()).all()


@router.get("/{dataset_id}", response_model=DatasetResponse)
def get_dataset(dataset_id: str, db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    dataset = db.get(Dataset, dataset_id)
    if not dataset or dataset.owner_id != user.id:
        raise NotFoundError("Dataset not found")
    return dataset


@router.get("/{dataset_id}/versions", response_model=list[DatasetVersionResponse])
def list_dataset_versions(dataset_id: str, db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    dataset = db.get(Dataset, dataset_id)
    if not dataset or dataset.owner_id != user.id:
        raise NotFoundError("Dataset not found")
    return dataset.versions


@router.get("/versions/{version_id}/preview", response_model=DatasetPreviewResponse)
def preview_dataset_version(version_id: str, db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    version = db.get(DatasetVersion, version_id)
    if not version:
        raise NotFoundError("Dataset version not found")
    return preview_dataset(version)


@router.delete("/{dataset_id}", status_code=204)
def delete_dataset(dataset_id: str, db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    dataset = db.get(Dataset, dataset_id)
    if not dataset or dataset.owner_id != user.id:
        raise NotFoundError("Dataset not found")
    db.delete(dataset)
    db.commit()
