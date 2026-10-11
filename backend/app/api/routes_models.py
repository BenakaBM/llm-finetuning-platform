from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.api.deps import get_current_user
from app.core.errors import NotFoundError
from app.db.session import get_db
from app.models.model import Model
from app.models.user import User

router = APIRouter(prefix="/api/models", tags=["models"])


class ModelRegisterRequest(BaseModel):
    name: str
    base_model_id: str
    architecture: str | None = None
    parameter_count: str | None = None
    context_length: int | None = None


@router.get("")
def list_models(db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    models = db.query(Model).order_by(Model.created_at.desc()).all()
    return [
        {
            "id": m.id,
            "name": m.name,
            "base_model_id": m.base_model_id,
            "architecture": m.architecture,
            "parameter_count": m.parameter_count,
            "context_length": m.context_length,
            "versions": [
                {
                    "id": v.id,
                    "version_label": v.version_label,
                    "training_method": v.training_method,
                    "evaluation_scores": v.evaluation_scores,
                    "artifact_path": v.artifact_path,
                }
                for v in m.versions
            ],
        }
        for m in models
    ]


@router.post("", status_code=201)
def register_model(payload: ModelRegisterRequest, db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    model = Model(**payload.model_dump())
    db.add(model)
    db.commit()
    db.refresh(model)
    return model


@router.get("/{model_id}")
def get_model(model_id: str, db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    model = db.get(Model, model_id)
    if not model:
        raise NotFoundError("Model not found")
    return model
