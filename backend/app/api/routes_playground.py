from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.api.deps import get_current_user
from app.core.errors import NotFoundError
from app.db.session import get_db
from app.ml.inference.generate import generate, load_adapter_model, load_base_model
from app.ml.training.hardware import detect_hardware
from app.models.model import ModelVersion
from app.models.user import User
from app.schemas.evaluation import PlaygroundRequest, PlaygroundResponse

router = APIRouter(prefix="/api/playground", tags=["playground"])
system_router = APIRouter(prefix="/api/system", tags=["system"])


@router.post("/generate", response_model=PlaygroundResponse)
def playground_generate(payload: PlaygroundRequest, db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    if payload.model_version_id:
        version = db.get(ModelVersion, payload.model_version_id)
        if not version:
            raise NotFoundError("Model version not found")
        model, tokenizer = load_adapter_model(version.model.base_model_id, version.artifact_path)
    elif payload.base_model_id:
        model, tokenizer = load_base_model(payload.base_model_id)
    else:
        from app.core.errors import ValidationAppError

        raise ValidationAppError("Provide either model_version_id or base_model_id.")

    result = generate(
        model, tokenizer, payload.prompt,
        temperature=payload.temperature, top_p=payload.top_p, max_new_tokens=payload.max_tokens,
    )
    return PlaygroundResponse(**result)


@system_router.get("/info")
def system_info():
    return detect_hardware()
