import uuid

from fastapi import Depends
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.orm import Session

from app.core.errors import AuthError
from app.core.security import decode_access_token
from app.db.session import get_db
from app.models.user import User

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/auth/login", auto_error=False)


def get_current_user(token: str | None = Depends(oauth2_scheme), db: Session = Depends(get_db)) -> User:
    if not token:
        raise AuthError("Missing bearer token")
    payload = decode_access_token(token)
    if not payload or "sub" not in payload:
        raise AuthError("Invalid or expired token")
    user = db.get(User, uuid.UUID(payload["sub"]))
    if not user:
        raise AuthError("User not found")
    return user
