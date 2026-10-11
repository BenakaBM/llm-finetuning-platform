"""Consistent, structured API error handling."""
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.core.logging import get_logger

logger = get_logger(__name__)


class AppError(Exception):
    """Base application error mapped to a structured JSON response."""

    def __init__(self, code: str, message: str, status_code: int = 400):
        self.code = code
        self.message = message
        self.status_code = status_code
        super().__init__(message)


class NotFoundError(AppError):
    def __init__(self, message: str = "Resource not found"):
        super().__init__("NOT_FOUND", message, 404)


class ValidationAppError(AppError):
    def __init__(self, message: str = "Validation failed"):
        super().__init__("VALIDATION_ERROR", message, 422)


class AuthError(AppError):
    def __init__(self, message: str = "Not authenticated"):
        super().__init__("AUTH_ERROR", message, 401)


class ResourceError(AppError):
    def __init__(self, message: str = "Insufficient resources"):
        super().__init__("RESOURCE_ERROR", message, 409)


def _error_payload(code: str, message: str) -> dict:
    return {"success": False, "error": {"code": code, "message": message}}


def register_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(AppError)
    async def handle_app_error(request: Request, exc: AppError):
        return JSONResponse(status_code=exc.status_code, content=_error_payload(exc.code, exc.message))

    @app.exception_handler(Exception)
    async def handle_unexpected(request: Request, exc: Exception):
        logger.exception("Unhandled exception on %s %s", request.method, request.url)
        return JSONResponse(
            status_code=500,
            content=_error_payload("INTERNAL_ERROR", "An unexpected error occurred."),
        )
