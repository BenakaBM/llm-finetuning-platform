from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes_auth import router as auth_router
from app.api.routes_dashboard import router as dashboard_router
from app.api.routes_datasets import router as datasets_router
from app.api.routes_evaluations import router as evaluations_router
from app.api.routes_experiments import router as experiments_router
from app.api.routes_models import router as models_router
from app.api.routes_playground import router as playground_router
from app.api.routes_playground import system_router
from app.api.routes_training import router as training_router
from app.core.config import get_settings
from app.core.errors import register_error_handlers
from app.core.logging import configure_logging
from app.db.session import init_db

settings = get_settings()
configure_logging()

app = FastAPI(title=settings.APP_NAME, version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

register_error_handlers(app)

app.include_router(auth_router)
app.include_router(datasets_router)
app.include_router(models_router)
app.include_router(training_router)
app.include_router(evaluations_router)
app.include_router(experiments_router)
app.include_router(playground_router)
app.include_router(system_router)
app.include_router(dashboard_router)


@app.on_event("startup")
def on_startup() -> None:
    # Auto-creates tables for local development. Production deployments
    # should use `alembic upgrade head` instead (see backend/alembic/).
    init_db()


@app.get("/")
def root():
    return {"name": settings.APP_NAME, "status": "ok", "docs": "/docs"}


@app.get("/health")
def health():
    return {"status": "healthy"}
