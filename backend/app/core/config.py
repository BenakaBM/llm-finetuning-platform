"""Central application configuration, loaded from environment variables."""
from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    APP_NAME: str = "LLM Fine-Tuning & Evaluation Platform"
    ENV: str = "development"
    SECRET_KEY: str = "change-me-in-prod"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24

    DATABASE_URL: str = "postgresql+psycopg2://postgres:Ben0111@localhost:5432/llmft"
    REDIS_URL: str = "redis://localhost:6379/0"

    STORAGE_PATH: str = str(Path(__file__).resolve().parents[3] / "storage")
    HF_HOME: str = str(Path(__file__).resolve().parents[3] / "data" / "hf_cache")
    MODEL_CACHE: str = str(Path(__file__).resolve().parents[3] / "data" / "models")

    MAX_UPLOAD_MB: int = 200
    ALLOWED_DATASET_EXTENSIONS: tuple = (".json", ".jsonl", ".csv", ".parquet")

    CORS_ORIGINS: list[str] = ["http://localhost:5173", "http://localhost:3000"]

    DEFAULT_BASE_MODEL: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    DEMO_TINY_MODEL: str = "sshleifer/tiny-gpt2"


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    Path(settings.STORAGE_PATH).mkdir(parents=True, exist_ok=True)
    for sub in ("datasets", "models", "checkpoints", "evaluations", "experiments"):
        Path(settings.STORAGE_PATH, sub).mkdir(parents=True, exist_ok=True)
    Path(settings.HF_HOME).mkdir(parents=True, exist_ok=True)
    Path(settings.MODEL_CACHE).mkdir(parents=True, exist_ok=True)
    return settings
