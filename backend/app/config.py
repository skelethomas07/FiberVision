from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_url: str = "sqlite:///./sem_fiber.sqlite3"
    redis_url: str = "redis://redis:6379/0"
    storage_backend: str = "local"
    local_storage_dir: Path = Path("./storage")
    s3_endpoint_url: str | None = None
    s3_access_key: str = "minioadmin"
    s3_secret_key: str = "minioadmin"
    s3_bucket: str = "sem-fiber"
    s3_region: str = "us-east-1"
    model_checkpoint: Path = Path("./models/v6_12/best_full.pt")
    model_version: str = "v6.12"
    model_device: str = "auto"
    work_dir: Path = Path("./work")
    frontend_origin: str = "http://localhost:3000"
    auth_session_days: int = 7
    auth_cookie_name: str = "fibervision_session"
    auth_cookie_secure: bool = False


@lru_cache
def get_settings() -> Settings:
    return Settings()
