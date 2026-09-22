"""Runtime configuration. Everything comes from the environment (12-factor)."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="VEIL_", env_file=".env", extra="ignore")

    database_url: str = "sqlite:///./var/veil.db"
    artifact_root: Path = Path("./var/artifacts")
    # Optional seed key for local development / first run. Never log it.
    bootstrap_api_key: str = ""
    max_upload_bytes: int = 25 * 1024 * 1024
    allowed_upload_types: tuple[str, ...] = (
        "image/png",
        "image/jpeg",
        "image/webp",
        "video/mp4",
    )
    log_level: str = "INFO"
    # Browser origins allowed to call the API (VEIL_CORS_ORIGINS='["https://..."]').
    cors_origins: tuple[str, ...] = ("http://localhost:3000", "http://127.0.0.1:3000")
    # Experiment workers run in-process (see ADR-005). One at a time keeps
    # GPU/CPU contention predictable on a single box.
    max_concurrent_runs: int = 1
    # At startup, runs still queued/running are closed as failed: their worker
    # thread died with the previous process. True only while ONE API process
    # owns the database - with several workers or containers, set this to
    # false, or each start fails the others' live runs.
    fail_orphaned_runs_on_start: bool = True


@lru_cache
def get_settings() -> Settings:
    return Settings()
