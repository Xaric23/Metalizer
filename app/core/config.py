from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from sqlalchemy.engine import make_url


class AppSettings(BaseSettings):
    project_name: str = Field(default="Metalizer")
    media_root: Path = Field(default=Path("data"))
    input_dir: Path = Field(default=Path("data/inputs"))
    output_dir: Path = Field(default=Path("data/outputs"))
    assets_dir: Path = Field(default=Path("assets"))
    asset_max_bytes: int = Field(default=25 * 1024 * 1024, description="Max asset upload size in bytes")
    jobs_db_url: str = Field(default="sqlite:///data/jobs.sqlite3")
    job_retention_hours: int = Field(default=24)
    queue_backend: str = Field(default="local", description="local or rq")
    redis_url: str | None = Field(default="redis://localhost:6379/0")
    redis_queue_name: str = Field(default="metalizer")
    assets_api_key: str | None = Field(default=None)

    model_config = SettingsConfigDict(env_prefix="METALIZER_", env_file=".env", extra="ignore")


@lru_cache
def get_settings() -> AppSettings:
    settings = AppSettings()
    settings.input_dir.mkdir(parents=True, exist_ok=True)
    settings.output_dir.mkdir(parents=True, exist_ok=True)
    settings.assets_dir.mkdir(parents=True, exist_ok=True)
    _ensure_sqlite_dir(settings.jobs_db_url)
    return settings


def _ensure_sqlite_dir(url: str) -> None:
    try:
        sa_url = make_url(url)
    except Exception:
        return
    if sa_url.get_backend_name() != "sqlite" or not sa_url.database:
        return
    db_path = Path(sa_url.database)
    if not db_path.is_absolute():
        db_path = Path.cwd() / db_path
    db_path.parent.mkdir(parents=True, exist_ok=True)
