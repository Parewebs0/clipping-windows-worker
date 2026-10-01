"""Configuración central del Worker mediante pydantic-settings."""
from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    worker_id: str = "windows-gpu-worker-01"
    api_base_url: str = "https://internal-api.example.com"
    api_token: str = "CHANGE_ME"
    poll_interval: float = Field(default=5.0, ge=1.0)
    heartbeat_interval: float = Field(default=30.0, ge=1.0)
    max_concurrent_jobs: int = Field(default=1, ge=1, le=8)
    working_directory: Path = PROJECT_ROOT / "data"
    clip_storage_root: Path = Field(default=Path(r"C:\CODIANT\clipping\storage\clips"))
    render_output_fps: float = Field(default=30.0, gt=0.0)
    device: Literal["cuda", "cpu"] = "cuda"
    compute_type: str = "float16"
    allow_cpu_fallback: bool = False
    auto_cleanup: bool = True
    job_retention_hours: float = Field(default=24.0, ge=0.0)
    log_level: str = "INFO"
    ffmpeg_path: str | None = None
    ffprobe_path: str | None = None
    gog_path: str | None = None
    gog_account: str | None = None
    gog_keyring_password: str | None = None
    youtube_client_id: str | None = None
    youtube_client_secret: str | None = None
    youtube_refresh_token: str | None = None
    youtube_privacy: str = "public"
    instagram_access_token: str | None = None
    instagram_ig_user_id: str | None = None
    tiktok_access_token: str | None = None
    tiktok_privacy: str = "SELF_ONLY"

    @property
    def downloads_dir(self) -> Path:
        return self.working_directory / "downloads"

    @property
    def jobs_dir(self) -> Path:
        return self.working_directory / "jobs"

    @property
    def outputs_dir(self) -> Path:
        return self.working_directory / "outputs"

    @property
    def temp_dir(self) -> Path:
        return self.working_directory / "temp"

    @property
    def logs_dir(self) -> Path:
        return self.working_directory / "logs"

    @field_validator("working_directory", mode="before")
    @classmethod
    def _expand_working_directory(cls, value: object) -> object:
        if isinstance(value, str):
            return Path(value).expanduser()
        return value

    def ensure_directories(self) -> None:
        for directory in (
            self.working_directory,
            self.downloads_dir,
            self.jobs_dir,
            self.outputs_dir,
            self.temp_dir,
            self.logs_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)


def get_settings() -> Settings:
    if not hasattr(get_settings, "_cached"):
        get_settings._cached = Settings()  # type: ignore[attr-defined]
    return get_settings._cached  # type: ignore[attr-defined]
