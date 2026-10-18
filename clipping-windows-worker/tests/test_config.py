"""Tests de configuración y rutas derivadas."""

from __future__ import annotations

from pathlib import Path

from app.config import Settings


def test_default_settings(tmp_path: Path) -> None:
    settings = Settings(working_directory=str(tmp_path))
    assert settings.worker_id == "windows-gpu-worker-01"
    assert settings.max_concurrent_jobs == 1
    assert settings.device == "cuda"
    assert settings.allow_cpu_fallback is False


def test_derived_directories(tmp_path: Path) -> None:
    settings = Settings(working_directory=str(tmp_path))
    assert settings.downloads_dir == tmp_path / "downloads"
    assert settings.jobs_dir == tmp_path / "jobs"
    assert settings.outputs_dir == tmp_path / "outputs"
    assert settings.temp_dir == tmp_path / "temp"
    assert settings.logs_dir == tmp_path / "logs"


def test_ensure_directories_creates_all(tmp_path: Path) -> None:
    settings = Settings(working_directory=str(tmp_path))
    settings.ensure_directories()
    for directory in (
        settings.working_directory,
        settings.downloads_dir,
        settings.jobs_dir,
        settings.outputs_dir,
        settings.temp_dir,
        settings.logs_dir,
    ):
        assert directory.is_dir()


def test_working_directory_expands_user(tmp_path: Path) -> None:
    settings = Settings(working_directory=str(tmp_path))
    assert settings.working_directory == tmp_path


def test_youtube_privacy_defaults_to_private(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("YOUTUBE_PRIVACY", raising=False)
    settings = Settings(working_directory=str(tmp_path))
    assert settings.youtube_privacy == "private"


def test_youtube_privacy_from_env(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("YOUTUBE_PRIVACY", " Unlisted ")
    assert Settings(working_directory=str(tmp_path)).youtube_privacy == "unlisted"
    monkeypatch.setenv("YOUTUBE_PRIVACY", "")
    assert Settings(working_directory=str(tmp_path)).youtube_privacy == "private"


def test_youtube_privacy_rejects_invalid(tmp_path: Path, monkeypatch) -> None:
    import pytest
    from pydantic import ValidationError

    monkeypatch.setenv("YOUTUBE_PRIVACY", "everyone")
    with pytest.raises(ValidationError):
        Settings(working_directory=str(tmp_path))


def test_instagram_auth_and_graph_version_defaults(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("INSTAGRAM_AUTH", raising=False)
    monkeypatch.delenv("INSTAGRAM_GRAPH_VERSION", raising=False)
    monkeypatch.delenv("INSTAGRAM_APP_ID", raising=False)
    monkeypatch.delenv("INSTAGRAM_APP_SECRET", raising=False)
    settings = Settings(working_directory=str(tmp_path))
    assert settings.instagram_auth == "facebook"
    assert settings.instagram_graph_version == "v21.0"
    assert settings.instagram_app_id is None
    assert settings.instagram_app_secret is None


def test_instagram_auth_and_graph_version_from_env(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("INSTAGRAM_AUTH", " Instagram ")
    monkeypatch.setenv("INSTAGRAM_GRAPH_VERSION", " v22.0 ")
    monkeypatch.setenv("INSTAGRAM_APP_ID", "")
    monkeypatch.setenv("INSTAGRAM_APP_SECRET", "")
    settings = Settings(working_directory=str(tmp_path))
    assert settings.instagram_auth == "instagram"
    assert settings.instagram_graph_version == "v22.0"
    assert settings.instagram_app_id is None
    assert settings.instagram_app_secret is None


def test_instagram_auth_and_graph_version_reject_invalid(tmp_path: Path, monkeypatch) -> None:
    import pytest
    from pydantic import ValidationError

    monkeypatch.setenv("INSTAGRAM_AUTH", "meta")
    with pytest.raises(ValidationError):
        Settings(working_directory=str(tmp_path))
    monkeypatch.setenv("INSTAGRAM_AUTH", "facebook")
    monkeypatch.setenv("INSTAGRAM_GRAPH_VERSION", "21.0")
    with pytest.raises(ValidationError):
        Settings(working_directory=str(tmp_path))


def test_tiktok_mode_defaults_to_inbox(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("TIKTOK_MODE", raising=False)
    settings = Settings(working_directory=str(tmp_path))
    assert settings.tiktok_mode == "inbox"
    assert settings.tiktok_privacy == "SELF_ONLY"


def test_tiktok_mode_normalises_and_rejects_unknown(tmp_path: Path, monkeypatch) -> None:
    import pytest
    from pydantic import ValidationError

    monkeypatch.setenv("TIKTOK_MODE", " Direct ")
    assert Settings(working_directory=str(tmp_path)).tiktok_mode == "direct"
    monkeypatch.setenv("TIKTOK_MODE", "")
    assert Settings(working_directory=str(tmp_path)).tiktok_mode == "inbox"
    monkeypatch.setenv("TIKTOK_MODE", "story")
    with pytest.raises(ValidationError):
        Settings(working_directory=str(tmp_path))


def test_empty_publish_state_dir_means_default(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("PUBLISH_STATE_DIR", "")
    assert Settings(working_directory=str(tmp_path)).publish_state_dir is None
