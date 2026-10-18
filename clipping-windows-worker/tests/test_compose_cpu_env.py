"""docker-compose.cpu.yml must not blank YOUTUBE_REFRESH_TOKEN.

`environment:` overrides `env_file:`, so a literal "" there silently disabled
YouTube publishing on the mini PC worker (it takes the token from .env.cpu).
"""
from __future__ import annotations

from pathlib import Path

import yaml

COMPOSE = Path(__file__).resolve().parents[1] / "docker-compose.cpu.yml"


def _service():
    return yaml.safe_load(COMPOSE.read_text(encoding="utf-8"))["services"]["worker-cpu"]


def test_youtube_token_comes_from_env_file():
    svc = _service()
    assert svc.get("env_file") in (".env.cpu", [".env.cpu"])
    env = svc.get("environment") or {}
    for key in ("YOUTUBE_REFRESH_TOKEN", "YOUTUBE_CLIENT_ID", "YOUTUBE_CLIENT_SECRET", "YOUTUBE_PRIVACY"):
        assert key not in env, f"{key} must come from .env.cpu, not be overridden in compose"


def test_instagram_and_tiktok_stay_off():
    env = _service().get("environment") or {}
    assert env.get("INSTAGRAM_ACCESS_TOKEN") == ""
    assert env.get("TIKTOK_ACCESS_TOKEN") == ""
