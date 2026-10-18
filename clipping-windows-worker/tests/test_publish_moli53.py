"""MOLI-53 worker publish: scheduled_at guard, per-account credentials and
TikTok full per-platform copy. Uploaders are faked (no network)."""
from __future__ import annotations

import json
import sys
import types
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from app.config import Settings
from app.jobs.publish import PublishJob, _account_env
from app.models.job import Job

CLIP_ID = "33333333-2222-3333-4444-555555555555"
CAMPAIGN_ID = 9
FUTURE = (datetime.now(timezone.utc) + timedelta(hours=2)).isoformat()
PAST = (datetime.now(timezone.utc) - timedelta(hours=2)).isoformat()
COPY = "Watch till the end.\n@clipeand22\n#forge #shorts"


def _settings(tmp_path: Path, **kw) -> Settings:
    return Settings(
        api_base_url="http://localhost",
        api_token="test",
        worker_id="test-worker",
        working_directory=str(tmp_path / "data"),
        clip_storage_root=str(tmp_path / "clips"),
        instagram_access_token="tok-default",
        instagram_ig_user_id="111",
        tiktok_access_token="tok-tt-default",
        tiktok_privacy="SELF_ONLY",
        youtube_client_id="cid",
        youtube_client_secret="secret",
        youtube_refresh_token="rt",
        **kw,
    )


def _pending_clip(tmp_path: Path) -> Path:
    src = tmp_path / "clips" / str(CAMPAIGN_ID) / "pending_upload" / f"{CLIP_ID}.mp4"
    src.parent.mkdir(parents=True, exist_ok=True)
    src.write_bytes(b"fake-mp4")
    return src


def _job(*, platform: str = "tiktok", dry_run: bool = False, payload_extra: dict | None = None) -> Job:
    payload = {
        "clip_id": CLIP_ID,
        "campaign_id": CAMPAIGN_ID,
        "platform": platform,
        "title": "t",
        "caption": COPY,
        "hashtags": ["#forge", "#shorts"],
        "dry_run": dry_run,
    }
    payload.update(payload_extra or {})
    return Job(id=f"job-{CLIP_ID}-{platform}", type="publish", payload=payload)


@pytest.fixture
def fake_uploaders(monkeypatch):
    yt = MagicMock(return_value={"video_id": "VID1", "post_url": "https://youtu.be/VID1"})
    ig = MagicMock(return_value={"media_id": "M1", "post_url": "https://www.instagram.com/reel/abc/"})
    tt = MagicMock(return_value={"publish_id": "P1", "privacy_level": "SELF_ONLY", "post_url": "https://www.tiktok.com/@x"})
    for name, attr, fn in (
        ("app.services.youtube_upload", "upload_short", yt),
        ("app.services.instagram_upload", "upload_reel", ig),
        ("app.services.tiktok_upload", "upload_video", tt),
    ):
        mod = types.ModuleType(name)
        setattr(mod, attr, fn)
        monkeypatch.setitem(sys.modules, name, mod)
    return {"youtube": yt, "instagram": ig, "tiktok": tt}


# --- scheduling -----------------------------------------------------------

def test_scheduled_publish_refused_before_its_hour(tmp_path, fake_uploaders):
    _pending_clip(tmp_path)
    with pytest.raises(ValueError, match="scheduled_at .* not reached"):
        PublishJob(_settings(tmp_path), _job(payload_extra={"scheduled_at": FUTURE})).execute()
    assert fake_uploaders["tiktok"].call_count == 0  # nothing went out


def test_scheduled_publish_runs_once_hour_reached(tmp_path, fake_uploaders):
    _pending_clip(tmp_path)
    out = PublishJob(_settings(tmp_path), _job(payload_extra={"scheduled_at": PAST})).execute()
    assert out["dry_run"] is False
    assert fake_uploaders["tiktok"].call_count == 1


def test_naive_scheduled_at_treated_as_utc(tmp_path, fake_uploaders):
    _pending_clip(tmp_path)
    naive = (datetime.now(timezone.utc) + timedelta(minutes=10)).replace(tzinfo=None).isoformat()
    with pytest.raises(ValueError, match="not reached"):
        PublishJob(_settings(tmp_path), _job(payload_extra={"scheduled_at": naive})).execute()


def test_dry_run_ignores_schedule(tmp_path, fake_uploaders):
    _pending_clip(tmp_path)
    out = PublishJob(_settings(tmp_path), _job(dry_run=True, payload_extra={"scheduled_at": FUTURE})).execute()
    assert out["dry_run"] is True
    assert fake_uploaders["tiktok"].call_count == 0
    assert out["publications"][0]["status"] == "posted"  # fabricated dry URL


def test_invalid_scheduled_at_ignored_not_fatal(tmp_path, fake_uploaders):
    _pending_clip(tmp_path)
    out = PublishJob(_settings(tmp_path), _job(payload_extra={"scheduled_at": "not-a-date"})).execute()
    assert fake_uploaders["tiktok"].call_count == 1


# --- accounts ---------------------------------------------------------------

def test_default_account_uses_legacy_credentials(tmp_path, fake_uploaders):
    _pending_clip(tmp_path)
    PublishJob(
        _settings(tmp_path),
        _job(platform="instagram", payload_extra={"account": "default"}),
    ).execute()
    kwargs = fake_uploaders["instagram"].call_args.kwargs
    assert kwargs["access_token"] == "tok-default"
    assert kwargs["ig_user_id"] == "111"


def test_youtube_default_account_uses_settings_credentials(tmp_path, fake_uploaders):
    _pending_clip(tmp_path)
    out = PublishJob(
        _settings(tmp_path),
        _job(platform="youtube", payload_extra={"account": "default"}),
    ).execute()
    kwargs = fake_uploaders["youtube"].call_args.kwargs
    assert kwargs["client_id"] == "cid"
    assert kwargs["client_secret"] == "secret"
    assert kwargs["refresh_token"] == "rt"
    pub = out["publications"][0]
    assert pub["account"] == "default"
    assert pub["video_id"] == "VID1"


def test_youtube_omitted_account_matches_default(tmp_path, fake_uploaders):
    _pending_clip(tmp_path)
    out = PublishJob(_settings(tmp_path), _job(platform="youtube")).execute()
    kwargs = fake_uploaders["youtube"].call_args.kwargs
    assert kwargs["client_id"] == "cid"
    assert kwargs["client_secret"] == "secret"
    assert kwargs["refresh_token"] == "rt"
    assert out["publications"][0]["account"] == "default"


def test_youtube_yt2_reads_suffixed_env(tmp_path, fake_uploaders, monkeypatch):
    monkeypatch.setenv("YOUTUBE_CLIENT_ID_YT2", "cid-yt2")
    monkeypatch.setenv("YOUTUBE_CLIENT_SECRET_YT2", "secret-yt2")
    monkeypatch.setenv("YOUTUBE_REFRESH_TOKEN_YT2", "rt-yt2")
    _pending_clip(tmp_path)
    out = PublishJob(
        _settings(tmp_path),
        _job(platform="youtube", payload_extra={"account": "yt2"}),
    ).execute()
    kwargs = fake_uploaders["youtube"].call_args.kwargs
    assert kwargs["client_id"] == "cid-yt2"
    assert kwargs["client_secret"] == "secret-yt2"
    assert kwargs["refresh_token"] == "rt-yt2"
    assert out["publications"][0]["account"] == "yt2"


def test_youtube_unknown_account_does_not_use_default_credentials(tmp_path, fake_uploaders, monkeypatch):
    monkeypatch.delenv("YOUTUBE_CLIENT_ID_GHOST", raising=False)
    monkeypatch.delenv("YOUTUBE_CLIENT_SECRET_GHOST", raising=False)
    monkeypatch.delenv("YOUTUBE_REFRESH_TOKEN_GHOST", raising=False)
    _pending_clip(tmp_path)
    with pytest.raises(RuntimeError, match="YOUTUBE_CLIENT_ID_GHOST missing") as exc:
        PublishJob(
            _settings(tmp_path),
            _job(platform="youtube", payload_extra={"account": "ghost"}),
        ).execute()
    message = str(exc.value)
    assert "YOUTUBE_CLIENT_SECRET_GHOST missing" in message
    assert "YOUTUBE_REFRESH_TOKEN_GHOST missing" in message
    assert fake_uploaders["youtube"].call_count == 0


def test_youtube_partial_suffix_does_not_mix_with_default(tmp_path, fake_uploaders, monkeypatch):
    monkeypatch.setenv("YOUTUBE_CLIENT_ID_YT2", "cid-yt2")
    monkeypatch.delenv("YOUTUBE_CLIENT_SECRET_YT2", raising=False)
    monkeypatch.delenv("YOUTUBE_REFRESH_TOKEN_YT2", raising=False)
    _pending_clip(tmp_path)
    with pytest.raises(RuntimeError, match="YOUTUBE_CLIENT_SECRET_YT2 missing") as exc:
        PublishJob(
            _settings(tmp_path),
            _job(platform="youtube", payload_extra={"account": "yt2"}),
        ).execute()
    message = str(exc.value)
    assert "YOUTUBE_REFRESH_TOKEN_YT2 missing" in message
    assert "YOUTUBE_CLIENT_ID_YT2" not in message
    assert fake_uploaders["youtube"].call_count == 0


def test_account_key_picks_override_env(tmp_path, fake_uploaders, monkeypatch):
    monkeypatch.setenv("INSTAGRAM_ACCESS_TOKEN_IG2", "tok-ig2")
    monkeypatch.setenv("INSTAGRAM_IG_USER_ID_IG2", "222")
    monkeypatch.setenv("TIKTOK_ACCESS_TOKEN_STUDIO", "tok-tt-studio")
    _pending_clip(tmp_path)
    PublishJob(
        _settings(tmp_path),
        _job(platform="instagram", payload_extra={"account": "ig2", "accounts": ["default", "ig2"]}),
    ).execute()
    kwargs = fake_uploaders["instagram"].call_args.kwargs
    assert kwargs["access_token"] == "tok-ig2"
    assert kwargs["ig_user_id"] == "222"


def test_unknown_account_falls_back_to_default_credentials(tmp_path, fake_uploaders):
    _pending_clip(tmp_path)
    PublishJob(
        _settings(tmp_path),
        _job(platform="tiktok", payload_extra={"account": "ghost"}),
    ).execute()
    kwargs = fake_uploaders["tiktok"].call_args.kwargs
    assert kwargs["access_token"] == "tok-tt-default"


def test_account_env_helper_normalises_key(tmp_path):
    settings = _settings(tmp_path)
    assert _account_env(settings, "instagram_access_token", "default") == "tok-default"
    assert _account_env(settings, "instagram_access_token", "Ghost") == "tok-default"
    import os

    os.environ["INSTAGRAM_ACCESS_TOKEN_A_B"] = "x"
    try:
        assert _account_env(settings, "instagram_access_token", "a b") == "x"
    finally:
        os.environ.pop("INSTAGRAM_ACCESS_TOKEN_A_B", None)
    assert _account_env(settings, "instagram_access_token", "") == "tok-default"


# --- per-platform copy -------------------------------------------------------

def test_tiktok_gets_full_per_platform_copy(tmp_path, fake_uploaders):
    _pending_clip(tmp_path)
    PublishJob(_settings(tmp_path), _job(platform="tiktok", payload_extra={"scheduled_at": PAST})).execute()
    kwargs = fake_uploaders["tiktok"].call_args.kwargs
    # the whole caption (body + mentions + hashtags), not just the short title
    assert kwargs["title"] == COPY


def test_tiktok_falls_back_to_title_when_no_caption(tmp_path, fake_uploaders):
    _pending_clip(tmp_path)
    job = _job(platform="tiktok", payload_extra={"caption": "", "scheduled_at": PAST})
    PublishJob(_settings(tmp_path), job).execute()
    kwargs = fake_uploaders["tiktok"].call_args.kwargs
    assert kwargs["title"] == "t"


def test_account_recorded_in_publication_state(tmp_path, fake_uploaders):
    _pending_clip(tmp_path)
    PublishJob(
        _settings(tmp_path),
        _job(platform="instagram", payload_extra={"account": "ig2", "scheduled_at": PAST}),
    ).execute()
    state = tmp_path / "data" / "publish_state" / f"{CLIP_ID}__instagram.json"
    data = json.loads(state.read_text(encoding="utf-8"))
    assert data["publication"]["account"] == "ig2"
    assert fake_uploaders["instagram"].call_args.kwargs["access_token"] == "tok-default"