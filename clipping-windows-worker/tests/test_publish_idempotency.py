"""Publish idempotency: a retry after a successful upload must not re-publish.

All uploaders are replaced by fake modules in ``sys.modules`` → no network,
no Google/Meta/TikTok SDKs needed.
"""
from __future__ import annotations

import json
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from app.config import Settings
from app.jobs.publish import PublishJob
from app.models.job import Job
from app.services.job_runner import JobRunner
from app.services.publish_state import PublishStateError, PublishStateStore

CLIP_ID = "11111111-2222-3333-4444-555555555555"
CAMPAIGN_ID = 7


def _settings(tmp_path: Path, **kw) -> Settings:
    return Settings(
        api_base_url="http://localhost",
        api_token="test",
        worker_id="test-worker",
        working_directory=str(tmp_path / "data"),
        clip_storage_root=str(tmp_path / "clips"),
        youtube_client_id="cid",
        youtube_client_secret="secret",
        youtube_refresh_token="rt",
        instagram_access_token="ig",
        instagram_ig_user_id="123",
        tiktok_access_token="tt",
        **kw,
    )


def _pending_clip(tmp_path: Path) -> Path:
    src = tmp_path / "clips" / str(CAMPAIGN_ID) / "pending_upload" / f"{CLIP_ID}.mp4"
    src.parent.mkdir(parents=True, exist_ok=True)
    src.write_bytes(b"fake-mp4")
    return src


def _job(platform: str = "youtube", job_id: str = "job-1", dry_run: bool = False) -> Job:
    return Job(
        id=job_id,
        type="publish",
        payload={
            "clip_id": CLIP_ID,
            "campaign_id": CAMPAIGN_ID,
            "platform": platform,
            "title": "t",
            "caption": "c",
            "dry_run": dry_run,
        },
    )


@pytest.fixture
def fake_uploaders(monkeypatch):
    yt = MagicMock(return_value={"video_id": "VID1", "post_url": "https://www.youtube.com/shorts/VID1"})
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


@pytest.mark.parametrize("platform", ["youtube", "instagram", "tiktok"])
def test_retry_after_success_does_not_republish(tmp_path, fake_uploaders, platform):
    settings = _settings(tmp_path)
    _pending_clip(tmp_path)

    first = PublishJob(settings, _job(platform)).execute()
    assert fake_uploaders[platform].call_count == 1
    url = first["publications"][0]["post_url"]

    # Same job retried by the API (e.g. reporting the result failed).
    second = PublishJob(settings, _job(platform)).execute()
    assert fake_uploaders[platform].call_count == 1, "must not upload twice"
    pub = second["publications"][0]
    assert pub["post_url"] == url
    assert pub["status"] == "posted"
    assert pub["reused_from_state"] is True
    assert second["dry_run"] is False
    assert second["source_moved"] is True
    assert second["final_path_worker"] == first["final_path_worker"]


def test_state_is_written_before_returning(tmp_path, fake_uploaders):
    settings = _settings(tmp_path)
    _pending_clip(tmp_path)
    PublishJob(settings, _job("youtube", job_id="job-42")).execute()

    path = tmp_path / "data" / "publish_state" / f"{CLIP_ID}__youtube.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["job_id"] == "job-42"
    assert data["publication"]["video_id"] == "VID1"
    assert data["publication"]["post_url"] == "https://www.youtube.com/shorts/VID1"
    assert not list(path.parent.glob("*.tmp"))


def test_new_job_same_clip_platform_is_also_skipped(tmp_path, fake_uploaders):
    settings = _settings(tmp_path)
    _pending_clip(tmp_path)
    PublishJob(settings, _job("youtube", job_id="a")).execute()
    PublishJob(settings, _job("youtube", job_id="b")).execute()
    assert fake_uploaders["youtube"].call_count == 1


def test_other_platform_still_publishes(tmp_path, fake_uploaders):
    settings = _settings(tmp_path)
    _pending_clip(tmp_path)
    PublishJob(settings, _job("youtube")).execute()
    PublishJob(settings, _job("instagram", job_id="job-ig")).execute()
    assert fake_uploaders["youtube"].call_count == 1
    assert fake_uploaders["instagram"].call_count == 1


def test_failed_upload_is_not_recorded_and_retry_uploads(tmp_path, fake_uploaders):
    settings = _settings(tmp_path)
    _pending_clip(tmp_path)
    fake_uploaders["youtube"].side_effect = [RuntimeError("quota"), fake_uploaders["youtube"].return_value]

    with pytest.raises(RuntimeError):
        PublishJob(settings, _job("youtube")).execute()
    assert not (tmp_path / "data" / "publish_state" / f"{CLIP_ID}__youtube.json").exists()

    out = PublishJob(settings, _job("youtube")).execute()
    assert fake_uploaders["youtube"].call_count == 2
    assert "reused_from_state" not in out["publications"][0]


def test_dry_run_never_records_state(tmp_path, fake_uploaders):
    settings = _settings(tmp_path)
    _pending_clip(tmp_path)
    PublishJob(settings, _job("youtube", dry_run=True)).execute()
    assert not (tmp_path / "data" / "publish_state").exists()
    PublishJob(settings, _job("youtube")).execute()
    assert fake_uploaders["youtube"].call_count == 1


def test_state_write_failure_does_not_fail_job(tmp_path, fake_uploaders, monkeypatch):
    settings = _settings(tmp_path)
    _pending_clip(tmp_path)

    def boom(*a, **k):
        raise OSError("disk full")

    monkeypatch.setattr(PublishStateStore, "record", boom)
    out = PublishJob(settings, _job("youtube")).execute()
    assert out["publications"][0]["post_url"].endswith("VID1")


def test_corrupt_state_refuses_to_upload(tmp_path, fake_uploaders):
    settings = _settings(tmp_path)
    _pending_clip(tmp_path)
    state = tmp_path / "data" / "publish_state" / f"{CLIP_ID}__youtube.json"
    state.parent.mkdir(parents=True)
    state.write_text("{not json", encoding="utf-8")
    with pytest.raises(PublishStateError):
        PublishJob(settings, _job("youtube")).execute()
    assert fake_uploaders["youtube"].call_count == 0


def test_publish_state_dir_setting(tmp_path, fake_uploaders):
    custom = tmp_path / "state"
    settings = _settings(tmp_path, publish_state_dir=str(custom))
    _pending_clip(tmp_path)
    PublishJob(settings, _job("tiktok")).execute()
    assert (custom / f"{CLIP_ID}__tiktok.json").exists()


def test_job_runner_retry_after_result_upload_failure(tmp_path, fake_uploaders):
    """End-to-end through JobRunner: upload ok, reporting fails, retry reuses."""
    settings = _settings(tmp_path)
    _pending_clip(tmp_path)
    api = MagicMock()
    api.start_job.return_value = True
    api.upload_result.side_effect = [RuntimeError("API down"), True]
    manager = MagicMock()
    manager.create_handler.side_effect = lambda job: PublishJob(settings, job)
    runner = JobRunner(settings=settings, api_client=api, job_manager=manager)

    with pytest.raises(RuntimeError, match="API down"):
        runner.run(_job("youtube"))
    api.fail_job.assert_called_once()

    result = runner.run(_job("youtube"))
    assert fake_uploaders["youtube"].call_count == 1
    assert result["publications"][0]["post_url"] == "https://www.youtube.com/shorts/VID1"
    assert api.upload_result.call_count == 2


def test_store_path_is_sanitised(tmp_path):
    store = PublishStateStore(tmp_path)
    p = store.path_for("../evil", "YouTube")
    assert p.parent == tmp_path
    assert p.name == ".._evil__youtube.json"
