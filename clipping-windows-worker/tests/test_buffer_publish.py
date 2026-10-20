"""Instagram/TikTok through Buffer. Buffer is faked with httpx.MockTransport."""
from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

from app.config import Settings
from app.jobs.publish import PublishJob
from app.models.job import Job
from app.services import buffer_publish
from app.services.buffer_publish import BufferPublishError, publish_video

CLIP_ID = "44444444-2222-3333-4444-555555555555"
BASE = "https://box.example.ts.net:8443/c"


class FakeBuffer:
    """Answers createPost once, then the queued post statuses in order."""

    def __init__(self, statuses, create=None):
        self.statuses = list(statuses)
        self.create = create or {"__typename": "PostActionSuccess", "post": {"id": "P1", "status": "sending"}}
        self.requests: list[dict] = []
        self.auth: list[str] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        self.requests.append(body)
        self.auth.append(request.headers.get("authorization", ""))
        if "createPost" in body["query"]:
            return httpx.Response(200, json={"data": {"createPost": self.create}})
        status, link = self.statuses.pop(0)
        post = {"id": body["variables"]["id"], "status": status, "externalLink": link, "error": None}
        if status == "error":
            post["error"] = {"message": "video too long"}
        return httpx.Response(200, json={"data": {"post": post}})

    @property
    def creates(self):
        return [r for r in self.requests if "createPost" in r["query"]]


def _clip(tmp_path: Path) -> Path:
    f = tmp_path / "clip.mp4"
    f.write_bytes(b"fake-mp4")
    return f


def _run(tmp_path, fake, platform="instagram", **kw):
    public = tmp_path / "public"
    public.mkdir(exist_ok=True)
    args = dict(
        file_path=_clip(tmp_path),
        text="Watch this #ad",
        platform=platform,
        channel_id="CH1",
        api_key="k-secret",
        public_dir=public,
        public_base_url=BASE,
        state_dir=tmp_path / "state",
        clip_id=CLIP_ID,
        poll_s=1,
        client=httpx.Client(transport=httpx.MockTransport(fake)),
        sleep=lambda s: None,
    )
    args.update(kw)
    return publish_video(**args), public


def test_instagram_reel_posted_now_and_public_copy_removed(tmp_path):
    fake = FakeBuffer([("sending", None), ("sent", "https://www.instagram.com/reel/ABC/")])
    out, public = _run(tmp_path, fake)
    assert out == {"post_url": "https://www.instagram.com/reel/ABC/", "buffer_post_id": "P1"}
    inp = fake.creates[0]["variables"]["input"]
    assert inp["mode"] == "shareNow" and inp["schedulingType"] == "automatic"
    assert inp["channelId"] == "CH1" and inp["text"] == "Watch this #ad"
    assert inp["metadata"] == {"instagram": {"type": "reel", "shouldShareToFeed": True}}
    url = inp["assets"][0]["video"]["url"]
    assert url.startswith(BASE + "/") and url.endswith(".mp4") and CLIP_ID not in url
    assert fake.auth[0] == "Bearer k-secret"
    assert list(public.iterdir()) == []  # public copy deleted once sent
    assert not (tmp_path / "state" / f"{CLIP_ID}__instagram.buffer.json").exists()


def test_tiktok_has_no_instagram_metadata(tmp_path):
    fake = FakeBuffer([("sent", "https://www.tiktok.com/@clipeand22/video/1")])
    out, _ = _run(tmp_path, fake, platform="tiktok")
    assert "metadata" not in fake.creates[0]["variables"]["input"]
    assert out["post_url"].endswith("/video/1")


def test_sent_without_link_keeps_polling(tmp_path):
    fake = FakeBuffer([("sent", None), ("sent", ""), ("sent", "https://www.tiktok.com/@x/video/2")])
    out, _ = _run(tmp_path, fake, platform="tiktok")
    assert out["post_url"].endswith("/video/2")


def test_create_error_removes_public_copy(tmp_path):
    fake = FakeBuffer([], create={"__typename": "InvalidInputError", "message": "Failed to fetch video"})
    with pytest.raises(BufferPublishError, match="Failed to fetch video"):
        _run(tmp_path, fake)
    assert list((tmp_path / "public").iterdir()) == []


def test_post_error_status_fails_and_cleans_up(tmp_path):
    fake = FakeBuffer([("sending", None), ("error", None)])
    with pytest.raises(BufferPublishError, match="error: video too long"):
        _run(tmp_path, fake)
    assert list((tmp_path / "public").iterdir()) == []
    assert not (tmp_path / "state" / f"{CLIP_ID}__instagram.buffer.json").exists()


def test_timeout_keeps_pending_and_retry_resumes_without_new_post(tmp_path, monkeypatch):
    fake = FakeBuffer([("sending", None)])
    with pytest.raises(BufferPublishError, match="not public after 0s"):
        _run(tmp_path, fake, timeout_s=0)
    pending = tmp_path / "state" / f"{CLIP_ID}__instagram.buffer.json"
    assert json.loads(pending.read_text())["post_id"] == "P1"
    assert len(list((tmp_path / "public").iterdir())) == 1  # Buffer may still fetch it

    retry = FakeBuffer([("sent", "https://www.instagram.com/reel/R/")])
    out, public = _run(tmp_path, retry)
    assert retry.creates == []  # resumed, never posted twice
    assert out["post_url"] == "https://www.instagram.com/reel/R/"
    assert list(public.iterdir()) == [] and not pending.exists()


def test_missing_config_names_the_variables(tmp_path):
    with pytest.raises(BufferPublishError, match="BUFFER_CHANNEL_TIKTOK, BUFFER_PUBLIC_BASE_URL missing"):
        _run(tmp_path, FakeBuffer([]), platform="tiktok", channel_id="", public_base_url="")


def test_graphql_errors_do_not_leak_key(tmp_path):
    def bad(request):
        return httpx.Response(200, json={"errors": [{"message": "Unknown argument"}]})
    with pytest.raises(BufferPublishError) as exc:
        _run(tmp_path, bad)
    assert "k-secret" not in str(exc.value)


# --- PublishJob routing -------------------------------------------------------

def _settings(tmp_path: Path, **kw) -> Settings:
    return Settings(
        api_base_url="http://localhost",
        api_token="test",
        worker_id="test-worker",
        working_directory=str(tmp_path / "data"),
        clip_storage_root=str(tmp_path / "clips"),
        **kw,
    )


def _job(platform: str, account: str = "default") -> Job:
    src = Path("/nonexistent")
    return Job(
        id=f"job-{platform}",
        type="publish",
        payload={"clip_id": CLIP_ID, "campaign_id": 9, "platform": platform, "title": "t",
                 "caption": "copy #ad", "dry_run": False, "account": account, "file_path": str(src)},
    )


def _pending(tmp_path: Path) -> None:
    src = tmp_path / "clips" / "9" / "pending_upload" / f"{CLIP_ID}.mp4"
    src.parent.mkdir(parents=True, exist_ok=True)
    src.write_bytes(b"fake-mp4")


@pytest.mark.parametrize("platform", ["instagram", "tiktok"])
def test_publish_job_uses_buffer_when_key_set(tmp_path, monkeypatch, platform):
    _pending(tmp_path)
    calls = []

    def fake_publish(**kw):
        calls.append(kw)
        return {"post_url": f"https://{platform}.com/p/1", "buffer_post_id": "P9"}

    monkeypatch.setattr(buffer_publish, "publish_video", fake_publish)
    monkeypatch.setenv(f"BUFFER_CHANNEL_{platform.upper()}_IG2", "CH-IG2")
    settings = _settings(
        tmp_path,
        buffer_api_key="k",
        buffer_channel_instagram="CH-IG",
        buffer_channel_tiktok="CH-TT",
        buffer_public_dir=str(tmp_path / "pub"),
        buffer_public_base_url=BASE,
    )
    out = PublishJob(settings, _job(platform)).execute()
    pub = out["publications"][0]
    assert pub["status"] == "posted" and pub["post_url"] == f"https://{platform}.com/p/1"
    assert pub["provider"] == "buffer" and pub["buffer_post_id"] == "P9"
    assert calls[0]["channel_id"] == ("CH-IG" if platform == "instagram" else "CH-TT")
    assert calls[0]["text"] == "copy #ad"

    # Per-account channel override, like the other providers.
    _pending(tmp_path)
    job = _job(platform, account="ig2")
    job.payload["clip_id"] = CLIP_ID.replace("4444", "5555", 1)
    src = tmp_path / "clips" / "9" / "pending_upload" / f"{job.payload['clip_id']}.mp4"
    src.write_bytes(b"x")
    PublishJob(settings, job).execute()
    assert calls[1]["channel_id"] == "CH-IG2"


def test_publish_job_without_key_keeps_direct_uploader(tmp_path, monkeypatch):
    import sys
    import types

    _pending(tmp_path)
    mod = types.ModuleType("app.services.instagram_upload")
    mod.upload_reel = lambda **kw: {"media_id": "M", "post_url": "https://www.instagram.com/reel/direct/"}
    monkeypatch.setitem(sys.modules, "app.services.instagram_upload", mod)
    monkeypatch.setattr(buffer_publish, "publish_video", lambda **kw: pytest.fail("buffer must not be used"))
    out = PublishJob(_settings(tmp_path, buffer_api_key=""), _job("instagram")).execute()
    assert out["publications"][0]["post_url"].endswith("/direct/")


def test_empty_public_dir_is_none(tmp_path):
    assert _settings(tmp_path, buffer_public_dir="").buffer_public_dir is None


# --- YouTube Shorts through Buffer for accounts without OAuth ------------------

def test_youtube_post_input_has_shorts_metadata():
    data = buffer_publish._post_input("youtube", "CH-YT", "copy #PBInvesting", "https://x/v.mp4",
                                      title="A" * 120, youtube_privacy="public", youtube_category_id="27")
    assert data["metadata"] == {"youtube": {"title": "A" * 100, "categoryId": "27", "privacy": "public",
                                            "madeForKids": False, "isAiGenerated": False}}
    assert buffer_publish._post_input("youtube", "c", "t", "u", youtube_privacy="weird")["metadata"]["youtube"]["privacy"] == "private"


def _yt_settings(tmp_path, **kw):
    return _settings(tmp_path, buffer_api_key="k-global", buffer_public_dir=str(tmp_path / "pub"),
                     buffer_public_base_url=BASE, youtube_privacy="public", **kw)


def test_dedicated_youtube_without_oauth_goes_through_buffer(tmp_path, monkeypatch):
    _pending(tmp_path)
    calls = []
    monkeypatch.setattr(buffer_publish, "publish_video",
                        lambda **kw: calls.append(kw) or {"post_url": "https://youtube.com/shorts/x", "buffer_post_id": "P1"})
    monkeypatch.setenv("BUFFER_API_KEY_PBINVESTING", "k-pbi")
    monkeypatch.setenv("BUFFER_CHANNEL_YOUTUBE_PBINVESTING", "CH-YT-PB")
    out = PublishJob(_yt_settings(tmp_path), _job("youtube", account="pbinvesting")).execute()
    assert out["publications"][0]["provider"] == "buffer"
    kw = calls[0]
    assert (kw["platform"], kw["channel_id"], kw["api_key"], kw["title"], kw["youtube_privacy"]) == \
        ("youtube", "CH-YT-PB", "k-pbi", "t", "public")


def test_youtube_with_oauth_keeps_the_worker_upload(tmp_path, monkeypatch):
    import sys
    import types

    _pending(tmp_path)
    mod = types.ModuleType("app.services.youtube_upload")
    mod.upload_short = lambda **kw: {"video_id": "V", "post_url": "https://youtube.com/shorts/oauth"}
    monkeypatch.setitem(sys.modules, "app.services.youtube_upload", mod)
    monkeypatch.setattr(buffer_publish, "publish_video", lambda **kw: pytest.fail("buffer must not be used"))
    monkeypatch.setenv("BUFFER_CHANNEL_YOUTUBE", "CH-YT")  # a default channel never diverts @moliclips
    settings = _yt_settings(tmp_path, youtube_client_id="i", youtube_client_secret="s", youtube_refresh_token="r")
    out = PublishJob(settings, _job("youtube")).execute()
    assert out["publications"][0]["post_url"].endswith("/oauth")


def test_dedicated_youtube_without_oauth_or_channel_fails(tmp_path, monkeypatch):
    _pending(tmp_path)
    monkeypatch.setattr(buffer_publish, "publish_video", lambda **kw: pytest.fail("buffer must not be used"))
    with pytest.raises(RuntimeError, match="YOUTUBE_CLIENT_ID_PBINVESTING missing"):
        PublishJob(_yt_settings(tmp_path), _job("youtube", account="pbinvesting")).execute()
