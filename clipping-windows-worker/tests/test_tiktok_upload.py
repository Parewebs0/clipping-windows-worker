"""TikTok inbox/direct upload and token refresh. httpx is mocked; no network."""
from __future__ import annotations

import json
import sys
import types
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock

import httpx
import pytest

from app.config import Settings
from app.jobs.publish import PublishJob
from app.models.job import Job
from app.services.tiktok_oauth import TikTokOAuthStateError, TikTokOAuthStore
from app.services.tiktok_upload import (
    build_post_url,
    iter_chunk_ranges,
    plan_upload_chunks,
    upload_video,
)

OLD_ACCESS = "act.OLDACCESSTOKEN"
NEW_ACCESS = "act.NEWACCESSTOKEN"
OLD_REFRESH = "rft.OLDREFRESHTOKEN"
NEW_REFRESH = "rft.NEWREFRESHTOKEN"
CLIENT_SECRET = "cs.CLIENTSECRETVALUE"
CLIENT_KEY = "ck.CLIENTKEYVALUE"
SECRETS = (OLD_ACCESS, NEW_ACCESS, OLD_REFRESH, NEW_REFRESH, CLIENT_SECRET, CLIENT_KEY)


def _video(tmp_path: Path, payload: bytes = b"fake-mp4-bytes") -> Path:
    path = tmp_path / "clip.mp4"
    path.write_bytes(payload)
    return path


def _ok(data: dict) -> httpx.Response:
    return httpx.Response(200, json={"data": data, "error": {"code": "ok", "message": "", "log_id": "l"}})


def _capture(capsys, caplog) -> str:
    captured = capsys.readouterr()
    return captured.out + captured.err + caplog.text


def _assert_no_secrets(text: str, *extra: str) -> None:
    for secret in (*SECRETS, *extra):
        assert secret not in text


def test_chunk_plan_keeps_small_files_whole_and_splits_past_64mb():
    assert plan_upload_chunks(100) == (100, 1)
    assert plan_upload_chunks(4_194_304) == (4_194_304, 1)
    assert plan_upload_chunks(50_000_123) == (50_000_123, 1)
    assert plan_upload_chunks(64_000_000) == (64_000_000, 1)

    size = 64_000_000 + 123
    chunk, count = plan_upload_chunks(size)
    assert count >= 2
    assert count <= 1000
    assert 5_000_000 <= chunk <= 64_000_000
    ranges = list(iter_chunk_ranges(size, chunk, count))
    assert ranges[0] == (0, chunk)
    assert sum(length for _, length in ranges) == size
    assert all(length == chunk for _, length in ranges[:-1])
    assert 0 < ranges[-1][1] <= 128 * 1024 * 1024


def test_post_url_uses_username_and_id_and_never_invents_a_handle():
    assert build_post_url("alice", "1234567890123456789") == "https://www.tiktok.com/@alice/video/1234567890123456789"
    assert build_post_url("@alice", None) == "https://www.tiktok.com/@alice"
    assert build_post_url("", "123") == ""
    assert "clipeand22" not in build_post_url("alice", "99")
    source = Path(__file__).resolve().parents[1].joinpath("app/services/tiktok_upload.py").read_text(encoding="utf-8")
    assert "clipeand22" not in source


def test_missing_mode_stays_inbox_and_does_not_call_creator_info(tmp_path):
    video = _video(tmp_path)
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(f"{request.method} {request.url.path}")
        if request.url.path.endswith("/inbox/video/init/"):
            body = json.loads(request.content)
            assert "post_info" not in body
            assert body["source_info"] == {
                "source": "FILE_UPLOAD",
                "video_size": video.stat().st_size,
                "chunk_size": video.stat().st_size,
                "total_chunk_count": 1,
            }
            return _ok({"publish_id": "p1", "upload_url": "https://upload.test/v"})
        if request.method == "PUT":
            size = video.stat().st_size
            assert request.headers["content-range"] == f"bytes 0-{size - 1}/{size}"
            return httpx.Response(201)
        if request.url.path.endswith("/status/fetch/"):
            return _ok({"status": "SEND_TO_USER_INBOX"})
        raise AssertionError(request.url.path)

    result = upload_video(
        file_path=video,
        title="Watch till the end.\n@clipeand22\n#forge",
        access_token="tok-static",
        privacy_level="SELF_ONLY",
        handle="studio",
        transport=httpx.MockTransport(handler),
        poll_interval=0,
    )
    assert seen == [
        "POST /v2/post/publish/inbox/video/init/",
        "PUT /v",
        "POST /v2/post/publish/status/fetch/",
    ]
    assert result["post_url"] == "https://www.tiktok.com/@studio"
    assert "clipeand22" not in result["post_url"]
    assert result["privacy_level"] == "SELF_ONLY"
    assert result["status"] == "SEND_TO_USER_INBOX"


def test_direct_calls_creator_info_init_upload_status_in_order(tmp_path):
    video = _video(tmp_path)
    seen: list[str] = []
    init_body: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.url.path)
        if request.url.path.endswith("/creator_info/query/"):
            return _ok({
                "creator_username": "alice",
                "privacy_level_options": ["PUBLIC_TO_EVERYONE", "SELF_ONLY"],
                "duet_disabled": True,
                "stitch_disabled": False,
                "comment_disabled": False,
                "max_video_post_duration_sec": 300,
            })
        if request.url.path.endswith("/video/init/"):
            init_body.update(json.loads(request.content))
            return _ok({"publish_id": "pub1", "upload_url": "https://upload.test/v"})
        if request.method == "PUT":
            return httpx.Response(201)
        if request.url.path.endswith("/status/fetch/"):
            return _ok({
                "status": "PUBLISH_COMPLETE",
                "publicaly_available_post_id": ["7123456789012345678"],
            })
        raise AssertionError(request.url.path)

    result = upload_video(
        file_path=video,
        title="hola @clipeand22",
        access_token="tok-static",
        privacy_level="PUBLIC_TO_EVERYONE",
        mode="direct",
        paid_promotion=True,
        handle="fallback",
        duration_sec=12,
        transport=httpx.MockTransport(handler),
        poll_interval=0,
    )
    assert seen == [
        "/v2/post/publish/creator_info/query/",
        "/v2/post/publish/video/init/",
        "/v",
        "/v2/post/publish/status/fetch/",
    ]
    post = init_body["post_info"]
    assert post["title"] == "hola @clipeand22"
    assert post["privacy_level"] == "PUBLIC_TO_EVERYONE"
    assert post["disable_comment"] is False
    assert post["disable_duet"] is True
    assert post["disable_stitch"] is False
    assert post["brand_content_toggle"] is True
    assert result["post_url"] == "https://www.tiktok.com/@alice/video/7123456789012345678"
    assert "clipeand22" not in result["post_url"]
    assert result["post_id"] == "7123456789012345678"
    assert result["publicaly_available_post_id"] == ["7123456789012345678"]


def test_direct_without_post_id_uses_the_profile(tmp_path):
    video = _video(tmp_path)

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/creator_info/query/"):
            return _ok({
                "creator_username": "",
                "privacy_level_options": ["SELF_ONLY"],
                "max_video_post_duration_sec": 600,
            })
        if request.url.path.endswith("/video/init/"):
            return _ok({"publish_id": "pub1", "upload_url": "https://upload.test/v"})
        if request.method == "PUT":
            return httpx.Response(201)
        return _ok({"status": "PUBLISH_COMPLETE"})

    result = upload_video(
        file_path=video,
        title="t",
        access_token="tok-static",
        mode="direct",
        handle="@studio",
        duration_sec=1,
        transport=httpx.MockTransport(handler),
        poll_interval=0,
    )
    assert result["post_url"] == "https://www.tiktok.com/@studio"
    assert "post_id" not in result


def test_invalid_privacy_errors_before_upload(tmp_path):
    video = _video(tmp_path)
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.url.path)
        return _ok({"creator_username": "alice", "privacy_level_options": ["SELF_ONLY"], "max_video_post_duration_sec": 300})

    with pytest.raises(RuntimeError, match="privacy_level 'PUBLIC_TO_EVERYONE' is not in privacy_level_options"):
        upload_video(
            file_path=video,
            title="t",
            access_token="tok-static",
            privacy_level="PUBLIC_TO_EVERYONE",
            mode="direct",
            duration_sec=1,
            transport=httpx.MockTransport(handler),
            poll_interval=0,
        )
    assert seen == ["/v2/post/publish/creator_info/query/"]


def test_duration_over_max_errors_before_upload(tmp_path):
    video = _video(tmp_path)
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.method)
        return _ok({"privacy_level_options": ["SELF_ONLY"], "max_video_post_duration_sec": 300})

    with pytest.raises(RuntimeError, match="exceeds max_video_post_duration_sec 300"):
        upload_video(
            file_path=video,
            title="t",
            access_token="tok-static",
            mode="direct",
            duration_sec=301,
            transport=httpx.MockTransport(handler),
            poll_interval=0,
        )
    assert seen == ["POST"]


def test_paid_promotion_false_sends_brand_content_toggle_false(tmp_path):
    video = _video(tmp_path)
    init_body: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/creator_info/query/"):
            return _ok({"creator_username": "a", "privacy_level_options": ["SELF_ONLY"], "max_video_post_duration_sec": 300})
        if request.url.path.endswith("/video/init/"):
            init_body.update(json.loads(request.content))
            return _ok({"publish_id": "p", "upload_url": "https://upload.test/v"})
        if request.method == "PUT":
            return httpx.Response(201)
        return _ok({"status": "PUBLISH_COMPLETE", "publicaly_available_post_id": ["42"]})

    upload_video(
        file_path=video,
        title="t",
        access_token="tok-static",
        mode="direct",
        paid_promotion=False,
        duration_sec=1,
        transport=httpx.MockTransport(handler),
        poll_interval=0,
    )
    assert init_body["post_info"]["brand_content_toggle"] is False


def test_large_upload_follows_the_chunk_plan(tmp_path, monkeypatch):
    video = _video(tmp_path, b"abcdefgh")
    monkeypatch.setattr("app.services.tiktok_upload.plan_upload_chunks", lambda size: (3, 2))
    ranges: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/inbox/video/init/"):
            body = json.loads(request.content)
            assert body["source_info"]["chunk_size"] == 3
            assert body["source_info"]["total_chunk_count"] == 2
            return _ok({"publish_id": "p", "upload_url": "https://upload.test/v"})
        if request.method == "PUT":
            ranges.append(request.headers["content-range"])
            assert request.content in {b"abc", b"defgh"}
            return httpx.Response(206 if request.content == b"abc" else 201)
        return _ok({"status": "SEND_TO_USER_INBOX"})

    upload_video(
        file_path=video,
        title="t",
        access_token="tok-static",
        handle="studio",
        transport=httpx.MockTransport(handler),
        poll_interval=0,
    )
    assert ranges == ["bytes 0-2/8", "bytes 3-7/8"]


def test_expiring_token_refreshes_once_and_persists_without_logging_secrets(tmp_path, capsys, caplog):
    video = _video(tmp_path)
    oauth_dir = tmp_path / "oauth_state"
    TikTokOAuthStore(oauth_dir).save("studio", {
        "access_token": OLD_ACCESS,
        "refresh_token": OLD_REFRESH,
        "access_expires_at": (datetime.now(timezone.utc) + timedelta(minutes=2)).isoformat(),
    })
    refresh_calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v2/oauth/token/":
            refresh_calls["n"] += 1
            body = request.content.decode()
            assert "grant_type=refresh_token" in body
            assert OLD_REFRESH in body
            assert CLIENT_SECRET in body
            assert "authorization" not in {k.lower() for k in request.headers}
            return httpx.Response(200, json={
                "access_token": NEW_ACCESS,
                "expires_in": 86400,
                "refresh_token": NEW_REFRESH,
                "refresh_expires_in": 31536000,
                "open_id": "oid",
                "scope": "video.publish",
                "token_type": "Bearer",
            })
        if request.url.host == "open.tiktokapis.com":
            assert request.headers["authorization"] == f"Bearer {NEW_ACCESS}"
        if request.url.path.endswith("/inbox/video/init/"):
            return _ok({"publish_id": "p", "upload_url": "https://upload.test/v?upload_token=UPLOADTOKENSECRET"})
        if request.method == "PUT":
            return httpx.Response(201)
        return _ok({"status": "SEND_TO_USER_INBOX"})

    transport = httpx.MockTransport(handler)
    common = dict(
        file_path=video,
        title="t",
        access_token=OLD_ACCESS,
        account="Studio",
        client_key=CLIENT_KEY,
        client_secret=CLIENT_SECRET,
        refresh_token="rft.ENVREFRESHSHOULDNOTWIN",
        oauth_state_dir=oauth_dir,
        handle="studio",
        transport=transport,
        poll_interval=0,
    )
    first = upload_video(**common)
    second = upload_video(**common)
    assert first["post_url"] == second["post_url"] == "https://www.tiktok.com/@studio"
    assert refresh_calls["n"] == 1

    path = oauth_dir / "tiktok_studio.json"
    assert path.exists()
    assert not (oauth_dir / "tiktok_default.json").exists()
    assert (path.stat().st_mode & 0o777) == 0o600
    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved["access_token"] == NEW_ACCESS
    assert saved["refresh_token"] == NEW_REFRESH
    assert saved["access_expires_at"]
    assert saved["refresh_expires_at"]

    blob = _capture(capsys, caplog)
    _assert_no_secrets(blob, "rft.ENVREFRESHSHOULDNOTWIN", "UPLOADTOKENSECRET")
    assert "tiktok access token refreshed" in blob


def test_invalid_refresh_errors_without_secrets(tmp_path, capsys, caplog):
    video = _video(tmp_path)
    oauth_dir = tmp_path / "oauth_state"
    TikTokOAuthStore(oauth_dir).save("default", {
        "access_token": OLD_ACCESS,
        "refresh_token": OLD_REFRESH,
        "access_expires_at": (datetime.now(timezone.utc) - timedelta(minutes=1)).isoformat(),
    })
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.url.path)
        return httpx.Response(400, json={
            "error": "invalid_grant",
            "error_description": f"Refresh token {OLD_REFRESH} rejected for secret {CLIENT_SECRET}",
            "log_id": "log-1",
        })

    with pytest.raises(RuntimeError, match="tiktok token refresh failed") as raised:
        upload_video(
            file_path=video,
            title="t",
            access_token=OLD_ACCESS,
            client_key=CLIENT_KEY,
            client_secret=CLIENT_SECRET,
            refresh_token=OLD_REFRESH,
            oauth_state_dir=oauth_dir,
            transport=httpx.MockTransport(handler),
            poll_interval=0,
        )
    message = str(raised.value)
    assert message == "tiktok token refresh failed: invalid_grant"
    assert raised.value.__cause__ is None
    _assert_no_secrets(message)
    _assert_no_secrets(_capture(capsys, caplog))
    assert seen == ["/v2/oauth/token/"]
    saved = json.loads((oauth_dir / "tiktok_default.json").read_text(encoding="utf-8"))
    assert saved["access_token"] == OLD_ACCESS
    assert saved["refresh_token"] == OLD_REFRESH


def test_refresh_without_a_static_access_token(tmp_path):
    video = _video(tmp_path)
    oauth_dir = tmp_path / "oauth_state"

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v2/oauth/token/":
            return httpx.Response(200, json={
                "access_token": NEW_ACCESS,
                "expires_in": 86400,
                "refresh_token": NEW_REFRESH,
                "token_type": "Bearer",
            })
        if request.url.host == "open.tiktokapis.com":
            assert request.headers["authorization"] == f"Bearer {NEW_ACCESS}"
        if request.url.path.endswith("/inbox/video/init/"):
            return _ok({"publish_id": "p", "upload_url": "https://upload.test/v"})
        if request.method == "PUT":
            return httpx.Response(201)
        return _ok({"status": "SEND_TO_USER_INBOX"})

    upload_video(
        file_path=video,
        title="t",
        access_token="",
        client_key=CLIENT_KEY,
        client_secret=CLIENT_SECRET,
        refresh_token=OLD_REFRESH,
        oauth_state_dir=oauth_dir,
        handle="studio",
        transport=httpx.MockTransport(handler),
        poll_interval=0,
    )
    saved = json.loads((oauth_dir / "tiktok_default.json").read_text(encoding="utf-8"))
    assert saved["access_token"] == NEW_ACCESS
    assert saved["refresh_token"] == NEW_REFRESH


def test_failed_put_does_not_echo_the_upload_token(tmp_path):
    video = _video(tmp_path)
    upload_token = "UPLOADTOKENSECRET"

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/inbox/video/init/"):
            return _ok({"publish_id": "p", "upload_url": f"https://upload.test/v?upload_token={upload_token}"})
        if request.method == "PUT":
            return httpx.Response(403, text=f"forbidden url token {upload_token}")
        return _ok({"status": "SEND_TO_USER_INBOX"})

    with pytest.raises(RuntimeError, match="tiktok upload 403") as raised:
        upload_video(
            file_path=video,
            title="t",
            access_token="tok-static",
            transport=httpx.MockTransport(handler),
            poll_interval=0,
        )
    assert upload_token not in str(raised.value)
    assert "[redacted]" in str(raised.value)


def test_static_access_token_is_not_refreshed(tmp_path):
    video = _video(tmp_path)
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.url.path)
        if request.url.host == "open.tiktokapis.com":
            assert request.headers["authorization"] == "Bearer tok-static"
        if request.url.path.endswith("/inbox/video/init/"):
            return _ok({"publish_id": "p", "upload_url": "https://upload.test/v"})
        if request.method == "PUT":
            return httpx.Response(201)
        return _ok({"status": "SEND_TO_USER_INBOX", "uploaded_bytes": video.stat().st_size})

    upload_video(
        file_path=video,
        title="t",
        access_token="tok-static",
        client_key=CLIENT_KEY,
        client_secret=CLIENT_SECRET,
        refresh_token=OLD_REFRESH,
        oauth_state_dir=tmp_path / "oauth_state",
        transport=httpx.MockTransport(handler),
        poll_interval=0,
    )
    assert "/v2/oauth/token/" not in seen
    assert not (tmp_path / "oauth_state").exists()


def test_api_error_body_does_not_leak_the_access_token(tmp_path, capsys, caplog):
    video = _video(tmp_path)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, text=f"unauthorized {OLD_ACCESS}")

    with pytest.raises(RuntimeError, match="tiktok inbox init 401") as raised:
        upload_video(
            file_path=video,
            title="t",
            access_token=OLD_ACCESS,
            transport=httpx.MockTransport(handler),
            poll_interval=0,
        )
    assert OLD_ACCESS not in str(raised.value)
    assert "[redacted]" in str(raised.value)
    _assert_no_secrets(_capture(capsys, caplog))


def test_corrupt_oauth_state_does_not_leak_the_file(tmp_path):
    oauth_dir = tmp_path / "oauth_state"
    oauth_dir.mkdir()
    (oauth_dir / "tiktok_default.json").write_text('{"access_token": "' + OLD_ACCESS, encoding="utf-8")
    with pytest.raises(TikTokOAuthStateError, match="unreadable tiktok oauth state") as raised:
        upload_video(
            file_path=_video(tmp_path),
            title="t",
            access_token="tok-static",
            oauth_state_dir=oauth_dir,
            transport=httpx.MockTransport(lambda request: httpx.Response(500)),
            poll_interval=0,
        )
    assert OLD_ACCESS not in str(raised.value)
    assert raised.value.__cause__ is None


def test_title_is_capped_at_2200_utf16_units(tmp_path):
    video = _video(tmp_path)
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/creator_info/query/"):
            return _ok({"creator_username": "a", "privacy_level_options": ["SELF_ONLY"], "max_video_post_duration_sec": 300})
        if request.url.path.endswith("/video/init/"):
            captured.update(json.loads(request.content))
            return _ok({"publish_id": "p", "upload_url": "https://upload.test/v"})
        if request.method == "PUT":
            return httpx.Response(201)
        return _ok({"status": "PUBLISH_COMPLETE"})

    upload_video(
        file_path=video,
        title="á" * 2201,
        access_token="tok-static",
        mode="direct",
        duration_sec=1,
        transport=httpx.MockTransport(handler),
        poll_interval=0,
    )
    assert len(captured["post_info"]["title"].encode("utf-16-le")) // 2 == 2200


def test_publish_job_passes_direct_credentials_and_keeps_idempotency(tmp_path, monkeypatch):
    clip_id = "33333333-2222-3333-4444-555555555555"
    src = tmp_path / "clips" / "9" / "pending_upload" / f"{clip_id}.mp4"
    src.parent.mkdir(parents=True)
    src.write_bytes(b"fake-mp4")
    monkeypatch.setenv("TIKTOK_CLIENT_KEY_STUDIO", CLIENT_KEY)
    monkeypatch.setenv("TIKTOK_CLIENT_SECRET_STUDIO", CLIENT_SECRET)
    monkeypatch.setenv("TIKTOK_REFRESH_TOKEN_STUDIO", OLD_REFRESH)
    settings = Settings(
        api_base_url="http://localhost",
        api_token="test",
        worker_id="test-worker",
        working_directory=str(tmp_path / "data"),
        clip_storage_root=str(tmp_path / "clips"),
        tiktok_access_token="tok-default",
        tiktok_mode="direct",
        tiktok_privacy="PUBLIC_TO_EVERYONE",
    )
    tt = MagicMock(return_value={
        "publish_id": "P1",
        "privacy_level": "PUBLIC_TO_EVERYONE",
        "post_url": "https://www.tiktok.com/@alice/video/99",
        "post_id": "99",
        "publicaly_available_post_id": ["99"],
    })
    mod = types.ModuleType("app.services.tiktok_upload")
    mod.upload_video = tt
    monkeypatch.setitem(sys.modules, "app.services.tiktok_upload", mod)
    job = Job(id="job-1", type="publish", payload={
        "clip_id": clip_id,
        "campaign_id": 9,
        "platform": "tiktok",
        "title": "t",
        "caption": "copy @clipeand22",
        "dry_run": False,
        "account": "studio",
        "handle": "@alice",
        "paid_promotion": True,
    })
    out = PublishJob(settings, job).execute()
    kwargs = tt.call_args.kwargs
    assert kwargs["mode"] == "direct"
    assert kwargs["paid_promotion"] is True
    assert kwargs["handle"] == "@alice"
    assert kwargs["account"] == "studio"
    assert kwargs["client_key"] == CLIENT_KEY
    assert kwargs["client_secret"] == CLIENT_SECRET
    assert kwargs["refresh_token"] == OLD_REFRESH
    assert kwargs["access_token"] == "tok-default"
    assert kwargs["privacy_level"] == "PUBLIC_TO_EVERYONE"
    assert kwargs["oauth_state_dir"] == tmp_path / "data" / "oauth_state"
    assert kwargs["title"] == "copy @clipeand22"
    assert out["publications"][0]["post_url"] == "https://www.tiktok.com/@alice/video/99"
    assert out["publications"][0]["post_id"] == "99"
    assert "clipeand22" not in out["publications"][0]["post_url"]

    again = PublishJob(settings, job).execute()
    assert tt.call_count == 1
    assert again["publications"][0]["reused_from_state"] is True
    assert again["publications"][0]["post_url"] == out["publications"][0]["post_url"]
