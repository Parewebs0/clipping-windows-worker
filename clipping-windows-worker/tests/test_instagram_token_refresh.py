"""Instagram long-lived token refresh (MOLI-77). httpx is faked; no network."""
from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

import httpx
import pytest

_RealClient = httpx.Client

from app.config import Settings
from app.jobs.publish import PublishJob, _account_env
from app.models.job import Job
from app.services.instagram_upload import (
    GRAPH_FACEBOOK,
    GRAPH_INSTAGRAM,
    _token_fingerprint,
    prepare_instagram_token,
    resolve_auth,
    resolve_graph_version,
    upload_reel,
)
from app.services.oauth_state import load_oauth_state, oauth_state_path, save_oauth_state

NOW = datetime(2026, 10, 18, 12, 0, tzinfo=timezone.utc)
OLD = "current-token-value"
NEW = "renewed-token-value"
APP_ID = "app-id-111"
APP_SECRET = "app-secret-222"
IG_USER = "17841400000000000"


class _Response:
    def __init__(self, status_code: int, payload: dict | None = None, text: str | None = None):
        self.status_code = status_code
        self._payload = {} if payload is None else payload
        body = text if text is not None else json.dumps(self._payload)
        self.text = body
        self.content = body.encode()

    def json(self):
        return self._payload


class FakeClient:
    def __init__(self, handler):
        self.handler = handler
        self.calls: list[dict] = []

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def close(self):
        return None

    def get(self, url, headers=None, params=None, **kwargs):
        call = {"method": "GET", "url": url, "headers": headers or {}, "params": dict(params or {})}
        self.calls.append(call)
        return self.handler("GET", url, call["headers"], call["params"], None)

    def post(self, url, headers=None, data=None, content=None, params=None, **kwargs):
        call = {
            "method": "POST",
            "url": url,
            "headers": headers or {},
            "params": dict(params or {}),
            "data": data,
        }
        self.calls.append(call)
        return self.handler("POST", url, call["headers"], call["params"], data)


@pytest.fixture(autouse=True)
def _no_real_http(monkeypatch):
    def boom(*args, **kwargs):
        raise AssertionError("real httpx.Client is disabled in these tests")

    monkeypatch.setattr("app.services.instagram_upload.httpx.Client", boom)
    monkeypatch.setattr("app.services.instagram_upload._missing_facebook_app_warned", False)
    monkeypatch.delenv("INSTAGRAM_AUTH", raising=False)
    monkeypatch.delenv("INSTAGRAM_GRAPH_VERSION", raising=False)
    monkeypatch.delenv("INSTAGRAM_APP_ID", raising=False)
    monkeypatch.delenv("INSTAGRAM_APP_SECRET", raising=False)
    monkeypatch.delenv("INSTAGRAM_ACCESS_TOKEN", raising=False)
    monkeypatch.delenv("INSTAGRAM_ACCESS_TOKEN_IG2", raising=False)


@pytest.fixture
def warnings(monkeypatch):
    found: list[tuple[str, dict]] = []

    class _Logger:
        def warning(self, msg, **kwargs):
            found.append((msg, kwargs))

        def info(self, *args, **kwargs):
            return None

        def error(self, *args, **kwargs):
            return None

        def debug(self, *args, **kwargs):
            return None

    monkeypatch.setattr("app.services.instagram_upload.logger", _Logger())
    return found


def _iso(delta: timedelta) -> str:
    return (NOW + delta).isoformat()


def _seed(tmp_path: Path, account: str, token: str, expires_at: str | None) -> Path:
    save_oauth_state(
        tmp_path,
        "instagram",
        account,
        {
            "provider": "instagram",
            "account": account,
            "access_token": token,
            "expires_at": expires_at,
        },
    )
    return oauth_state_path(tmp_path, "instagram", account)


def _patch(monkeypatch, handler) -> FakeClient:
    client = FakeClient(handler)
    monkeypatch.setattr("app.services.instagram_upload.httpx.Client", lambda *a, **k: client)
    return client


def _refresh_calls(client: FakeClient) -> list[dict]:
    return [
        c
        for c in client.calls
        if "/oauth/access_token" in c["url"] or "/refresh_access_token" in c["url"]
    ]


def _debug_calls(client: FakeClient) -> list[dict]:
    return [c for c in client.calls if "/debug_token" in c["url"]]


def _create_calls(client: FakeClient) -> list[dict]:
    return [c for c in client.calls if c["method"] == "POST" and c["url"].endswith("/media")]


def _rupload_calls(client: FakeClient) -> list[dict]:
    return [c for c in client.calls if "rupload.facebook.com" in c["url"]]


def _upload_handler(
    *,
    new_token: str = NEW,
    refresh_status: int = 200,
    debug_expires: int | None = None,
    refresh_body_has_token: bool = True,
    expires_in: int | None = 5_184_000,
    include_uri: bool = False,
    version: str = "v21.0",
    rupload_status: int = 200,
):
    def handler(method, url, headers, params, data):
        if "/debug_token" in url:
            data_obj: dict = {"is_valid": True}
            if debug_expires is not None:
                data_obj["expires_at"] = debug_expires
            return _Response(200, {"data": data_obj})
        if "/oauth/access_token" in url or url.endswith("/refresh_access_token"):
            if refresh_status >= 400:
                leaked = f"denied for {OLD} secret {APP_SECRET}"
                return _Response(refresh_status, {"error": {"message": "bad"}}, text=leaked)
            body: dict = {"token_type": "bearer"}
            if refresh_body_has_token:
                body["access_token"] = new_token
            if expires_in is not None:
                body["expires_in"] = expires_in
            return _Response(200, body)
        if method == "POST" and url.endswith("/media"):
            payload = {"id": "C1"}
            if include_uri:
                payload["uri"] = f"https://rupload.facebook.com/ig-api-upload/{version}/C1"
            return _Response(200, payload)
        if "rupload.facebook.com" in url:
            return _Response(rupload_status, {"success": rupload_status < 400}, text="rupload")
        if method == "GET" and str(params.get("fields", "")).startswith("status_code"):
            return _Response(200, {"status_code": "FINISHED"})
        if method == "POST" and url.endswith("/media_publish"):
            return _Response(200, {"id": "M1"})
        if method == "GET" and "permalink" in str(params.get("fields", "")):
            return _Response(200, {"permalink": "https://www.instagram.com/reel/M1/"})
        return _Response(500, {"error": "unhandled"}, text=f"unhandled {method} {url}")

    return handler


def _upload(tmp_path: Path, **kwargs):
    file_path = tmp_path / "clip.mp4"
    if not file_path.exists():
        file_path.write_bytes(b"fake-mp4")
    args = dict(
        file_path=file_path,
        caption="hola",
        access_token=OLD,
        ig_user_id=IG_USER,
        working_directory=tmp_path,
        now=NOW,
        app_id=APP_ID,
        app_secret=APP_SECRET,
        graph_version="v21.0",
        auth="facebook",
        account_key="default",
    )
    args.update(kwargs)
    return upload_reel(**args)


def _assert_no_secrets(blob: str) -> None:
    assert OLD not in blob
    assert NEW not in blob
    assert APP_SECRET not in blob
    assert "app-secret" not in blob


# --- oauth state store -------------------------------------------------------


def test_oauth_state_roundtrip_is_atomic_and_private(tmp_path: Path) -> None:
    old_umask = os.umask(0)
    try:
        path = save_oauth_state(
            tmp_path,
            "instagram",
            "ig2",
            {"access_token": OLD, "expires_at": _iso(timedelta(days=30))},
        )
    finally:
        os.umask(old_umask)
    assert path == tmp_path / "oauth_state" / "instagram_ig2.json"
    assert (path.stat().st_mode & 0o777) == 0o600
    assert load_oauth_state(tmp_path, "instagram", "ig2")["access_token"] == OLD
    assert list(path.parent.glob("*.tmp")) == []
    tiktok = oauth_state_path(tmp_path, "TikTok", "Studio")
    assert tiktok.name == "tiktok_studio.json"
    escaped = oauth_state_path(tmp_path, "instagram", "../ig2")
    assert escaped.parent == tmp_path / "oauth_state"
    assert "/" not in escaped.name and "\\" not in escaped.name


def test_oauth_state_corrupt_file_is_missing(tmp_path: Path) -> None:
    path = oauth_state_path(tmp_path, "instagram", "default")
    path.parent.mkdir(parents=True)
    path.write_text("{not-json", encoding="utf-8")
    assert load_oauth_state(tmp_path, "instagram", "default") is None


def test_auth_and_version_resolution(monkeypatch) -> None:
    assert resolve_auth(None) == "facebook"
    assert resolve_auth("Instagram") == "instagram"
    assert resolve_auth("nope") == "facebook"
    assert resolve_graph_version(None) == "v21.0"
    assert resolve_graph_version("v22.0") == "v22.0"
    assert resolve_graph_version("21.0") == "v21.0"
    monkeypatch.setenv("INSTAGRAM_AUTH", "instagram")
    monkeypatch.setenv("INSTAGRAM_GRAPH_VERSION", "v19.0")
    assert resolve_auth(None) == "instagram"
    assert resolve_graph_version(None) == "v19.0"
    assert resolve_auth("facebook") == "facebook"
    assert resolve_graph_version("v22.0") == "v22.0"


# --- refresh decisions -------------------------------------------------------


def test_token_with_more_than_7_days_is_not_refreshed(tmp_path: Path, warnings) -> None:
    _seed(tmp_path, "default", "persisted-token", _iso(timedelta(days=30)))
    client = FakeClient(_upload_handler())
    token = prepare_instagram_token(
        access_token=OLD,
        working_directory=tmp_path,
        now=NOW,
        app_id=APP_ID,
        app_secret=APP_SECRET,
        client=client,
    )
    assert token == "persisted-token"
    assert _refresh_calls(client) == []
    assert _debug_calls(client) == []
    assert warnings == []


def test_token_with_exactly_7_days_is_not_refreshed(tmp_path: Path) -> None:
    _seed(tmp_path, "default", OLD, _iso(timedelta(days=7)))
    client = FakeClient(_upload_handler())
    token = prepare_instagram_token(
        access_token=OLD,
        working_directory=tmp_path,
        now=NOW,
        app_id=APP_ID,
        app_secret=APP_SECRET,
        client=client,
    )
    assert token == OLD
    assert client.calls == []


def test_token_under_7_days_refreshes_persists_and_upload_uses_new(tmp_path, monkeypatch, warnings) -> None:
    _seed(tmp_path, "default", OLD, _iso(timedelta(days=7) - timedelta(seconds=1)))
    client = _patch(monkeypatch, _upload_handler())
    result = _upload(tmp_path)
    assert result["media_id"] == "M1"
    assert len(_refresh_calls(client)) == 1
    refresh = _refresh_calls(client)[0]
    assert refresh["url"] == f"{GRAPH_FACEBOOK}/v21.0/oauth/access_token"
    assert refresh["params"]["grant_type"] == "fb_exchange_token"
    assert refresh["params"]["client_id"] == APP_ID
    assert refresh["params"]["client_secret"] == APP_SECRET
    assert refresh["params"]["fb_exchange_token"] == OLD
    assert _debug_calls(client) == []
    create = _create_calls(client)[0]
    assert create["headers"]["Authorization"] == f"Bearer {NEW}"
    assert create["url"] == f"{GRAPH_FACEBOOK}/v21.0/{IG_USER}/media"
    assert _rupload_calls(client)[0]["headers"]["Authorization"] == f"OAuth {NEW}"
    stored = load_oauth_state(tmp_path, "instagram", "default")
    assert stored["access_token"] == NEW
    assert datetime.fromisoformat(stored["expires_at"]) - NOW == timedelta(seconds=5_184_000)
    path = oauth_state_path(tmp_path, "instagram", "default")
    assert path.name == "instagram_default.json"
    assert (path.stat().st_mode & 0o777) == 0o600
    assert APP_SECRET not in path.read_text(encoding="utf-8")
    assert stored["seed"] == _token_fingerprint(OLD)
    assert warnings == []
    again = prepare_instagram_token(
        access_token=OLD,
        working_directory=tmp_path,
        now=NOW,
        app_id=APP_ID,
        app_secret=APP_SECRET,
        client=client,
    )
    assert again == NEW
    assert len(_refresh_calls(client)) == 1


def test_unknown_expiry_uses_debug_token_and_skips_refresh_when_fresh(tmp_path, monkeypatch) -> None:
    far = int((NOW + timedelta(days=40)).timestamp())
    client = _patch(monkeypatch, _upload_handler(debug_expires=far))
    _upload(tmp_path)
    assert len(_debug_calls(client)) == 1
    debug = _debug_calls(client)[0]
    assert debug["url"] == f"{GRAPH_FACEBOOK}/v21.0/debug_token"
    assert debug["params"]["input_token"] == OLD
    assert debug["params"]["access_token"] == f"{APP_ID}|{APP_SECRET}"
    assert _refresh_calls(client) == []
    assert _create_calls(client)[0]["headers"]["Authorization"] == f"Bearer {OLD}"
    stored = load_oauth_state(tmp_path, "instagram", "default")
    assert stored["access_token"] == OLD
    assert not _needs_refresh_iso(stored["expires_at"])


def _needs_refresh_iso(value: str) -> bool:
    return datetime.fromisoformat(value) - NOW < timedelta(days=7)


def test_debug_token_zero_means_no_expiry_and_no_refresh(tmp_path, monkeypatch) -> None:
    client = _patch(monkeypatch, _upload_handler(debug_expires=0))
    _upload(tmp_path)
    assert _refresh_calls(client) == []
    assert _create_calls(client)[0]["headers"]["Authorization"] == f"Bearer {OLD}"


def test_refresh_failure_uploads_current_token_and_warns_without_secrets(tmp_path, monkeypatch, warnings) -> None:
    _seed(tmp_path, "default", OLD, _iso(timedelta(days=2)))
    before = oauth_state_path(tmp_path, "instagram", "default").read_text(encoding="utf-8")
    client = _patch(monkeypatch, _upload_handler(refresh_status=400))
    result = _upload(tmp_path)
    assert result["media_id"] == "M1"
    assert _create_calls(client)[0]["headers"]["Authorization"] == f"Bearer {OLD}"
    assert _rupload_calls(client)[0]["headers"]["Authorization"] == f"OAuth {OLD}"
    assert oauth_state_path(tmp_path, "instagram", "default").read_text(encoding="utf-8") == before
    assert len(warnings) == 1
    message, fields = warnings[0]
    assert "refresh failed" in message
    blob = message + " " + json.dumps(fields)
    _assert_no_secrets(blob)
    assert fields["account"] == "default"
    assert fields["auth"] == "facebook"
    assert fields["error"] == "HTTP 400"
    assert fields["expires_in_days"] < 7


def test_httpx_info_log_does_not_include_the_token(tmp_path, caplog) -> None:
    _seed(tmp_path, "default", OLD, _iso(timedelta(days=1)))

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.params.get("fb_exchange_token") == OLD:
            return httpx.Response(200, json={"access_token": NEW, "token_type": "bearer", "expires_in": 5184000})
        return httpx.Response(200, json={"ok": True})

    transport = httpx.MockTransport(handler)
    with caplog.at_level(logging.DEBUG):
        with _RealClient(transport=transport) as raw:
            raw.get(
                "https://graph.facebook.com/v21.0/oauth/access_token",
                params={"fb_exchange_token": "VISIBLE_TOKEN", "client_secret": "VISIBLE_SECRET"},
            )
        assert "VISIBLE_TOKEN" in caplog.text
        caplog.clear()
        token = prepare_instagram_token(
            access_token=OLD,
            working_directory=tmp_path,
            now=NOW,
            app_id=APP_ID,
            app_secret=APP_SECRET,
            client=_RealClient(transport=transport),
        )
    assert token == NEW
    _assert_no_secrets(caplog.text)


def test_refresh_exception_text_is_not_logged(tmp_path, warnings) -> None:
    _seed(tmp_path, "default", OLD, _iso(timedelta(days=1)))

    class Leak(FakeClient):
        def get(self, url, headers=None, params=None, **kwargs):
            raise RuntimeError(f"transport blew up {OLD} {APP_SECRET}")

    token = prepare_instagram_token(
        access_token=OLD,
        working_directory=tmp_path,
        now=NOW,
        app_id=APP_ID,
        app_secret=APP_SECRET,
        client=Leak(lambda *a: None),
    )
    assert token == OLD
    assert len(warnings) == 1
    blob = warnings[0][0] + " " + json.dumps(warnings[0][1])
    _assert_no_secrets(blob)
    assert warnings[0][1]["error"] == "RuntimeError"


def test_facebook_missing_app_secret_skips_refresh_and_warns_once(tmp_path, monkeypatch, warnings) -> None:
    _seed(tmp_path, "default", OLD, _iso(timedelta(days=1)))
    client = _patch(monkeypatch, _upload_handler())
    _upload(tmp_path, app_id="", app_secret=None)
    _upload(tmp_path, app_id="", app_secret="")
    assert _refresh_calls(client) == []
    assert _debug_calls(client) == []
    assert _create_calls(client)[0]["headers"]["Authorization"] == f"Bearer {OLD}"
    assert _create_calls(client)[1]["headers"]["Authorization"] == f"Bearer {OLD}"
    assert len(warnings) == 1
    message, fields = warnings[0]
    assert "INSTAGRAM_APP_ID" in message and "INSTAGRAM_APP_SECRET" in message
    assert "skipped" in message
    assert fields["auth"] == "facebook"
    assert "error" not in fields
    _assert_no_secrets(message + " " + json.dumps(fields))


def test_fresh_token_with_missing_facebook_app_does_not_warn(tmp_path, warnings) -> None:
    _seed(tmp_path, "default", OLD, _iso(timedelta(days=30)))
    client = FakeClient(_upload_handler())
    token = prepare_instagram_token(
        access_token=OLD,
        working_directory=tmp_path,
        now=NOW,
        app_id="",
        app_secret="",
        client=client,
    )
    assert token == OLD
    assert client.calls == []
    assert warnings == []


def test_changed_env_token_discards_saved_token(tmp_path, monkeypatch) -> None:
    save_oauth_state(
        tmp_path,
        "instagram",
        "default",
        {
            "access_token": "persisted-token",
            "expires_at": _iso(timedelta(days=30)),
            "seed": _token_fingerprint(OLD),
        },
    )
    far = int((NOW + timedelta(days=40)).timestamp())
    client = _patch(monkeypatch, _upload_handler(debug_expires=far))
    token = prepare_instagram_token(
        access_token="rotated-env-token",
        working_directory=tmp_path,
        now=NOW,
        app_id=APP_ID,
        app_secret=APP_SECRET,
        client=client,
    )
    assert token == "rotated-env-token"
    assert _debug_calls(client)[0]["params"]["input_token"] == "rotated-env-token"
    assert _refresh_calls(client) == []
    stored = load_oauth_state(tmp_path, "instagram", "default")
    assert stored["access_token"] == "rotated-env-token"
    assert stored["seed"] == _token_fingerprint("rotated-env-token")
    assert "persisted-token" not in json.dumps(stored)


def test_matching_env_seed_keeps_saved_token(tmp_path) -> None:
    save_oauth_state(
        tmp_path,
        "instagram",
        "default",
        {
            "access_token": "persisted-token",
            "expires_at": _iso(timedelta(days=30)),
            "seed": _token_fingerprint(OLD),
        },
    )
    client = FakeClient(_upload_handler())
    token = prepare_instagram_token(
        access_token=OLD,
        working_directory=tmp_path,
        now=NOW,
        app_id=APP_ID,
        app_secret=APP_SECRET,
        client=client,
    )
    assert token == "persisted-token"
    assert client.calls == []


def test_instagram_login_uses_graph_instagram_refresh(tmp_path, monkeypatch) -> None:
    _seed(tmp_path, "default", OLD, _iso(timedelta(days=1)))
    client = _patch(monkeypatch, _upload_handler())
    _upload(tmp_path, auth="instagram", app_id="", app_secret="")
    refresh = _refresh_calls(client)
    assert len(refresh) == 1
    assert refresh[0]["url"] == f"{GRAPH_INSTAGRAM}/refresh_access_token"
    assert refresh[0]["params"] == {"grant_type": "ig_refresh_token", "access_token": OLD}
    assert "client_secret" not in refresh[0]["params"]
    assert _debug_calls(client) == []
    create = _create_calls(client)[0]
    assert create["url"] == f"{GRAPH_INSTAGRAM}/v21.0/{IG_USER}/media"
    assert create["headers"]["Authorization"] == f"Bearer {NEW}"
    assert all(GRAPH_FACEBOOK not in c["url"] for c in client.calls)
    status_urls = [c["url"] for c in client.calls if c["method"] == "GET" and "status_code" in str(c["params"])]
    assert status_urls == [f"{GRAPH_INSTAGRAM}/v21.0/C1"]


def test_facebook_mode_does_not_call_instagram_host(tmp_path, monkeypatch) -> None:
    _seed(tmp_path, "default", OLD, _iso(timedelta(days=1)))
    client = _patch(monkeypatch, _upload_handler(version="v22.0"))
    _upload(tmp_path, graph_version="v22.0")
    refresh = _refresh_calls(client)[0]
    assert refresh["url"] == f"{GRAPH_FACEBOOK}/v22.0/oauth/access_token"
    assert all(GRAPH_INSTAGRAM not in c["url"] for c in client.calls)
    assert all("v21.0" not in c["url"] for c in client.calls)
    assert _create_calls(client)[0]["url"] == f"{GRAPH_FACEBOOK}/v22.0/{IG_USER}/media"


def test_rupload_fallback_uses_only_the_configured_version(tmp_path, monkeypatch) -> None:
    _seed(tmp_path, "default", OLD, _iso(timedelta(days=30)))
    client = _patch(monkeypatch, _upload_handler(version="v19.0", rupload_status=400))
    with pytest.raises(RuntimeError, match="rupload failed"):
        _upload(tmp_path, graph_version="v19.0")
    urls = [c["url"] for c in _rupload_calls(client)]
    assert urls == [
        "https://rupload.facebook.com/ig-api-upload/v19.0/C1",
        "https://rupload.facebook.com/ig-api-upload/v19.0/C1",
    ]
    assert all("v21.0" not in c["url"] for c in client.calls)


def test_account_ig2_reads_env_token_and_persists_its_own_file(tmp_path, monkeypatch, warnings) -> None:
    monkeypatch.setenv("INSTAGRAM_ACCESS_TOKEN_IG2", "tok-ig2")
    monkeypatch.setenv("INSTAGRAM_IG_USER_ID_IG2", "222")
    settings = Settings(
        working_directory=str(tmp_path / "data"),
        clip_storage_root=str(tmp_path / "clips"),
        instagram_access_token="tok-default",
        instagram_ig_user_id="111",
        instagram_auth="facebook",
        instagram_app_id=APP_ID,
        instagram_app_secret=APP_SECRET,
        instagram_graph_version="v21.0",
    )
    assert _account_env(settings, "instagram_access_token", "ig2") == "tok-ig2"
    soon = int((NOW + timedelta(days=3)).timestamp())
    client = _patch(monkeypatch, _upload_handler(debug_expires=soon))

    clip_id = "33333333-2222-3333-4444-555555555555"
    src = tmp_path / "clips" / "9" / "pending_upload" / f"{clip_id}.mp4"
    src.parent.mkdir(parents=True)
    src.write_bytes(b"fake-mp4")
    job = Job(
        id="job-ig2",
        type="publish",
        payload={
            "clip_id": clip_id,
            "campaign_id": 9,
            "platform": "instagram",
            "account": "ig2",
            "title": "t",
            "caption": "caption",
            "dry_run": False,
        },
    )
    monkeypatch.setattr("app.services.instagram_upload.datetime", _FixedDatetime)

    out = PublishJob(settings, job).execute()
    assert out["publications"][0]["account"] == "ig2"
    refresh = _refresh_calls(client)
    assert len(refresh) == 1
    assert refresh[0]["params"]["fb_exchange_token"] == "tok-ig2"
    assert refresh[0]["params"]["client_id"] == APP_ID
    assert _create_calls(client)[0]["headers"]["Authorization"] == f"Bearer {NEW}"
    assert _create_calls(client)[0]["url"].startswith(f"{GRAPH_FACEBOOK}/v21.0/222/media")
    state_path = tmp_path / "data" / "oauth_state" / "instagram_ig2.json"
    assert state_path.is_file()
    assert (state_path.stat().st_mode & 0o777) == 0o600
    stored = json.loads(state_path.read_text(encoding="utf-8"))
    assert stored["access_token"] == NEW
    assert stored["account"] == "ig2"
    assert APP_SECRET not in state_path.read_text(encoding="utf-8")
    assert "tok-ig2" not in state_path.read_text(encoding="utf-8")
    assert not (tmp_path / "data" / "oauth_state" / "instagram_default.json").exists()
    assert warnings == []


class _FixedDatetime(datetime):
    @classmethod
    def now(cls, tz=None):
        return NOW if tz is not None else NOW.replace(tzinfo=None)

    @classmethod
    def fromtimestamp(cls, ts, tz=None):
        return datetime.fromtimestamp(ts, tz)
