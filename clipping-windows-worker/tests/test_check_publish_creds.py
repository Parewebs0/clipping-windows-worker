"""Read-only publish credential check. googleapiclient, google.oauth2 and httpx are mocked."""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

from app.config import Settings

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "check_publish_creds.py"

YT_CLIENT_ID = "yt-client-id-DO-NOT-PRINT-7f3a1c"
YT_CLIENT_SECRET = "yt-client-secret-DO-NOT-PRINT-7f3a1c"
YT_REFRESH = "yt-refresh-token-DO-NOT-PRINT-7f3a1c"
IG_TOKEN = "ig-access-token-DO-NOT-PRINT-7f3a1c"
IG_USER = "ig-user-id-DO-NOT-PRINT-7f3a1c"
TT_TOKEN = "tt-access-token-DO-NOT-PRINT-7f3a1c"
SECRETS = (YT_CLIENT_ID, YT_CLIENT_SECRET, YT_REFRESH, IG_TOKEN, IG_USER, TT_TOKEN)

UPLOAD = "https://www.googleapis.com/auth/youtube.upload"
ANALYTICS = "https://www.googleapis.com/auth/yt-analytics.readonly"
YOUTUBE_MANAGE = "https://www.googleapis.com/auth/youtube"


def _load():
    spec = importlib.util.spec_from_file_location("check_publish_creds", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def creds():
    return _load()


@pytest.fixture(autouse=True)
def _clear_instagram_endpoint_env(monkeypatch):
    monkeypatch.delenv("INSTAGRAM_AUTH", raising=False)
    monkeypatch.delenv("INSTAGRAM_GRAPH_VERSION", raising=False)


def _settings(**overrides) -> Settings:
    values = dict(
        api_base_url="http://localhost",
        api_token="test",
        worker_id="test-worker",
        working_directory="/tmp/check-publish-creds",
        youtube_client_id=YT_CLIENT_ID,
        youtube_client_secret=YT_CLIENT_SECRET,
        youtube_refresh_token=YT_REFRESH,
        instagram_access_token=IG_TOKEN,
        instagram_ig_user_id=IG_USER,
        tiktok_access_token=TT_TOKEN,
    )
    values.update(overrides)
    return Settings(**values)


def _assert_no_secrets(text: str) -> None:
    for secret in SECRETS:
        assert secret not in text


class _Response:
    def __init__(self, status_code: int, payload: dict, text: str | None = None):
        self.status_code = status_code
        self._payload = payload
        self.content = b"1" if payload or text else b""
        self.text = text if text is not None else ""

    def json(self):
        return self._payload


class _Client:
    def __init__(self, routes):
        self.routes = routes
        self.calls = []

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def request(self, method, url, **kwargs):
        self.calls.append((method, url, kwargs))
        handler = self.routes[(method, url)]
        return handler(method, url, kwargs)


def _install_http(monkeypatch, creds, routes):
    client = _Client(routes)

    def factory(*args, **kwargs):
        client.init = (args, kwargs)
        return client

    monkeypatch.setattr(creds.httpx, "Client", factory)
    return client


def _install_youtube(monkeypatch, scope_text: str, title: str = "Boxabl", error: Exception | None = None):
    calls = {"scope_text": scope_text}

    def refresh_grant(request, token_uri, refresh_token, client_id, client_secret, scopes=None, **kwargs):
        calls["refresh"] = {
            "token_uri": token_uri,
            "refresh_token": refresh_token,
            "client_id": client_id,
            "client_secret": client_secret,
            "scopes": scopes,
        }
        if error is not None:
            raise error
        return (
            "ya-access-DO-NOT-PRINT-7f3a1c",
            refresh_token,
            None,
            {"scope": calls["scope_text"]},
            None,
        )

    def build(service, version, credentials=None, cache_discovery=None):
        calls["build"] = {
            "service": service,
            "version": version,
            "cache_discovery": cache_discovery,
            "credentials": credentials,
        }

        class _YouTube:
            def channels(self):
                return self

            def list(self, **kwargs):
                calls["list"] = kwargs
                return self

            def execute(self):
                if title is None:
                    return {"items": []}
                return {"items": [{"snippet": {"title": title}}]}

            def videos(self):
                raise AssertionError("videos.insert would upload")

        return _YouTube()

    monkeypatch.setattr("google.oauth2.reauth.refresh_grant", refresh_grant)
    monkeypatch.setattr("googleapiclient.discovery.build", build)
    return calls


def _prime_youtube(monkeypatch, scope_text: str, title: str = "Boxabl", error: Exception | None = None):
    return _install_youtube(monkeypatch, scope_text, title=title, error=error)


def test_hosts_match_uploaders(creds):
    import app.services.instagram_upload as ig
    import app.services.tiktok_upload as tt
    import app.services.youtube_upload as yt

    # MOLI-77 (#26) renames GRAPH / API_VERSION to GRAPH_FACEBOOK /
    # DEFAULT_GRAPH_VERSION. This file has to pass on both.
    facebook = getattr(ig, "GRAPH_FACEBOOK", None)
    if facebook is None:
        facebook = ig.GRAPH
    version = getattr(ig, "DEFAULT_GRAPH_VERSION", None)
    if version is None:
        version = ig.API_VERSION
    assert creds.GRAPH_FACEBOOK == facebook
    assert creds.DEFAULT_GRAPH_VERSION == version
    instagram_login = getattr(ig, "GRAPH_INSTAGRAM", None)
    if instagram_login is not None:
        assert creds.GRAPH_INSTAGRAM == instagram_login
    legacy_graph = getattr(ig, "GRAPH", None)
    if legacy_graph is not None:
        assert creds.GRAPH == legacy_graph
    legacy_version = getattr(ig, "API_VERSION", None)
    if legacy_version is not None:
        assert creds.API_VERSION == legacy_version
    assert creds.TIKTOK_OPEN == tt.OPEN
    assert creds.TOKEN_URI == yt.TOKEN_URI


def test_missing_refresh_token_does_not_call_network(creds, monkeypatch, capsys):
    def boom(*args, **kwargs):
        raise AssertionError("network")

    monkeypatch.setattr("google.oauth2.reauth.refresh_grant", boom)
    monkeypatch.setattr("googleapiclient.discovery.build", boom)
    monkeypatch.setattr(creds.httpx, "Client", boom)
    settings = _settings(youtube_refresh_token="")
    code = creds.main(["--platform", "youtube", "--account", "default"], settings=settings)
    captured = capsys.readouterr()
    assert code == 1
    assert captured.out.strip() == "youtube default FAIL YOUTUBE_REFRESH_TOKEN missing"
    _assert_no_secrets(captured.out + captured.err)
    assert YT_CLIENT_ID not in captured.out


def test_yt2_missing_override_fails_without_network_even_if_default_exists(creds, monkeypatch, capsys):
    def boom(*args, **kwargs):
        raise AssertionError("network")

    monkeypatch.delenv("YOUTUBE_CLIENT_ID_YT2", raising=False)
    monkeypatch.delenv("YOUTUBE_CLIENT_SECRET_YT2", raising=False)
    monkeypatch.delenv("YOUTUBE_REFRESH_TOKEN_YT2", raising=False)
    monkeypatch.setattr("google.oauth2.reauth.refresh_grant", boom)
    monkeypatch.setattr("googleapiclient.discovery.build", boom)
    code = creds.main(["--platform", "youtube", "--account", "yt2"], settings=_settings())
    captured = capsys.readouterr()
    assert code == 1
    assert "YOUTUBE_CLIENT_ID_YT2 missing" in captured.out
    assert "YOUTUBE_CLIENT_SECRET_YT2 missing" in captured.out
    assert "YOUTUBE_REFRESH_TOKEN_YT2 missing" in captured.out
    _assert_no_secrets(captured.out + captured.err)


def test_missing_instagram_and_tiktok_do_not_call_network(creds, monkeypatch, capsys):
    def boom(*args, **kwargs):
        raise AssertionError("network")

    monkeypatch.setattr(creds.httpx, "Client", boom)
    settings = _settings(instagram_access_token="", instagram_ig_user_id="", tiktok_access_token="")
    ig = creds.check_instagram(settings, "default")
    tt = creds.check_tiktok(settings, "default")
    assert ig == "instagram default FAIL INSTAGRAM_ACCESS_TOKEN missing, INSTAGRAM_IG_USER_ID missing"
    assert tt == "tiktok default FAIL TIKTOK_ACCESS_TOKEN missing"
    captured = capsys.readouterr()
    _assert_no_secrets(ig + tt + captured.out + captured.err)


def test_youtube_ok_prints_title_upload_and_analytics(creds, monkeypatch, capsys):
    calls = _prime_youtube(monkeypatch, f"{UPLOAD} {ANALYTICS} {YOUTUBE_MANAGE}")
    code = creds.main(["--platform", "youtube"], settings=_settings())
    captured = capsys.readouterr()
    assert code == 0
    assert (
        captured.out.strip()
        == 'youtube default OK title="Boxabl" youtube.upload=yes yt-analytics.readonly=yes'
    )
    assert calls["refresh"]["scopes"] is None
    assert calls["refresh"]["refresh_token"] == YT_REFRESH
    assert calls["list"] == {"part": "snippet", "mine": True}
    assert calls["build"]["service"] == "youtube"
    assert calls["build"]["version"] == "v3"
    assert calls["build"]["cache_discovery"] is False
    _assert_no_secrets(captured.out + captured.err)


def test_channel_title_containing_fail_stays_exit_zero(creds, monkeypatch, capsys):
    _prime_youtube(monkeypatch, UPLOAD, title="DO NOT FAIL the show")
    code = creds.main(["--platform", "youtube"], settings=_settings())
    captured = capsys.readouterr()
    assert code == 0
    assert captured.out.split()[2] == "OK"
    assert 'title="DO NOT FAIL the show"' in captured.out
    _assert_no_secrets(captured.out + captured.err)


def test_youtube_omits_analytics_when_absent(creds, monkeypatch):
    _prime_youtube(monkeypatch, f"{UPLOAD} {YOUTUBE_MANAGE}")
    line = creds.check_youtube(_settings(), "default")
    assert line == 'youtube default OK title="Boxabl" youtube.upload=yes'
    assert "yt-analytics.readonly" not in line


def test_youtube_fails_when_upload_scope_missing(creds, monkeypatch):
    _prime_youtube(monkeypatch, YOUTUBE_MANAGE, title="Boxabl")
    line = creds.check_youtube(_settings(), "default")
    assert line.startswith("youtube default FAIL missing youtube.upload")
    assert 'title="Boxabl"' in line
    assert "youtube.upload=yes" not in line


def test_refresh_error_redacts_token(creds, monkeypatch, capsys):
    _prime_youtube(
        monkeypatch,
        UPLOAD,
        error=RuntimeError(f"invalid_grant {YT_REFRESH} secret {YT_CLIENT_SECRET}"),
    )
    code = creds.main(["--platform", "youtube"], settings=_settings())
    captured = capsys.readouterr()
    assert code == 1
    assert "invalid_grant" in captured.out
    assert "[redacted]" in captured.out
    _assert_no_secrets(captured.out + captured.err)


def test_instagram_business_and_creator(creds, monkeypatch):
    def routes_for(account_type):
        def handler(method, url, kwargs):
            assert method == "GET"
            assert url.endswith(f"/{IG_USER}")
            assert kwargs["params"] == {"fields": "username,account_type"}
            assert kwargs["headers"]["Authorization"] == f"Bearer {IG_TOKEN}"
            return _Response(200, {"username": "boxabl", "account_type": account_type})

        return {("GET", f"{creds.GRAPH}/{creds.API_VERSION}/{IG_USER}"): handler}

    for account_type in ("BUSINESS", "CREATOR", "MEDIA_CREATOR"):
        _install_http(monkeypatch, creds, routes_for(account_type))
        line = creds.check_instagram(_settings(), "default")
        assert line == f'instagram default OK username="boxabl" account_type={account_type}'
        _assert_no_secrets(line)


def test_instagram_personal_fails(creds, monkeypatch):
    url = f"{creds.GRAPH}/{creds.API_VERSION}/{IG_USER}"

    def handler(method, url, kwargs):
        return _Response(200, {"username": "boxabl", "account_type": "PERSONAL"})

    _install_http(monkeypatch, creds, {("GET", url): handler})
    line = creds.check_instagram(_settings(), "default")
    assert line.startswith("instagram default FAIL account_type=PERSONAL")
    assert "BUSINESS or CREATOR" in line
    assert 'username="boxabl"' in line
    _assert_no_secrets(line)


def test_instagram_http_error_redacts_token(creds, monkeypatch, capsys):
    url = f"{creds.GRAPH}/{creds.API_VERSION}/{IG_USER}"

    def handler(method, url, kwargs):
        return _Response(400, {"error": {"message": f"bad {IG_TOKEN}", "code": 190}})

    _install_http(monkeypatch, creds, {("GET", url): handler})
    code = creds.main(["--platform", "instagram"], settings=_settings())
    captured = capsys.readouterr()
    assert code == 1
    assert "190" in captured.out
    _assert_no_secrets(captured.out + captured.err)


def test_tiktok_prints_username_and_privacy_options(creds, monkeypatch):
    url = f"{creds.TIKTOK_OPEN}/v2/post/publish/creator_info/query/"

    def handler(method, url, kwargs):
        assert method == "POST"
        assert kwargs["content"] == b"{}"
        assert kwargs["headers"]["Authorization"] == f"Bearer {TT_TOKEN}"
        return _Response(
            200,
            {
                "data": {
                    "creator_username": "boxabl",
                    "privacy_level_options": ["PUBLIC_TO_EVERYONE", "SELF_ONLY"],
                },
                "error": {"code": "ok", "message": ""},
            },
        )

    _install_http(monkeypatch, creds, {("POST", url): handler})
    line = creds.check_tiktok(_settings(), "default")
    assert line == 'tiktok default OK username="boxabl" privacy_level_options=PUBLIC_TO_EVERYONE,SELF_ONLY'
    _assert_no_secrets(line)


def test_tiktok_error_redacts_token(creds, monkeypatch):
    url = f"{creds.TIKTOK_OPEN}/v2/post/publish/creator_info/query/"

    def handler(method, url, kwargs):
        return _Response(401, {"error": {"code": "access_token_invalid", "message": TT_TOKEN}})

    _install_http(monkeypatch, creds, {("POST", url): handler})
    line = creds.check_tiktok(_settings(), "default")
    assert "access_token_invalid" in line
    assert line.startswith("tiktok default FAIL")
    _assert_no_secrets(line)


def test_exit_codes_and_secrets_absent_from_stdout(creds, monkeypatch, capsys):
    scope = f"{UPLOAD} {ANALYTICS}"
    _prime_youtube(monkeypatch, scope)
    ig_url = f"{creds.GRAPH}/{creds.API_VERSION}/{IG_USER}"
    tt_url = f"{creds.TIKTOK_OPEN}/v2/post/publish/creator_info/query/"

    def ig_handler(method, url, kwargs):
        assert "media" not in url
        return _Response(200, {"username": "boxabl", "account_type": "BUSINESS"})

    def tt_handler(method, url, kwargs):
        assert not url.endswith("/video/init/")
        assert not url.endswith("/inbox/video/init/")
        return _Response(
            200,
            {
                "data": {"creator_username": "boxabl", "privacy_level_options": ["SELF_ONLY"]},
                "error": {"code": "ok"},
            },
        )

    _install_http(
        monkeypatch,
        creds,
        {("GET", ig_url): ig_handler, ("POST", tt_url): tt_handler},
    )
    code = creds.main(["--platform", "all", "--account", "default"], settings=_settings())
    captured = capsys.readouterr()
    lines = [line for line in captured.out.splitlines() if line]
    assert code == 0
    assert lines[0].startswith("youtube default OK")
    assert lines[1].startswith("instagram default OK")
    assert lines[2].startswith("tiktok default OK")
    _assert_no_secrets(captured.out + captured.err)

    _install_http(monkeypatch, creds, {("GET", ig_url): ig_handler, ("POST", tt_url): tt_handler})
    settings = _settings(tiktok_access_token="")
    code = creds.main(["--platform", "all"], settings=settings)
    captured = capsys.readouterr()
    assert code == 1
    assert "tiktok default FAIL TIKTOK_ACCESS_TOKEN missing" in captured.out
    assert "youtube default OK" in captured.out
    _assert_no_secrets(captured.out + captured.err)


def _write_oauth(tmp_path: Path, provider: str, account: str, payload: dict) -> Path:
    path = tmp_path / "oauth_state" / f"{provider}_{account}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_instagram_auth_and_graph_version(creds, monkeypatch):
    monkeypatch.setenv("INSTAGRAM_AUTH", "instagram")
    monkeypatch.setenv("INSTAGRAM_GRAPH_VERSION", "v22.0")
    url = f"{creds.GRAPH_INSTAGRAM}/v22.0/{IG_USER}"

    def handler(method, url, kwargs):
        assert kwargs["headers"]["Authorization"] == f"Bearer {IG_TOKEN}"
        return _Response(200, {"username": "boxabl", "account_type": "BUSINESS"})

    _install_http(monkeypatch, creds, {("GET", url): handler})
    line = creds.check_instagram(_settings(), "default")
    assert line == 'instagram default OK username="boxabl" account_type=BUSINESS'
    _assert_no_secrets(line)


def test_invalid_graph_version_stays_on_default(creds, monkeypatch):
    # Settings rejects a malformed INSTAGRAM_GRAPH_VERSION since MOLI-77 (#26),
    # so build it first; the script must still ignore a bad raw env value.
    settings = _settings()
    monkeypatch.setenv("INSTAGRAM_GRAPH_VERSION", "latest")
    url = f"{creds.GRAPH_FACEBOOK}/{creds.DEFAULT_GRAPH_VERSION}/{IG_USER}"

    def handler(method, url, kwargs):
        return _Response(200, {"username": "boxabl", "account_type": "BUSINESS"})

    _install_http(monkeypatch, creds, {("GET", url): handler})
    line = creds.check_instagram(settings, "default")
    assert line.startswith("instagram default OK")


def test_instagram_oauth_state_token_is_used_and_not_printed(creds, monkeypatch, tmp_path):
    stored = "ig-stored-token-DO-NOT-PRINT-7f3a1c"
    path = _write_oauth(
        tmp_path,
        "instagram",
        "default",
        {"access_token": stored, "seed": hashlib.sha256(IG_TOKEN.encode()).hexdigest()},
    )
    before = path.read_bytes()
    url = f"{creds.GRAPH}/{creds.API_VERSION}/{IG_USER}"

    def handler(method, url, kwargs):
        assert kwargs["headers"]["Authorization"] == f"Bearer {stored}"
        return _Response(200, {"username": "boxabl", "account_type": "CREATOR"})

    _install_http(monkeypatch, creds, {("GET", url): handler})
    settings = _settings(working_directory=str(tmp_path))
    line = creds.check_instagram(settings, "default")
    assert line == 'instagram default OK username="boxabl" account_type=CREATOR'
    assert stored not in line
    assert IG_TOKEN not in line
    assert path.read_bytes() == before


def test_instagram_oauth_state_covers_missing_env_token(creds, monkeypatch, tmp_path):
    stored = "ig-stored-token-DO-NOT-PRINT-7f3a1c"
    _write_oauth(tmp_path, "instagram", "ig2", {"access_token": stored})
    monkeypatch.delenv("INSTAGRAM_ACCESS_TOKEN_IG2", raising=False)
    monkeypatch.setenv("INSTAGRAM_IG_USER_ID_IG2", IG_USER)
    url = f"{creds.GRAPH}/{creds.API_VERSION}/{IG_USER}"

    def handler(method, url, kwargs):
        assert kwargs["headers"]["Authorization"] == f"Bearer {stored}"
        return _Response(200, {"username": "boxabl", "account_type": "BUSINESS"})

    _install_http(monkeypatch, creds, {("GET", url): handler})
    settings = _settings(working_directory=str(tmp_path), instagram_access_token="")
    line = creds.check_instagram(settings, "ig2")
    assert line.startswith("instagram ig2 OK")
    assert stored not in line
    assert IG_TOKEN not in line


def test_instagram_oauth_state_still_requires_ig_user_id(creds, monkeypatch, tmp_path):
    stored = "ig-stored-token-DO-NOT-PRINT-7f3a1c"
    _write_oauth(tmp_path, "instagram", "default", {"access_token": stored})

    def boom(*args, **kwargs):
        raise AssertionError("network")

    monkeypatch.setattr(creds.httpx, "Client", boom)
    settings = _settings(
        working_directory=str(tmp_path),
        instagram_access_token="",
        instagram_ig_user_id="",
    )
    line = creds.check_instagram(settings, "default")
    assert line == "instagram default FAIL INSTAGRAM_IG_USER_ID missing"
    assert stored not in line


def test_instagram_seed_mismatch_keeps_env_token(creds, monkeypatch, tmp_path):
    stored = "ig-stored-token-DO-NOT-PRINT-7f3a1c"
    _write_oauth(tmp_path, "instagram", "default", {"access_token": stored, "seed": "0" * 64})
    url = f"{creds.GRAPH}/{creds.API_VERSION}/{IG_USER}"

    def handler(method, url, kwargs):
        assert kwargs["headers"]["Authorization"] == f"Bearer {IG_TOKEN}"
        return _Response(200, {"username": "boxabl", "account_type": "BUSINESS"})

    _install_http(monkeypatch, creds, {("GET", url): handler})
    line = creds.check_instagram(_settings(working_directory=str(tmp_path)), "default")
    assert line.startswith("instagram default OK")
    assert stored not in line


def test_corrupt_oauth_state_falls_back_to_env_token(creds, monkeypatch, tmp_path):
    path = tmp_path / "oauth_state" / "instagram_default.json"
    path.parent.mkdir(parents=True)
    path.write_text("{not-json " + IG_TOKEN, encoding="utf-8")
    url = f"{creds.GRAPH}/{creds.API_VERSION}/{IG_USER}"

    def handler(method, url, kwargs):
        assert kwargs["headers"]["Authorization"] == f"Bearer {IG_TOKEN}"
        return _Response(200, {"username": "boxabl", "account_type": "BUSINESS"})

    _install_http(monkeypatch, creds, {("GET", url): handler})
    line = creds.check_instagram(_settings(working_directory=str(tmp_path)), "default")
    assert line.startswith("instagram default OK")
    assert IG_TOKEN not in line
    assert "not-json" not in line


def test_tiktok_oauth_state_token_is_used_and_not_printed(creds, monkeypatch, tmp_path):
    stored = "tt-stored-token-DO-NOT-PRINT-7f3a1c"
    path = _write_oauth(tmp_path, "tiktok", "default", {"access_token": stored})
    before = path.read_bytes()
    url = f"{creds.TIKTOK_OPEN}/v2/post/publish/creator_info/query/"

    def handler(method, url, kwargs):
        assert kwargs["headers"]["Authorization"] == f"Bearer {stored}"
        assert kwargs["content"] == b"{}"
        return _Response(
            200,
            {
                "data": {"creator_username": "boxabl", "privacy_level_options": ["SELF_ONLY"]},
                "error": {"code": "ok"},
            },
        )

    _install_http(monkeypatch, creds, {("POST", url): handler})
    settings = _settings(working_directory=str(tmp_path), tiktok_access_token="")
    line = creds.check_tiktok(settings, "default")
    assert line == 'tiktok default OK username="boxabl" privacy_level_options=SELF_ONLY'
    assert stored not in line
    assert TT_TOKEN not in line
    assert path.read_bytes() == before


def test_yt2_check_reads_suffixed_env(creds, monkeypatch):
    monkeypatch.setenv("YOUTUBE_CLIENT_ID_YT2", YT_CLIENT_ID)
    monkeypatch.setenv("YOUTUBE_CLIENT_SECRET_YT2", YT_CLIENT_SECRET)
    monkeypatch.setenv("YOUTUBE_REFRESH_TOKEN_YT2", YT_REFRESH)
    calls = _prime_youtube(monkeypatch, UPLOAD, title="Second channel")
    line = creds.check_youtube(_settings(youtube_refresh_token="legacy-refresh-DO-NOT-PRINT"), "yt2")
    assert calls["refresh"]["refresh_token"] == YT_REFRESH
    assert calls["refresh"]["client_id"] == YT_CLIENT_ID
    assert 'title="Second channel"' in line
    assert line.startswith("youtube yt2 OK")
    assert "legacy-refresh-DO-NOT-PRINT" not in line
