"""Read-only check that publish credentials for one account work.

Never uploads or publishes. One line per platform, no tokens:

    platform account OK|FAIL reason

Exit 0 when every checked platform is OK, 1 otherwise.

    python scripts/check_publish_creds.py
    python scripts/check_publish_creds.py --platform youtube --account default
    python scripts/check_publish_creds.py --platform instagram --account ig2
    python scripts/check_publish_creds.py --platform tiktok --account studio

``default`` reads the legacy settings fields (``YOUTUBE_REFRESH_TOKEN``, ...).
Any other account reads only ``<FIELD>_<KEY>`` from the process environment
(``YOUTUBE_REFRESH_TOKEN_YT2``). A missing suffix does not fall back to the
default account: that would report OK for the wrong channel. YouTube publish
uses the same rule (all three suffixed variables, or the job fails). Instagram
and TikTok publish still fall back per field when the suffix is absent.

Instagram's host and version follow ``INSTAGRAM_AUTH`` (``facebook`` or
``instagram``) and ``INSTAGRAM_GRAPH_VERSION`` (``v21.0`` when unset or
invalid), the same rules as the uploader once that code is present. A stored
``access_token`` in ``<working_directory>/oauth_state/<provider>_<account>.json``
is preferred for Instagram (unless the file's ``seed`` says the env token
changed) and TikTok. The file is only read. Nothing is refreshed or written.
The job publishes exactly ``payload.account`` (or ``default``). It does not
pick the first healthy account, and ``channels.list`` items[0] is not a
campaign selector.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.jobs.publish import _account_key  # noqa: E402

# Same hosts as the uploaders. Compared in tests so they cannot drift.
# ``GRAPH`` / ``API_VERSION`` stay as the facebook + v21.0 names from before
# MOLI-77 renamed them to ``GRAPH_FACEBOOK`` / ``DEFAULT_GRAPH_VERSION``.
GRAPH_FACEBOOK = "https://graph.facebook.com"
GRAPH_INSTAGRAM = "https://graph.instagram.com"
DEFAULT_GRAPH_VERSION = "v21.0"
GRAPH = GRAPH_FACEBOOK
API_VERSION = DEFAULT_GRAPH_VERSION
TIKTOK_OPEN = "https://open.tiktokapis.com"
TOKEN_URI = "https://oauth2.googleapis.com/token"
_VERSION_RE = re.compile(r"v\d+(?:\.\d+)?")
_OAUTH_UNSAFE = re.compile(r"[^a-z0-9._-]+")

PLATFORMS = ("youtube", "instagram", "tiktok")
_UPLOAD_SCOPE = "youtube.upload"
_ANALYTICS_SCOPE = "yt-analytics.readonly"
_IG_PROFESSIONAL = {"BUSINESS", "CREATOR", "MEDIA_CREATOR"}

_YOUTUBE_FIELDS = (
    ("youtube_client_id", "YOUTUBE_CLIENT_ID"),
    ("youtube_client_secret", "YOUTUBE_CLIENT_SECRET"),
    ("youtube_refresh_token", "YOUTUBE_REFRESH_TOKEN"),
)
_INSTAGRAM_FIELDS = (
    ("instagram_access_token", "INSTAGRAM_ACCESS_TOKEN"),
    ("instagram_ig_user_id", "INSTAGRAM_IG_USER_ID"),
)
_TIKTOK_FIELDS = (("tiktok_access_token", "TIKTOK_ACCESS_TOKEN"),)


def _account_label(account: str | None) -> str:
    raw = str(account or "").strip()
    if not raw:
        return "default"
    return re.sub(r"\s+", "_", raw)


def _env_name(base_field: str, account: str) -> str:
    key = _account_key(account)
    base = base_field.upper()
    if key and key != "DEFAULT":
        return f"{base}_{key}"
    return base


def _lookup(settings, base_field: str, account: str) -> tuple[str, str]:
    """Return ``(value, env var name)``.

    Non-default accounts read only the suffixed variable. They do not fall
    back to the legacy field.
    """
    name = _env_name(base_field, account)
    key = _account_key(account)
    if key and key != "DEFAULT":
        return (os.environ.get(name) or "").strip(), name
    return (getattr(settings, base_field, None) or "").strip(), name


def _missing(settings, fields: tuple[tuple[str, str], ...], account: str) -> tuple[dict[str, str], str]:
    values: dict[str, str] = {}
    missing: list[str] = []
    for base_field, _base_name in fields:
        value, name = _lookup(settings, base_field, account)
        values[base_field] = value
        if not value:
            missing.append(f"{name} missing")
    return values, ", ".join(missing)


def _explicit(settings, field: str) -> str:
    value = getattr(settings, field, None)
    if value is None:
        return ""
    return str(value).strip()


def _instagram_auth(settings) -> str:
    """``facebook`` (default) or ``instagram``. Delegates when the uploader can."""
    try:
        from app.services.instagram_upload import resolve_auth
    except ImportError:
        resolve_auth = None  # type: ignore[assignment]
    explicit = _explicit(settings, "instagram_auth")
    if resolve_auth is not None:
        return resolve_auth(explicit or None)
    raw = explicit.lower() if explicit else os.environ.get("INSTAGRAM_AUTH", "").strip().lower()
    if raw in {"facebook", "instagram"}:
        return raw
    return "facebook"


def _instagram_version(settings) -> str:
    try:
        from app.services.instagram_upload import resolve_graph_version
    except ImportError:
        resolve_graph_version = None  # type: ignore[assignment]
    explicit = _explicit(settings, "instagram_graph_version")
    if resolve_graph_version is not None:
        return resolve_graph_version(explicit or None)
    raw = explicit or os.environ.get("INSTAGRAM_GRAPH_VERSION", "").strip()
    if _VERSION_RE.fullmatch(raw):
        return raw
    return DEFAULT_GRAPH_VERSION


def _instagram_host(auth: str) -> str:
    try:
        from app.services.instagram_upload import graph_host
    except ImportError:
        graph_host = None  # type: ignore[assignment]
    if graph_host is not None:
        return graph_host(auth)
    return GRAPH_INSTAGRAM if auth == "instagram" else GRAPH_FACEBOOK


def _oauth_slug(value: str, fallback: str) -> str:
    """Same filename slug as ``app.services.oauth_state.slug``."""
    raw = str(value or "").strip().lower()
    cleaned = _OAUTH_UNSAFE.sub("_", raw).strip("._")
    return cleaned or fallback


def _read_oauth_file(working_directory, provider: str, account: str) -> dict | None:
    """Read oauth state when ``app.services.oauth_state`` is not on this branch.

    Missing, unreadable or non-object files are 'no state'. Contents are never
    included in an error string.
    """
    provider_slug = _oauth_slug(provider, "unknown")
    account_slug = _oauth_slug(account, "default")
    path = Path(working_directory) / "oauth_state" / f"{provider_slug}_{account_slug}.json"
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(data, dict):
        return None
    return data


def _load_oauth_state(working_directory, provider: str, account: str) -> dict | None:
    if not working_directory:
        return None
    try:
        from app.services.oauth_state import load_oauth_state
    except ImportError:
        return _read_oauth_file(working_directory, provider, account)
    try:
        data = load_oauth_state(working_directory, provider, account)
    except Exception:
        return None
    if not isinstance(data, dict):
        return None
    return data


def _fingerprint(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def _stored_access_token(settings, provider: str, account: str, env_token: str, *, match_seed: bool) -> str:
    """Prefer a stored access token. Read-only: never refresh or rewrite the file.

    Instagram ignores the file when ``seed`` is set and the env token no longer
    matches it (the uploader treats that as a rotated env token). TikTok always
    prefers the cached access token when the file has one.
    """
    working = getattr(settings, "working_directory", None)
    state = _load_oauth_state(working, provider, account)
    if not state:
        return ""
    stored = str(state.get("access_token") or "").strip()
    if not stored:
        return ""
    if match_seed and env_token:
        seed = str(state.get("seed") or "").strip()
        if seed and _fingerprint(env_token) != seed and _fingerprint(env_token) != _fingerprint(stored):
            return ""
    return stored


def _without_missing(missing: str, name: str) -> str:
    needle = f"{name} missing"
    parts = [part for part in missing.split(", ") if part and part != needle]
    return ", ".join(parts)


def _redact(text: object, secrets: list[str]) -> str:
    cleaned = " ".join(str(text).split())
    for secret in sorted({s for s in secrets if s and len(s) >= 6}, key=len, reverse=True):
        cleaned = cleaned.replace(secret, "[redacted]")
    if len(cleaned) > 300:
        cleaned = cleaned[:300] + "..."
    return cleaned


def _line(platform: str, account: str, ok: bool, detail: str, secrets: list[str]) -> str:
    detail = _redact(detail, secrets).strip()
    status = "OK" if ok else "FAIL"
    if detail:
        return f"{platform} {account} {status} {detail}"
    return f"{platform} {account} {status}"


def _quote(value: str) -> str:
    escaped = value.replace("\\", "\\\\").replace('"', '\\"').replace("\r", " ").replace("\n", " ")
    return f'"{escaped}"'


def _has_scope(scopes: list[str], short: str) -> bool:
    needle = short.lower()
    for scope in scopes:
        text = scope.strip().lower()
        if text == needle or text.endswith("/" + needle):
            return True
    return False


def _youtube_probe(client_id: str, client_secret: str, refresh_token: str) -> tuple[str, list[str], bool]:
    """Refresh without downscoping, then ``channels.list(mine=true)``.

    ``scopes=None`` so the token response lists every granted scope. Requesting
    ``yt-analytics.readonly`` here would fail the refresh when that scope was
    never granted. Returns ``(title, scopes, scope_known)``.
    """
    from google.auth.transport.requests import Request
    from google.oauth2 import reauth
    from google.oauth2.credentials import Credentials
    from googleapiclient.discovery import build

    secrets = [client_id, client_secret, refresh_token]
    try:
        access_token, new_refresh, expiry, grant, _rapt = reauth.refresh_grant(
            Request(),
            TOKEN_URI,
            refresh_token,
            client_id,
            client_secret,
            scopes=None,
        )
        secrets.extend([access_token or "", new_refresh or ""])
        scope_known = isinstance(grant, dict) and "scope" in grant
        scopes = [s for s in ((grant or {}).get("scope") or "").split() if s] if isinstance(grant, dict) else []
        creds = Credentials(
            token=access_token,
            refresh_token=new_refresh or refresh_token,
            token_uri=TOKEN_URI,
            client_id=client_id,
            client_secret=client_secret,
            expiry=expiry,
        )
        youtube = build("youtube", "v3", credentials=creds, cache_discovery=False)
        # Read-only. videos.insert is the upload call and is never used here.
        response = youtube.channels().list(part="snippet", mine=True).execute()
    except Exception as exc:
        raise RuntimeError(_redact(exc, secrets)) from None

    items = (response or {}).get("items") or []
    title = ""
    if items:
        title = str(((items[0].get("snippet") or {}).get("title")) or "")
    return title, scopes, scope_known


def _http_json(method: str, url: str, *, token: str, **kwargs) -> tuple[int, dict]:
    secrets = [token]
    try:
        with httpx.Client(timeout=30.0, follow_redirects=False) as client:
            response = client.request(method, url, **kwargs)
        if response.status_code in {301, 302, 303, 307, 308}:
            raise RuntimeError(f"unexpected redirect HTTP {response.status_code}")
        try:
            body = response.json() if response.content else {}
        except Exception:
            body = {}
        if not isinstance(body, dict):
            body = {}
        if response.status_code >= 400:
            raise RuntimeError(_http_error(response.status_code, body, response.text or "", secrets))
        return response.status_code, body
    except RuntimeError:
        raise
    except Exception as exc:
        raise RuntimeError(_redact(exc, secrets)) from None


def _http_error(status: int, body: dict, raw: str, secrets: list[str]) -> str:
    err = body.get("error")
    if isinstance(err, dict):
        code = str(err.get("code") or "").strip()
        message = str(err.get("message") or "").strip()
        detail = " ".join(part for part in (code, message) if part)
        if detail:
            return _redact(detail, secrets)
    if raw.strip():
        return _redact(f"HTTP {status} {raw}", secrets)
    return f"HTTP {status}"


def check_youtube(settings, account: str) -> str:
    label = _account_label(account)
    values, missing = _missing(settings, _YOUTUBE_FIELDS, label)
    secrets = list(values.values())
    if missing:
        return _line("youtube", label, False, missing, secrets)
    try:
        title, scopes, scope_known = _youtube_probe(
            values["youtube_client_id"],
            values["youtube_client_secret"],
            values["youtube_refresh_token"],
        )
    except Exception as exc:
        return _line("youtube", label, False, str(exc), secrets)

    problems: list[str] = []
    if not scope_known:
        problems.append("token response missing scope")
    elif not _has_scope(scopes, _UPLOAD_SCOPE):
        problems.append(f"missing {_UPLOAD_SCOPE}")
    if not title:
        problems.append("no channel")
    bits = list(problems)
    if title:
        bits.append(f"title={_quote(title)}")
    if scope_known and _has_scope(scopes, _UPLOAD_SCOPE) and not problems:
        bits.append(f"{_UPLOAD_SCOPE}=yes")
    if scope_known and _has_scope(scopes, _ANALYTICS_SCOPE):
        bits.append(f"{_ANALYTICS_SCOPE}=yes")
    return _line("youtube", label, not problems, " ".join(bits), secrets)


def check_instagram(settings, account: str) -> str:
    label = _account_label(account)
    values, missing = _missing(settings, _INSTAGRAM_FIELDS, label)
    env_token = values["instagram_access_token"]
    stored = _stored_access_token(settings, "instagram", label, env_token, match_seed=True)
    token = stored or env_token
    if token:
        missing = _without_missing(missing, _env_name("instagram_access_token", label))
    secrets = [env_token, stored, token, values["instagram_ig_user_id"]]
    if missing:
        return _line("instagram", label, False, missing, secrets)
    ig_user_id = values["instagram_ig_user_id"]
    url = f"{_instagram_host(_instagram_auth(settings))}/{_instagram_version(settings)}/{ig_user_id}"
    try:
        _status, body = _http_json(
            "GET",
            url,
            token=token,
            headers={"Authorization": f"Bearer {token}"},
            params={"fields": "username,account_type"},
        )
    except Exception as exc:
        return _line("instagram", label, False, str(exc), secrets)
    if isinstance(body.get("error"), dict):
        return _line("instagram", label, False, _http_error(400, body, "", secrets), secrets)
    username = str(body.get("username") or "").strip()
    account_type = str(body.get("account_type") or "").strip()
    shown = account_type.upper()
    if not username:
        return _line("instagram", label, False, "username missing", secrets)
    if shown not in _IG_PROFESSIONAL:
        detail = f"account_type={shown or 'missing'} expected BUSINESS or CREATOR username={_quote(username)}"
        return _line("instagram", label, False, detail, secrets)
    return _line(
        "instagram",
        label,
        True,
        f"username={_quote(username)} account_type={shown}",
        secrets,
    )


def check_tiktok(settings, account: str) -> str:
    label = _account_label(account)
    values, missing = _missing(settings, _TIKTOK_FIELDS, label)
    env_token = values["tiktok_access_token"]
    stored = _stored_access_token(settings, "tiktok", label, env_token, match_seed=False)
    token = stored or env_token
    if token:
        missing = _without_missing(missing, _env_name("tiktok_access_token", label))
    secrets = [env_token, stored, token]
    if missing:
        return _line("tiktok", label, False, missing, secrets)
    url = f"{TIKTOK_OPEN}/v2/post/publish/creator_info/query/"
    try:
        _status, body = _http_json(
            "POST",
            url,
            token=token,
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json; charset=UTF-8",
            },
            content=b"{}",
        )
    except Exception as exc:
        return _line("tiktok", label, False, str(exc), secrets)
    err = body.get("error") or {}
    code = str(err.get("code") or "").strip() if isinstance(err, dict) else ""
    if code and code.lower() != "ok":
        message = str(err.get("message") or "").strip() if isinstance(err, dict) else ""
        detail = " ".join(part for part in (code, message) if part)
        return _line("tiktok", label, False, detail, secrets)
    data = body.get("data") or {}
    if not isinstance(data, dict):
        data = {}
    username = str(data.get("creator_username") or data.get("username") or "").strip()
    options = data.get("privacy_level_options")
    clean: list[str] = []
    if isinstance(options, list):
        clean = [str(item).strip() for item in options if str(item).strip()]
    problems: list[str] = []
    if not username:
        problems.append("creator_username missing")
    if not clean:
        problems.append("privacy_level_options missing")
    if problems:
        return _line("tiktok", label, False, " ".join(problems), secrets)
    joined = ",".join(clean)
    return _line(
        "tiktok",
        label,
        True,
        f"username={_quote(username)} privacy_level_options={joined}",
        secrets,
    )


_CHECKERS = {
    "youtube": check_youtube,
    "instagram": check_instagram,
    "tiktok": check_tiktok,
}


def check_platform(settings, platform: str, account: str) -> str:
    label = _account_label(account)
    name = (platform or "").strip().lower()
    checker = _CHECKERS.get(name)
    if checker is None:
        return _line(name or "unknown", label, False, "unknown platform", [])
    return checker(settings, label)


def run_checks(settings, platforms: tuple[str, ...], account: str) -> list[str]:
    return [check_platform(settings, platform, account) for platform in platforms]


def main(argv: list[str] | None = None, settings=None) -> int:
    parser = argparse.ArgumentParser(
        description="Read-only publish credential check. Never uploads or publishes.",
    )
    parser.add_argument(
        "--platform",
        default="all",
        choices=("all", *PLATFORMS),
        help="Platform to check (default: all)",
    )
    parser.add_argument(
        "--account",
        default="default",
        help="Account key (default: default). Non-default reads FIELD_<KEY> only.",
    )
    args = parser.parse_args(argv)
    if settings is None:
        from app.config import get_settings

        settings = get_settings()
    platforms = PLATFORMS if args.platform == "all" else (args.platform,)
    lines = run_checks(settings, platforms, args.account)
    failed = False
    for line in lines:
        print(line)
        # Account labels have no spaces, so field 3 is OK or FAIL even when
        # the detail quotes a channel title that contains the word FAIL.
        if line.split(" ", 3)[2:3] == ["FAIL"]:
            failed = True
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
