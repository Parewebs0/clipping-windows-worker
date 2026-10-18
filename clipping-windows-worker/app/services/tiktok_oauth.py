"""TikTok user-token refresh. Tokens live in oauth_state, never in .env or logs.

The file itself is the shared store from ``oauth_state.py`` (one JSON per
provider and account). This module only decides when to refresh.
"""
from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import httpx

from app.services.oauth_state import load_oauth_state, oauth_state_path, save_oauth_state, slug
from app.utils.logging import get_logger

TOKEN_URL = "https://open.tiktokapis.com/v2/oauth/token/"
logger = get_logger("tiktok_oauth")
REFRESH_SKEW = timedelta(minutes=10)
_SAFE_CODE = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
_PROVIDER = "tiktok"


class TikTokOAuthStateError(RuntimeError):
    """The state file exists but cannot be trusted. Do not upload."""


def known_secrets(*values: object) -> list[str]:
    """Long credential strings that must never appear in an error or a log line."""
    found: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value or "").strip()
        if len(text) < 8 or text in seen:
            continue
        seen.add(text)
        found.append(text)
    return found


def scrub(text: str, secrets: list[str]) -> str:
    out = text
    for secret in secrets:
        if secret:
            out = out.replace(secret, "[redacted]")
    return out


def _account(account: str | None) -> str:
    return slug(str(account or ""), fallback="default")


def _parse_dt(value: object) -> datetime | None:
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def _safe_error_code(body: dict[str, Any]) -> str:
    err = body.get("error")
    code = err if isinstance(err, str) else ""
    if isinstance(err, dict):
        code = str(err.get("code") or "")
    code = code.strip()
    if not _SAFE_CODE.fullmatch(code):
        return ""
    return code


def _refresh_failed(body: dict[str, Any] | None = None) -> RuntimeError:
    code = _safe_error_code(body or {})
    # Code only. error_description sometimes echoes the refresh token.
    logger.error("tiktok token refresh failed", error_code=code or "unknown")
    if code:
        return RuntimeError(f"tiktok token refresh failed: {code}")
    return RuntimeError("tiktok token refresh failed")


def _json_object(response: httpx.Response) -> dict[str, Any]:
    try:
        body = response.json()
    except ValueError:
        return {}
    return body if isinstance(body, dict) else {}


def _load_cached(working_directory: Path, account: str) -> dict[str, Any] | None:
    """Load the shared store. A present but unreadable file is fatal.

    ``load_oauth_state`` turns a broken file into None so other platforms can
    fall back to the env token. TikTok must not: the file may hold the rotated
    refresh token, and the exception must not quote the file (it is the token).
    """
    path = oauth_state_path(working_directory, _PROVIDER, account)
    if not path.is_file():
        return None
    cached = load_oauth_state(working_directory, _PROVIDER, account)
    if cached is None:
        raise TikTokOAuthStateError(
            f"unreadable tiktok oauth state for account {_account(account)}"
        ) from None
    return cached


def resolve_access_token(
    *,
    account: str,
    access_token: str,
    client_key: str,
    client_secret: str,
    refresh_token: str,
    working_directory: Path | None,
    client: httpx.Client,
    now: datetime | None = None,
    force: bool = False,
) -> tuple[str, list[str], bool, bool]:
    """Return ``(access_token, secrets, refreshed, can_refresh)``.

    Refresh once when client key, secret and a refresh token are available and
    any of these is true: there is no access token, the expiry is unknown
    (a static ``TIKTOK_ACCESS_TOKEN`` with no saved file), the cached expiry
    is inside 10 minutes, or the caller got a 401 (``force``).

    An access token with no refresh material is used as-is. The new access
    token, refresh token and expiries are written through ``oauth_state`` and
    are never logged.
    """
    moment = now or datetime.now(timezone.utc)
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)

    name = _account(account)
    cached = _load_cached(working_directory, name) if working_directory is not None else None
    cached_access = str((cached or {}).get("access_token") or "").strip()
    cached_refresh = str((cached or {}).get("refresh_token") or "").strip()
    env_access = (access_token or "").strip()
    env_refresh = (refresh_token or "").strip()
    key = (client_key or "").strip()
    secret = (client_secret or "").strip()

    # The cached access token is the rotated one. Prefer it over the env copy.
    access = cached_access or env_access
    refresh = cached_refresh or env_refresh
    expires_at = _parse_dt((cached or {}).get("access_expires_at")) if cached_access else None
    can_refresh = bool(key and secret and refresh)
    unknown_expiry = expires_at is None
    within_skew = expires_at is not None and expires_at - moment < REFRESH_SKEW
    needs_refresh = (not access) or (can_refresh and (force or unknown_expiry or within_skew))

    secrets = known_secrets(access, env_access, cached_access, refresh, env_refresh, cached_refresh, secret, key)

    if needs_refresh and can_refresh:
        if working_directory is None:
            raise RuntimeError("tiktok oauth state directory missing") from None
        fresh = _refresh(client, client_key=key, client_secret=secret, refresh_token=refresh)
        access_expires = moment + timedelta(seconds=int(fresh.get("expires_in") or 86400))
        refresh_in = fresh.get("refresh_expires_in")
        new_refresh = str(fresh.get("refresh_token") or refresh).strip()
        new_access = str(fresh["access_token"]).strip()
        payload: dict[str, Any] = {
            "account": name,
            "access_token": new_access,
            "refresh_token": new_refresh,
            "access_expires_at": access_expires.isoformat(),
            "updated_at": moment.isoformat(),
        }
        if refresh_in not in (None, ""):
            try:
                payload["refresh_expires_at"] = (moment + timedelta(seconds=int(refresh_in))).isoformat()
            except (TypeError, ValueError):
                pass
        if fresh.get("open_id"):
            payload["open_id"] = fresh["open_id"]
        if fresh.get("scope"):
            payload["scope"] = fresh["scope"]
        save_oauth_state(working_directory, _PROVIDER, name, payload)
        logger.info("tiktok access token refreshed", account=name)
        secrets = known_secrets(*secrets, new_access, new_refresh)
        return new_access, secrets, True, True

    if needs_refresh and not access:
        raise RuntimeError("TIKTOK_ACCESS_TOKEN missing") from None
    if needs_refresh and expires_at is not None:
        raise RuntimeError(
            "tiktok token refresh failed: missing TIKTOK_CLIENT_KEY, "
            "TIKTOK_CLIENT_SECRET or TIKTOK_REFRESH_TOKEN"
        ) from None
    if not access:
        raise RuntimeError("TIKTOK_ACCESS_TOKEN missing") from None
    return access, secrets, False, can_refresh


def _refresh(
    client: httpx.Client,
    *,
    client_key: str,
    client_secret: str,
    refresh_token: str,
) -> dict[str, Any]:
    try:
        response = client.post(
            TOKEN_URL,
            headers={"Cache-Control": "no-cache"},
            data={
                "client_key": client_key,
                "client_secret": client_secret,
                "grant_type": "refresh_token",
                "refresh_token": refresh_token,
            },
        )
    except httpx.HTTPError:
        raise _refresh_failed() from None
    body = _json_object(response)
    err = body.get("error")
    failed = response.status_code >= 400 or (isinstance(err, str) and err) or not body.get("access_token")
    if failed:
        raise _refresh_failed(body) from None
    return body
