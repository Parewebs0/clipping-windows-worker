"""TikTok user-token refresh. Tokens live in oauth_state, never in .env or logs."""
from __future__ import annotations

import json
import os
import re
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import httpx

from app.utils.logging import get_logger

TOKEN_URL = "https://open.tiktokapis.com/v2/oauth/token/"
logger = get_logger("tiktok_oauth")
REFRESH_SKEW = timedelta(minutes=10)
_SLUG = re.compile(r"[^A-Za-z0-9._-]+")
_SAFE_CODE = re.compile(r"^[a-z][a-z0-9_]{0,63}$")


class TikTokOAuthStateError(RuntimeError):
    """The state file exists but cannot be trusted. Do not upload."""


def account_slug(account: str | None) -> str:
    """Stable file key. Matches ``_account_env`` (case and separators folded)."""
    slug = _SLUG.sub("_", str(account or "default")).strip("_.")
    return (slug or "default").lower()


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


class TikTokOAuthStore:
    """``<working_directory>/oauth_state/tiktok_<account>.json``, mode 600."""

    def __init__(self, root: Path | str) -> None:
        self.root = Path(root)

    def path_for(self, account: str | None) -> Path:
        return self.root / f"tiktok_{account_slug(account)}.json"

    def load(self, account: str | None) -> dict[str, Any] | None:
        path = self.path_for(account)
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            # JSONDecodeError quotes a snippet of the file. Do not chain it:
            # that snippet is the token.
            raise TikTokOAuthStateError(
                f"unreadable tiktok oauth state for account {account_slug(account)}"
            ) from None
        if not isinstance(data, dict):
            raise TikTokOAuthStateError(
                f"unreadable tiktok oauth state for account {account_slug(account)}"
            )
        return data

    def save(self, account: str | None, payload: dict[str, Any]) -> Path:
        """Atomic write (tmp + fsync + replace), then mode 600. Same idea as publish_state."""
        self.root.mkdir(parents=True, exist_ok=True)
        try:
            os.chmod(self.root, 0o700)
        except OSError:
            pass
        path = self.path_for(account)
        fd, tmp = tempfile.mkstemp(dir=self.root, prefix=path.name, suffix=".tmp")
        try:
            os.chmod(tmp, 0o600)
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(payload, fh, ensure_ascii=False, indent=2)
                fh.flush()
                os.fsync(fh.fileno())
            os.chmod(tmp, 0o600)
            os.replace(tmp, path)
            os.chmod(path, 0o600)
        except BaseException:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise
        return path


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


def resolve_access_token(
    *,
    account: str,
    access_token: str,
    client_key: str,
    client_secret: str,
    refresh_token: str,
    oauth_state_dir: Path | None,
    client: httpx.Client,
    now: datetime | None = None,
) -> tuple[str, list[str]]:
    """Return a bearer token and every secret that must be scrubbed from errors.

    A static ``TIKTOK_ACCESS_TOKEN`` with no known expiry is used as-is.
    A cached token that expires in under 10 minutes is refreshed once.
    The new access token, refresh token and expiries are written to
    ``oauth_state`` and are never returned except as the bearer itself.
    """
    moment = now or datetime.now(timezone.utc)
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)

    store = TikTokOAuthStore(oauth_state_dir) if oauth_state_dir is not None else None
    cached = store.load(account) if store is not None else None
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
    needs_refresh = (not access) or (
        expires_at is not None and expires_at - moment < REFRESH_SKEW
    )

    secrets = known_secrets(access, env_access, cached_access, refresh, env_refresh, cached_refresh, secret, key)

    if needs_refresh and can_refresh:
        if store is None:
            raise RuntimeError("tiktok oauth state directory missing") from None
        fresh = _refresh(client, client_key=key, client_secret=secret, refresh_token=refresh)
        access_expires = moment + timedelta(seconds=int(fresh.get("expires_in") or 86400))
        refresh_in = fresh.get("refresh_expires_in")
        new_refresh = str(fresh.get("refresh_token") or refresh).strip()
        new_access = str(fresh["access_token"]).strip()
        payload: dict[str, Any] = {
            "account": account_slug(account),
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
        store.save(account, payload)
        logger.info("tiktok access token refreshed", account=account_slug(account))
        secrets = known_secrets(*secrets, new_access, new_refresh)
        return new_access, secrets

    if needs_refresh and not access:
        raise RuntimeError("TIKTOK_ACCESS_TOKEN missing") from None
    if needs_refresh and expires_at is not None:
        raise RuntimeError(
            "tiktok token refresh failed: missing TIKTOK_CLIENT_KEY, "
            "TIKTOK_CLIENT_SECRET or TIKTOK_REFRESH_TOKEN"
        ) from None
    if not access:
        raise RuntimeError("TIKTOK_ACCESS_TOKEN missing") from None
    return access, secrets


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
