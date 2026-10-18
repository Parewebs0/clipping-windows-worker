"""Instagram Reels via Graph API resumable upload. Tokens from Worker .env.

Long-lived tokens last about 60 days. Before an upload we refresh when the
known expiry is under 7 days, or when the expiry is unknown. The renewed
token and its expiry are stored in ``<working_directory>/oauth_state/``.
"""
from __future__ import annotations

import logging
import os
import re
import time
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import httpx

from app.services.oauth_state import load_oauth_state, save_oauth_state, slug
from app.utils.logging import get_logger

GRAPH_FACEBOOK = "https://graph.facebook.com"
GRAPH_INSTAGRAM = "https://graph.instagram.com"
DEFAULT_GRAPH_VERSION = "v21.0"
REFRESH_WITHIN = timedelta(days=7)
_VERSION_RE = re.compile(r"v\d+(?:\.\d+)?")

logger = get_logger("instagram_upload")


class InstagramTokenRefreshError(RuntimeError):
    """Refresh failed. ``str(self)`` is safe to log (no token, no secret)."""


def _utcnow(now: datetime | None) -> datetime:
    current = now or datetime.now(timezone.utc)
    if current.tzinfo is None:
        current = current.replace(tzinfo=timezone.utc)
    return current


def resolve_auth(explicit: str | None) -> str:
    """``facebook`` (default, linked Page) or ``instagram`` (Instagram Login)."""
    if explicit is not None and str(explicit).strip():
        raw = str(explicit).strip().lower()
    else:
        raw = os.environ.get("INSTAGRAM_AUTH", "").strip().lower()
    if raw in {"facebook", "instagram"}:
        return raw
    return "facebook"


def resolve_graph_version(explicit: str | None) -> str:
    if explicit is not None and str(explicit).strip():
        raw = str(explicit).strip()
    else:
        raw = os.environ.get("INSTAGRAM_GRAPH_VERSION", "").strip()
    if _VERSION_RE.fullmatch(raw):
        return raw
    return DEFAULT_GRAPH_VERSION


def graph_host(auth: str) -> str:
    return GRAPH_INSTAGRAM if auth == "instagram" else GRAPH_FACEBOOK


def _optional_setting(explicit: str | None, env_name: str) -> str:
    if explicit is None:
        return os.environ.get(env_name, "").strip()
    return str(explicit).strip()


def _parse_expires(value: object) -> datetime | None:
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        parsed = datetime.fromisoformat(value.strip())
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def _needs_refresh(expires_at: datetime | None, now: datetime) -> bool:
    if expires_at is None:
        return True
    return expires_at - now < REFRESH_WITHIN


def _json_dict(resp: httpx.Response) -> dict[str, Any]:
    try:
        data = resp.json()
    except (ValueError, TypeError):
        raise InstagramTokenRefreshError("invalid json") from None
    if not isinstance(data, dict):
        raise InstagramTokenRefreshError("invalid json")
    return data


def _expires_in_seconds(value: object) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        seconds = int(value)
    except (TypeError, ValueError):
        return None
    if seconds <= 0:
        return None
    return seconds


def _public_error(exc: BaseException) -> str:
    """Error text that is safe to log. Never the request URL or a token."""
    if isinstance(exc, InstagramTokenRefreshError):
        return str(exc)
    if isinstance(exc, httpx.HTTPStatusError):
        return f"HTTP {exc.response.status_code}"
    return type(exc).__name__


def _warn_refresh_failed(
    *,
    account: str,
    auth: str,
    expires_at: datetime | None,
    now: datetime,
    exc: BaseException,
) -> None:
    remaining_days: float | None
    if expires_at is None:
        remaining_days = None
    else:
        remaining_days = round((expires_at - now).total_seconds() / 86400, 3)
    logger.warning(
        "instagram access token refresh failed; uploading with the current token",
        account=account,
        auth=auth,
        expires_in_days=remaining_days,
        error=_public_error(exc),
    )


@contextmanager
def _silence_http_logs():
    """Token requests carry the secret in the query string. httpx logs that URL at INFO."""
    loggers = [logging.getLogger(name) for name in ("httpx", "httpcore")]
    previous = [item.level for item in loggers]
    for item in loggers:
        item.setLevel(logging.WARNING)
    try:
        yield
    finally:
        for item, level in zip(loggers, previous):
            item.setLevel(level)


def _get(client: httpx.Client, url: str, params: dict[str, str]) -> httpx.Response:
    try:
        with _silence_http_logs():
            return client.get(url, params=params)
    except httpx.HTTPError as exc:
        raise InstagramTokenRefreshError(type(exc).__name__) from None


def _debug_expires_at(
    client: httpx.Client,
    *,
    token: str,
    app_id: str,
    app_secret: str,
    version: str,
    now: datetime,
) -> datetime | None:
    """Facebook ``debug_token``. None when expiry cannot be read (not a warning)."""
    if not app_id or not app_secret or not token:
        return None
    url = f"{GRAPH_FACEBOOK}/{version}/debug_token"
    try:
        resp = _get(
            client,
            url,
            {"input_token": token, "access_token": f"{app_id}|{app_secret}"},
        )
    except InstagramTokenRefreshError:
        return None
    if resp.status_code >= 400:
        return None
    try:
        data = _json_dict(resp).get("data")
    except InstagramTokenRefreshError:
        return None
    if not isinstance(data, dict) or "expires_at" not in data:
        return None
    raw = data.get("expires_at")
    try:
        stamp = int(raw)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    # 0 means the token does not expire (typical long-lived Page token).
    if stamp == 0:
        return now + timedelta(days=3650)
    return datetime.fromtimestamp(stamp, tz=timezone.utc)


def _refresh_facebook(
    client: httpx.Client,
    *,
    token: str,
    app_id: str,
    app_secret: str,
    version: str,
) -> tuple[str, int | None]:
    if not app_id or not app_secret:
        raise InstagramTokenRefreshError("INSTAGRAM_APP_ID or INSTAGRAM_APP_SECRET unset")
    url = f"{GRAPH_FACEBOOK}/{version}/oauth/access_token"
    resp = _get(
        client,
        url,
        {
            "grant_type": "fb_exchange_token",
            "client_id": app_id,
            "client_secret": app_secret,
            "fb_exchange_token": token,
        },
    )
    if resp.status_code >= 400:
        raise InstagramTokenRefreshError(f"HTTP {resp.status_code}")
    body = _json_dict(resp)
    renewed = str(body.get("access_token") or "").strip()
    if not renewed:
        raise InstagramTokenRefreshError("missing access_token")
    return renewed, _expires_in_seconds(body.get("expires_in"))


def _refresh_instagram(client: httpx.Client, *, token: str) -> tuple[str, int | None]:
    url = f"{GRAPH_INSTAGRAM}/refresh_access_token"
    resp = _get(client, url, {"grant_type": "ig_refresh_token", "access_token": token})
    if resp.status_code >= 400:
        raise InstagramTokenRefreshError(f"HTTP {resp.status_code}")
    body = _json_dict(resp)
    renewed = str(body.get("access_token") or "").strip()
    if not renewed:
        raise InstagramTokenRefreshError("missing access_token")
    return renewed, _expires_in_seconds(body.get("expires_in"))


def _persist(
    working_directory: Path | str,
    account: str,
    token: str,
    expires_at: datetime | None,
    now: datetime,
) -> None:
    """Save the token. A disk error must not abort the upload."""
    try:
        save_oauth_state(
            working_directory,
            "instagram",
            account,
            {
                "provider": "instagram",
                "account": account,
                "access_token": token,
                "expires_at": expires_at.astimezone(timezone.utc).isoformat() if expires_at else None,
                "updated_at": now.astimezone(timezone.utc).isoformat(),
            },
        )
    except Exception as exc:  # noqa: BLE001 — the upload can still use the in-memory token
        logger.warning(
            "instagram oauth state was not saved",
            account=account,
            error=_public_error(exc),
        )


def prepare_instagram_token(
    *,
    access_token: str,
    account_key: str = "default",
    working_directory: Path | str | None = None,
    now: datetime | None = None,
    auth: str | None = None,
    app_id: str | None = None,
    app_secret: str | None = None,
    graph_version: str | None = None,
    client: httpx.Client | None = None,
) -> str:
    """Return the token an upload should use, refreshing it when required.

    A persisted token wins over ``access_token`` (the env value is the seed;
    the file holds the last renewal). When no expiry is known and
    ``auth=facebook``, expiry is read with ``debug_token`` before deciding.
    A failed refresh is logged at WARNING without secrets and the current
    token is returned so the upload still runs.
    """
    current = _utcnow(now)
    mode = resolve_auth(auth)
    version = resolve_graph_version(graph_version)
    account = slug(account_key, fallback="default")
    resolved_app_id = _optional_setting(app_id, "INSTAGRAM_APP_ID")
    resolved_app_secret = _optional_setting(app_secret, "INSTAGRAM_APP_SECRET")
    env_token = (access_token or "").strip()

    token = env_token
    expires_at: datetime | None = None
    if working_directory is not None:
        stored = load_oauth_state(working_directory, "instagram", account)
        if stored:
            stored_token = str(stored.get("access_token") or "").strip()
            if stored_token:
                token = stored_token
            expires_at = _parse_expires(stored.get("expires_at"))

    if not token:
        return ""

    owns_client = client is None
    http = client or httpx.Client(timeout=30.0)
    try:
        if expires_at is None and mode == "facebook":
            learned = _debug_expires_at(
                http,
                token=token,
                app_id=resolved_app_id,
                app_secret=resolved_app_secret,
                version=version,
                now=current,
            )
            if learned is not None:
                expires_at = learned
                if working_directory is not None and not _needs_refresh(expires_at, current):
                    _persist(working_directory, account, token, expires_at, current)

        if not _needs_refresh(expires_at, current):
            return token

        try:
            if mode == "instagram":
                renewed, expires_in = _refresh_instagram(http, token=token)
            else:
                renewed, expires_in = _refresh_facebook(
                    http,
                    token=token,
                    app_id=resolved_app_id,
                    app_secret=resolved_app_secret,
                    version=version,
                )
        except Exception as exc:  # noqa: BLE001 — upload must proceed with the current token
            _warn_refresh_failed(account=account, auth=mode, expires_at=expires_at, now=current, exc=exc)
            return token

        new_expires: datetime | None = None
        if expires_in is not None:
            new_expires = current + timedelta(seconds=expires_in)
        elif mode == "facebook":
            new_expires = _debug_expires_at(
                http,
                token=renewed,
                app_id=resolved_app_id,
                app_secret=resolved_app_secret,
                version=version,
                now=current,
            )
        if working_directory is not None:
            _persist(working_directory, account, renewed, new_expires, current)
        return renewed
    finally:
        if owns_client:
            http.close()


def _raise_http(resp: httpx.Response, step: str) -> None:
    body = (resp.text or "")[:800]
    raise RuntimeError(f"instagram {step} HTTP {resp.status_code}: {body}")


def upload_reel(
    *,
    file_path: Path,
    caption: str,
    access_token: str,
    ig_user_id: str,
    account_key: str = "default",
    working_directory: Path | str | None = None,
    now: datetime | None = None,
    auth: str | None = None,
    app_id: str | None = None,
    app_secret: str | None = None,
    graph_version: str | None = None,
) -> dict:
    if not file_path.exists():
        raise FileNotFoundError(str(file_path))

    mode = resolve_auth(auth)
    version = resolve_graph_version(graph_version)
    host = graph_host(mode)

    with httpx.Client(timeout=300.0, follow_redirects=True) as client:
        access_token = prepare_instagram_token(
            access_token=access_token,
            account_key=account_key,
            working_directory=working_directory,
            now=now,
            auth=mode,
            app_id=app_id,
            app_secret=app_secret,
            graph_version=version,
            client=client,
        )
        if not access_token or not ig_user_id:
            raise RuntimeError("INSTAGRAM_ACCESS_TOKEN / INSTAGRAM_IG_USER_ID missing")

        size = file_path.stat().st_size
        raw = file_path.read_bytes()
        if len(raw) != size:
            raise RuntimeError("instagram file size mismatch")

        headers_auth = {"Authorization": f"Bearer {access_token}"}
        create = client.post(
            f"{host}/{version}/{ig_user_id}/media",
            headers=headers_auth,
            data={
                "media_type": "REELS",
                "upload_type": "resumable",
                "caption": (caption or "")[:2200],
                "share_to_feed": "true",
            },
        )
        if create.status_code >= 400:
            _raise_http(create, "create_container")
        created = create.json()
        container_id = created.get("id")
        if not container_id:
            raise RuntimeError(f"instagram container missing id: {created}")

        uris: list[str] = []
        if created.get("uri"):
            uris.append(created["uri"])
        fallback = f"https://rupload.facebook.com/ig-api-upload/{version}/{container_id}"
        if fallback not in uris:
            uris.append(fallback)

        last_err = None
        uploaded = False
        for uri in uris:
            for auth_header in (
                f"OAuth {access_token}",
                f"Bearer {access_token}",
            ):
                up = client.post(
                    uri,
                    headers={
                        "Authorization": auth_header,
                        "offset": "0",
                        "file_size": str(size),
                    },
                    content=raw,
                )
                if up.status_code < 400:
                    uploaded = True
                    break
                last_err = f"{uri} {up.status_code} {(up.text or '')[:400]}"
            if uploaded:
                break
        if not uploaded:
            raise RuntimeError(f"instagram rupload failed: {last_err}")

        status_code = "IN_PROGRESS"
        payload: dict = {}
        for _ in range(40):
            st = client.get(
                f"{host}/{version}/{container_id}",
                headers=headers_auth,
                params={"fields": "status_code,status"},
            )
            if st.status_code >= 400:
                _raise_http(st, "status")
            payload = st.json()
            status_code = (payload.get("status_code") or "").upper()
            if status_code in {"FINISHED", "PUBLISHED"}:
                break
            if status_code in {"ERROR", "EXPIRED"}:
                raise RuntimeError(f"instagram container {status_code}: {payload}")
            time.sleep(3)
        if status_code not in {"FINISHED", "PUBLISHED"}:
            raise TimeoutError(f"instagram container not ready: {status_code} {payload}")

        if status_code != "PUBLISHED":
            pub = client.post(
                f"{host}/{version}/{ig_user_id}/media_publish",
                headers=headers_auth,
                data={"creation_id": container_id},
            )
            if pub.status_code >= 400:
                _raise_http(pub, "media_publish")
            media_id = pub.json().get("id") or container_id
        else:
            media_id = container_id

        permalink = None
        info = client.get(
            f"{host}/{version}/{media_id}",
            headers=headers_auth,
            params={"fields": "permalink,shortcode"},
        )
        if info.status_code == 200:
            permalink = info.json().get("permalink")

    return {
        "media_id": media_id,
        "container_id": container_id,
        "post_url": permalink or f"https://www.instagram.com/reel/{media_id}/",
    }
