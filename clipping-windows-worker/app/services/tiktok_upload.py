"""TikTok Content Posting API.

``TIKTOK_MODE=inbox`` (default) sends a draft with scope ``video.upload``.
``TIKTOK_MODE=direct`` posts with scope ``video.publish``: creator info, then
init, chunked upload and status until ``PUBLISH_COMPLETE`` or ``FAILED``.

Access tokens last about a day. When client key, secret and a refresh token
are set, the token is refreshed once if its expiry is unknown or inside 10
minutes, and a 401 refreshes once more and retries that request. The new
token is stored via ``oauth_state`` (mode 600). An access token with no
refresh material is used unchanged.

A ``publish_id`` is written under ``tiktok_pending/`` after the bytes are
uploaded. A retry polls that id instead of uploading again, so a status
timeout cannot create a second direct post.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any, Callable, Iterator
from urllib.parse import parse_qs, urlsplit

import httpx

from app.services.oauth_state import slug
from app.services.tiktok_oauth import known_secrets, resolve_access_token, scrub

OPEN = "https://open.tiktokapis.com"
INBOX_DONE = {"PUBLISH_COMPLETE", "FAILED", "SEND_TO_USER_INBOX"}
DIRECT_DONE = {"PUBLISH_COMPLETE", "FAILED"}
# TikTok mixes a 4_194_304-byte "whole file" example with 10_000_000-byte
# chunks. One chunk only up to 64_000_000 bytes (both readings of "64 MB").
# Above that, 10_000_000-byte chunks; the final chunk absorbs the remainder
# and may be larger than chunk_size, up to 128 MiB. At most 1000 chunks.
_MAX_SINGLE = 64_000_000
_CHUNK = 10_000_000
_MIN_CHUNK = 5_000_000
_MAX_FINAL = 128 * 1024 * 1024
_TITLE_LIMIT = 2200


def plan_upload_chunks(size: int) -> tuple[int, int]:
    """Return ``(chunk_size, total_chunk_count)`` for ``source=FILE_UPLOAD``.

    ``total_chunk_count`` is ``floor(video_size / chunk_size)``. The last PUT
    carries whatever is left, so it can be larger than ``chunk_size``.
    """
    if size <= 0:
        raise RuntimeError("tiktok video file is empty")
    if size <= _MAX_SINGLE:
        return size, 1
    chunk_size = _CHUNK
    count = size // chunk_size
    if count > 1000:
        chunk_size = max(_MIN_CHUNK, (size + 999) // 1000)
        if chunk_size > _MAX_SINGLE:
            raise RuntimeError("tiktok video is too large to upload in 1000 chunks")
        count = size // chunk_size
    if count < 2:
        raise RuntimeError("tiktok video cannot be split within chunk limits")
    last = size - (count - 1) * chunk_size
    if last <= 0 or last > _MAX_FINAL:
        raise RuntimeError("tiktok chunk plan exceeds the final-chunk limit")
    return chunk_size, count


def iter_chunk_ranges(size: int, chunk_size: int, count: int) -> Iterator[tuple[int, int]]:
    """``(offset, length)`` pairs that cover ``size`` exactly."""
    if count < 1 or chunk_size <= 0:
        raise RuntimeError("tiktok chunk plan is empty")
    offset = 0
    for index in range(count):
        length = size - offset if index == count - 1 else chunk_size
        if length <= 0:
            raise RuntimeError("tiktok chunk plan is empty")
        yield offset, length
        offset += length
    if offset != size:
        raise RuntimeError("tiktok chunk plan does not cover the file")


def build_post_url(username: str | None, post_id: str | None) -> str:
    """``/@{username}/video/{post_id}``, or the profile when the id is missing."""
    user = _username(username)
    if not user:
        return ""
    post = (post_id or "").strip()
    if post.isalnum():
        return f"https://www.tiktok.com/@{user}/video/{post}"
    return f"https://www.tiktok.com/@{user}"


class _Auth:
    """Bearer for one upload. A 401 may refresh the token a single time."""

    def __init__(
        self,
        token: str,
        secrets: list[str],
        *,
        can_refresh: bool,
        refreshed: bool,
        refresher: Callable[[], tuple[str, list[str], bool, bool]],
    ) -> None:
        self.token = token
        self.secrets = secrets
        self.can_refresh = can_refresh
        self.refreshed = refreshed
        self._refresher = refresher

    def headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json; charset=UTF-8",
        }

    def retry_unauthorized(self) -> bool:
        if self.refreshed or not self.can_refresh:
            return False
        token, secrets, _refreshed, can_refresh = self._refresher()
        self.token = token
        self.secrets = secrets
        self.can_refresh = can_refresh
        self.refreshed = True
        return True


def upload_video(
    *,
    file_path: Path,
    title: str,
    access_token: str,
    privacy_level: str = "SELF_ONLY",
    mode: str = "inbox",
    paid_promotion: bool = False,
    handle: str | None = None,
    account: str = "default",
    client_key: str = "",
    client_secret: str = "",
    refresh_token: str = "",
    working_directory: Path | None = None,
    clip_id: str | None = None,
    duration_sec: float | None = None,
    transport: httpx.BaseTransport | None = None,
    poll_interval: float = 5.0,
    max_polls: int = 90,
) -> dict:
    if not file_path.exists():
        raise FileNotFoundError(str(file_path))
    post_mode = (mode or "inbox").strip().lower()
    if post_mode not in {"inbox", "direct"}:
        raise RuntimeError("TIKTOK_MODE must be inbox or direct")

    caption = _clip_title(title)
    size = file_path.stat().st_size
    if size <= 0:
        raise RuntimeError("tiktok video file is empty")
    pending = _load_pending(working_directory, account, clip_id)
    # TikTok rejects a paid brand-content post that is only visible to the creator.
    # Check before any request so a retry does not init a publish that will fail.
    if pending is None and post_mode == "direct" and paid_promotion and privacy_level == "SELF_ONLY":
        raise RuntimeError(
            "tiktok paid promotion cannot use privacy_level SELF_ONLY "
            "(brand_content_toggle is rejected); pick another privacy_level"
        )
    client_kwargs: dict[str, Any] = {"timeout": 300.0, "follow_redirects": True}
    if transport is not None:
        client_kwargs["transport"] = transport
    with httpx.Client(**client_kwargs) as client:
        token, secrets, refreshed, can_refresh = resolve_access_token(
            account=account,
            access_token=access_token,
            client_key=client_key,
            client_secret=client_secret,
            refresh_token=refresh_token,
            working_directory=working_directory,
            client=client,
        )

        def _force() -> tuple[str, list[str], bool, bool]:
            return resolve_access_token(
                account=account,
                access_token=access_token,
                client_key=client_key,
                client_secret=client_secret,
                refresh_token=refresh_token,
                working_directory=working_directory,
                client=client,
                force=True,
            )

        auth = _Auth(
            token,
            secrets,
            can_refresh=can_refresh,
            refreshed=refreshed,
            refresher=_force,
        )
        if pending is not None:
            return _resume(
                client,
                auth,
                pending,
                size=size,
                privacy_level=privacy_level,
                title=caption,
                poll_interval=poll_interval,
                max_polls=max_polls,
                working_directory=working_directory,
                account=account,
                clip_id=clip_id,
            )
        if post_mode == "direct":
            if duration_sec is None:
                duration_sec = _probe_duration(file_path)
            return _direct(
                client,
                auth,
                file_path=file_path,
                size=size,
                title=caption,
                privacy_level=privacy_level,
                paid_promotion=paid_promotion,
                handle=handle,
                duration_sec=duration_sec,
                poll_interval=poll_interval,
                max_polls=max_polls,
                working_directory=working_directory,
                account=account,
                clip_id=clip_id,
            )
        return _inbox(
            client,
            auth,
            file_path=file_path,
            size=size,
            title=caption,
            privacy_level=privacy_level,
            handle=handle,
            poll_interval=poll_interval,
            max_polls=max_polls,
            working_directory=working_directory,
            account=account,
            clip_id=clip_id,
        )


def _direct(
    client: httpx.Client,
    auth: _Auth,
    *,
    file_path: Path,
    size: int,
    title: str,
    privacy_level: str,
    paid_promotion: bool,
    handle: str | None,
    duration_sec: float | None,
    poll_interval: float,
    max_polls: int,
    working_directory: Path | None,
    account: str,
    clip_id: str | None,
) -> dict:
    info = _post_json(
        client,
        f"{OPEN}/v2/post/publish/creator_info/query/",
        auth,
        {},
        "tiktok creator_info",
    )
    creator = info.get("data") or {}
    options = creator.get("privacy_level_options")
    if not isinstance(options, list) or not options:
        raise RuntimeError("tiktok creator_info missing privacy_level_options")
    if privacy_level not in options:
        raise RuntimeError(
            f"tiktok privacy_level {privacy_level!r} is not in privacy_level_options {options}"
        )
    max_sec = creator.get("max_video_post_duration_sec")
    if duration_sec is not None and max_sec is not None:
        try:
            if float(duration_sec) > float(max_sec):
                raise RuntimeError(
                    f"tiktok video duration {duration_sec}s exceeds "
                    f"max_video_post_duration_sec {max_sec}"
                )
        except (TypeError, ValueError):
            pass

    chunk_size, count = plan_upload_chunks(size)
    init = _post_json(
        client,
        f"{OPEN}/v2/post/publish/video/init/",
        auth,
        {
            "post_info": {
                "title": title,
                "privacy_level": privacy_level,
                "disable_comment": False,
                # Account-level switches: a false here is rejected when the
                # creator has duets or stitches turned off.
                "disable_duet": bool(creator.get("duet_disabled")),
                "disable_stitch": bool(creator.get("stitch_disabled")),
                "brand_content_toggle": bool(paid_promotion),
            },
            "source_info": {
                "source": "FILE_UPLOAD",
                "video_size": size,
                "chunk_size": chunk_size,
                "total_chunk_count": count,
            },
        },
        "tiktok direct init",
    )
    publish_id, upload_url = _upload_target(init, auth.secrets, "tiktok direct missing upload_url")
    _put_chunks(client, file_path, upload_url, size, chunk_size, count, auth.secrets)
    username = _username(creator.get("creator_username")) or _username(handle)
    _save_pending(
        working_directory,
        account,
        clip_id,
        publish_id=publish_id,
        mode="direct",
        privacy_level=privacy_level,
        title=title,
        username=username,
    )
    status, payload = _poll(
        client,
        auth,
        publish_id,
        done=DIRECT_DONE,
        poll_interval=poll_interval,
        max_polls=max_polls,
    )
    return _conclude(
        mode="direct",
        status=status,
        payload=payload,
        publish_id=publish_id,
        secrets=auth.secrets,
        privacy_level=privacy_level,
        title=title,
        username=username,
        size=size,
        working_directory=working_directory,
        account=account,
        clip_id=clip_id,
    )


def _inbox(
    client: httpx.Client,
    auth: _Auth,
    *,
    file_path: Path,
    size: int,
    title: str,
    privacy_level: str,
    handle: str | None,
    poll_interval: float,
    max_polls: int,
    working_directory: Path | None,
    account: str,
    clip_id: str | None,
) -> dict:
    # Inbox ignores privacy_level. The value is only echoed in the result.
    chunk_size, count = plan_upload_chunks(size)
    init = _post_json(
        client,
        f"{OPEN}/v2/post/publish/inbox/video/init/",
        auth,
        {
            "source_info": {
                "source": "FILE_UPLOAD",
                "video_size": size,
                "chunk_size": chunk_size,
                "total_chunk_count": count,
            }
        },
        "tiktok inbox init",
    )
    publish_id, upload_url = _upload_target(init, auth.secrets, "tiktok inbox missing upload_url")
    _put_chunks(client, file_path, upload_url, size, chunk_size, count, auth.secrets)
    username = _username(handle)
    _save_pending(
        working_directory,
        account,
        clip_id,
        publish_id=publish_id,
        mode="inbox",
        privacy_level=privacy_level,
        title=title,
        username=username,
    )
    status, payload = _poll(
        client,
        auth,
        publish_id,
        done=INBOX_DONE,
        poll_interval=poll_interval,
        max_polls=max_polls,
    )
    return _conclude(
        mode="inbox",
        status=status,
        payload=payload,
        publish_id=publish_id,
        secrets=auth.secrets,
        privacy_level=privacy_level,
        title=title,
        username=username,
        size=size,
        working_directory=working_directory,
        account=account,
        clip_id=clip_id,
    )


def _resume(
    client: httpx.Client,
    auth: _Auth,
    pending: dict[str, Any],
    *,
    size: int,
    privacy_level: str,
    title: str,
    poll_interval: float,
    max_polls: int,
    working_directory: Path | None,
    account: str,
    clip_id: str | None,
) -> dict:
    """Poll the publish that already received the video. Do not init or PUT."""
    publish_id = str(pending.get("publish_id") or "").strip()
    mode = str(pending.get("mode") or "inbox")
    if mode not in {"inbox", "direct"}:
        mode = "inbox"
    done = DIRECT_DONE if mode == "direct" else INBOX_DONE
    status, payload = _poll(
        client,
        auth,
        publish_id,
        done=done,
        poll_interval=poll_interval,
        max_polls=max_polls,
    )
    return _conclude(
        mode=mode,
        status=status,
        payload=payload,
        publish_id=publish_id,
        secrets=auth.secrets,
        privacy_level=str(pending.get("privacy_level") or privacy_level),
        title=str(pending.get("title") or title),
        username=str(pending.get("username") or ""),
        size=size,
        working_directory=working_directory,
        account=account,
        clip_id=clip_id,
    )


def _conclude(
    *,
    mode: str,
    status: str,
    payload: dict[str, Any],
    publish_id: str,
    secrets: list[str],
    privacy_level: str,
    title: str,
    username: str,
    size: int,
    working_directory: Path | None,
    account: str,
    clip_id: str | None,
) -> dict[str, Any]:
    data = payload.get("data") or {}
    if status == "FAILED":
        # The publish is terminal and not public. A later attempt may upload again.
        _clear_pending(working_directory, account, clip_id)
        raise RuntimeError(scrub(f"tiktok publish failed: {payload}", secrets))
    if mode == "direct":
        if status != "PUBLISH_COMPLETE":
            raise TimeoutError(scrub(f"tiktok status={status} publish_id={publish_id} {payload}", secrets))
        return _result(
            publish_id=publish_id,
            status=status,
            privacy_level=privacy_level,
            title=title,
            username=username,
            data=data,
        )
    if status not in INBOX_DONE:
        uploaded = data.get("uploaded_bytes")
        if uploaded == size or status in {"PROCESSING_UPLOAD", "PROCESSING_DOWNLOAD"}:
            return _result(
                publish_id=publish_id,
                status=status or "PROCESSING_UPLOAD",
                privacy_level=privacy_level,
                title=title,
                username=username,
                data=data,
            )
        raise TimeoutError(scrub(f"tiktok status={status} publish_id={publish_id} {payload}", secrets))
    return _result(
        publish_id=publish_id,
        status=status,
        privacy_level=privacy_level,
        title=title,
        username=username,
        data=data,
    )


def _pending_path(working_directory: Path, account: str, clip_id: str) -> Path:
    name = f"{slug(account, fallback='default')}__{slug(clip_id, fallback='clip')}.json"
    return Path(working_directory) / "tiktok_pending" / name


def _load_pending(
    working_directory: Path | None,
    account: str,
    clip_id: str | None,
) -> dict[str, Any] | None:
    if working_directory is None or not clip_id:
        return None
    path = _pending_path(working_directory, account, clip_id)
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        raise RuntimeError(
            f"unreadable tiktok pending publish for clip {slug(clip_id, fallback='clip')}"
        ) from None
    if not isinstance(data, dict) or not str(data.get("publish_id") or "").strip():
        raise RuntimeError(
            f"unreadable tiktok pending publish for clip {slug(clip_id, fallback='clip')}"
        ) from None
    return data


def _save_pending(
    working_directory: Path | None,
    account: str,
    clip_id: str | None,
    *,
    publish_id: str,
    mode: str,
    privacy_level: str,
    title: str,
    username: str,
) -> None:
    if working_directory is None or not clip_id:
        return
    path = _pending_path(working_directory, account, clip_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "publish_id": publish_id,
        "mode": mode,
        "privacy_level": privacy_level,
        "title": title,
        "username": username,
        "uploaded": True,
    }
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=True, separators=(",", ":"))
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _clear_pending(working_directory: Path | None, account: str, clip_id: str | None) -> None:
    if working_directory is None or not clip_id:
        return
    path = _pending_path(working_directory, account, clip_id)
    try:
        path.unlink()
    except OSError:
        pass


def _post_allowing_refresh(
    client: httpx.Client,
    url: str,
    auth: _Auth,
    body: dict[str, Any],
    prefix: str,
) -> httpx.Response:
    try:
        response = client.post(url, headers=auth.headers(), json=body)
    except httpx.HTTPError:
        raise RuntimeError(f"{prefix} request failed") from None
    if response.status_code != 401 or not auth.retry_unauthorized():
        return response
    try:
        return client.post(url, headers=auth.headers(), json=body)
    except httpx.HTTPError:
        raise RuntimeError(f"{prefix} request failed") from None


def _post_json(
    client: httpx.Client,
    url: str,
    auth: _Auth,
    body: dict[str, Any],
    prefix: str,
) -> dict[str, Any]:
    response = _post_allowing_refresh(client, url, auth, body, prefix)
    payload: dict[str, Any] = {}
    if response.content:
        try:
            parsed = response.json()
        except ValueError:
            parsed = {}
        if isinstance(parsed, dict):
            payload = parsed
    err = payload.get("error")
    code = err.get("code") if isinstance(err, dict) else err
    if response.status_code >= 400 or code not in (None, "ok", ""):
        detail = scrub((response.text or "")[:800], auth.secrets)
        raise RuntimeError(f"{prefix} {response.status_code}: {detail}")
    return payload


def _upload_target(body: dict[str, Any], secrets: list[str], prefix: str) -> tuple[str, str]:
    data = body.get("data") or {}
    upload_url = data.get("upload_url")
    publish_id = data.get("publish_id")
    if not upload_url or not publish_id:
        raise RuntimeError(scrub(f"{prefix}: {body}", secrets))
    return str(publish_id), str(upload_url)


def _put_chunks(
    client: httpx.Client,
    file_path: Path,
    upload_url: str,
    size: int,
    chunk_size: int,
    count: int,
    secrets: list[str],
) -> None:
    # The upload URL carries an upload_token. Never echo it if the PUT fails.
    hidden = [*secrets, *known_secrets(upload_url)]
    try:
        for values in parse_qs(urlsplit(upload_url).query).values():
            hidden.extend(known_secrets(*values))
    except ValueError:
        pass
    with file_path.open("rb") as handle:
        for offset, length in iter_chunk_ranges(size, chunk_size, count):
            handle.seek(offset)
            blob = handle.read(length)
            if len(blob) != length:
                raise RuntimeError("tiktok upload short read")
            end = offset + length - 1
            try:
                put = client.put(
                    upload_url,
                    headers={
                        "Content-Type": "video/mp4",
                        "Content-Length": str(length),
                        "Content-Range": f"bytes {offset}-{end}/{size}",
                    },
                    content=blob,
                )
            except httpx.HTTPError:
                raise RuntimeError("tiktok upload request failed") from None
            if put.status_code not in {200, 201, 206}:
                detail = scrub((put.text or "")[:500], hidden)
                raise RuntimeError(f"tiktok upload {put.status_code}: {detail}")


def _poll(
    client: httpx.Client,
    auth: _Auth,
    publish_id: str,
    *,
    done: set[str],
    poll_interval: float,
    max_polls: int,
) -> tuple[str, dict[str, Any]]:
    status = ""
    payload: dict[str, Any] = {}
    for _ in range(max_polls):
        response = _post_allowing_refresh(
            client,
            f"{OPEN}/v2/post/publish/status/fetch/",
            auth,
            {"publish_id": publish_id},
            "tiktok status",
        )
        if response.status_code == 401:
            detail = scrub((response.text or "")[:800], auth.secrets)
            raise RuntimeError(f"tiktok status 401: {detail}")
        parsed: Any = {}
        if response.content:
            try:
                parsed = response.json()
            except ValueError:
                parsed = {}
        payload = parsed if isinstance(parsed, dict) else {}
        status = ((payload.get("data") or {}).get("status") or "").upper()
        if status in done:
            break
        if poll_interval:
            time.sleep(poll_interval)
    return status, payload


def _result(
    *,
    publish_id: str,
    status: str,
    privacy_level: str,
    title: str,
    username: str,
    data: dict[str, Any],
) -> dict[str, Any]:
    post_id = _post_id(data)
    out: dict[str, Any] = {
        "publish_id": publish_id,
        "status": status,
        "privacy_level": privacy_level,
        "title": title,
        "post_url": build_post_url(username, post_id),
    }
    if post_id:
        out["post_id"] = post_id
    raw = data.get("publicaly_available_post_id")
    if raw in (None, "", []):
        raw = data.get("publicly_available_post_id")
    if raw not in (None, "", []):
        # TikTok spells this field "publicaly". Keep that key.
        out["publicaly_available_post_id"] = raw
    return out


def _post_id(data: dict[str, Any]) -> str | None:
    raw = data.get("publicaly_available_post_id")
    if raw in (None, "", []):
        raw = data.get("publicly_available_post_id")
    if isinstance(raw, list):
        raw = raw[0] if raw else None
    if raw in (None, ""):
        return None
    return str(raw)


def _username(value: object) -> str:
    user = str(value or "").strip().lstrip("@")
    if not user or any(char in user for char in "/ \t\r\n"):
        return ""
    return user


def _clip_title(title: str) -> str:
    """TikTok caps the caption at 2200 UTF-16 code units."""
    text = title or ""
    encoded = text.encode("utf-16-le")
    if len(encoded) // 2 <= _TITLE_LIMIT:
        return text
    raw = encoded[: _TITLE_LIMIT * 2]
    if len(raw) >= 2:
        last = int.from_bytes(raw[-2:], "little")
        if 0xD800 <= last <= 0xDBFF:
            raw = raw[:-2]
    return raw.decode("utf-16-le", errors="ignore")


def _probe_duration(path: Path) -> float | None:
    binary = shutil.which("ffprobe")
    if not binary:
        return None
    try:
        result = subprocess.run(
            [
                binary,
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "json",
                str(path),
            ],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    try:
        payload = json.loads(result.stdout or "{}")
        return float((payload.get("format") or {}).get("duration"))
    except (TypeError, ValueError):
        return None
