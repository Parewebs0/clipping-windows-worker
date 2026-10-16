"""Permanent yt-dlp failures and the body posted to POST /worker/jobs/{id}/fail.

Keep the needles aligned with clipping-system-vps app/services/download_errors.py.
The API rejects error_message longer than its schema used to allow (2048) with
422, which left the job in processing until the lease requeued it forever.
"""
from __future__ import annotations

import re

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")

# Hard cap of the API field. Stay under it even against an API that has not
# picked up the server-side truncate yet.
FAIL_MESSAGE_LIMIT = 2048

_MARKERS: tuple[tuple[str, str], ...] = (
    ("private video", "video_private"),
    ("video is private", "video_private"),
    ("members-only", "members_only"),
    ("members only", "members_only"),
    ("join this channel to get access", "members_only"),
    ("sign in to confirm your age", "age_restricted"),
    ("confirm your age", "age_restricted"),
    ("age-restricted", "age_restricted"),
    ("age restricted", "age_restricted"),
    ("login required", "login_required"),
    ("sign in to view this video", "login_required"),
    ("removed by the uploader", "video_removed"),
    ("has been removed", "video_removed"),
    ("account associated with this video has been terminated", "video_unavailable"),
    ("this video is unavailable", "video_unavailable"),
    ("this video isn't available", "video_unavailable"),
    ("this video is not available", "video_unavailable"),
    ("video unavailable", "video_unavailable"),
    ("no longer available", "video_unavailable"),
    ("content isn't available", "video_unavailable"),
)


def strip_ansi(text: str) -> str:
    return _ANSI_RE.sub("", text or "")


def permanent_download_kind(text: str) -> str | None:
    lowered = strip_ansi(text or "").lower()
    for needle, kind in _MARKERS:
        if needle in lowered:
            return kind
    return None


def clip_error_message(text: str, limit: int = FAIL_MESSAGE_LIMIT) -> str:
    cleaned = strip_ansi(str(text or "")).replace("\x00", "").strip()
    if not cleaned:
        return "unknown error"
    if len(cleaned) <= limit:
        return cleaned
    if limit < 8:
        return cleaned[:limit]
    head = (limit - 1) // 2
    tail = limit - 1 - head
    return cleaned[:head] + "…" + cleaned[-tail:]


def worker_fail_body(error: str, *, permanent: bool = False) -> dict:
    """JSON body for POST /worker/jobs/{id}/fail.

    Contract with WorkerFailCreate: error_message is a non-empty string of at
    most 2048 chars with no ANSI, and permanent is an optional bool. Extra
    keys are not sent. The API ignores unknown keys.
    """
    message = clip_error_message(error, FAIL_MESSAGE_LIMIT)
    body: dict = {"error_message": message}
    if permanent:
        body["permanent"] = True
    return body


class PermanentDownloadError(RuntimeError):
    """The URL will not download on a later attempt."""

    def __init__(self, kind: str, message: str) -> None:
        self.kind = kind
        clean = strip_ansi(message).replace("\x00", "").strip() or kind
        super().__init__(f"{kind}: {clean[:500]}")
