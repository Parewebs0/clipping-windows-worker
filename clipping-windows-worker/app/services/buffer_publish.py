"""Instagram, TikTok and YouTube Shorts publishing through Buffer's GraphQL API.

Buffer has no upload endpoint: it fetches the video from a public URL. The
clip is copied under a random name into ``BUFFER_PUBLIC_DIR``, which the mini
PC serves through Tailscale Funnel at ``BUFFER_PUBLIC_BASE_URL``. The post is
created with ``mode: shareNow`` and polled until Buffer reports it ``sent``
with its ``externalLink`` (the public post URL that goes to Content Rewards).

The Buffer post id is written next to the publish state before polling, so a
retried job resumes polling that post instead of creating a second one.
"""
from __future__ import annotations

import json
import shutil
import time
import uuid
from pathlib import Path
from typing import Any, Callable

import httpx

API_URL = "https://api.buffer.com"
PLATFORMS = ("instagram", "tiktok", "youtube")
DEFAULT_TIMEOUT_S = 1200.0
# 250 requests/day per key on Buffer's plans: poll slowly.
DEFAULT_POLL_S = 30.0

CREATE_POST = """
mutation CreatePost($input: CreatePostInput!) {
  createPost(input: $input) {
    __typename
    ... on PostActionSuccess { post { id status } }
    ... on MutationError { message }
  }
}
"""

POST_STATUS = """
query PostStatus($id: PostId!) {
  post(input: {id: $id}) { id status externalLink error { message } }
}
"""


class BufferPublishError(RuntimeError):
    """Publishing through Buffer failed. The message never carries the key."""


def _graphql(client: httpx.Client, api_key: str, query: str, variables: dict[str, Any]) -> dict[str, Any]:
    resp = client.post(
        API_URL,
        json={"query": query, "variables": variables},
        headers={"Authorization": f"Bearer {api_key}"},
    )
    if resp.status_code != 200:
        raise BufferPublishError(f"buffer http {resp.status_code}")
    body = resp.json()
    if body.get("errors"):
        msg = "; ".join(str(e.get("message", "")) for e in body["errors"])[:300]
        raise BufferPublishError(f"buffer graphql error: {msg}")
    return body.get("data") or {}


def _post_input(
    platform: str, channel_id: str, text: str, video_url: str, *, title: str = "",
    youtube_privacy: str = "public", youtube_category_id: str = "22",
) -> dict[str, Any]:
    data: dict[str, Any] = {
        "channelId": channel_id,
        "text": text,
        "schedulingType": "automatic",
        "mode": "shareNow",
        "assets": [{"video": {"url": video_url}}],
    }
    if platform == "instagram":
        data["metadata"] = {"instagram": {"type": "reel", "shouldShareToFeed": True}}
    elif platform == "youtube":
        # Buffer posts YouTube as Shorts only (vertical, up to 3 min).
        privacy = youtube_privacy if youtube_privacy in {"public", "private", "unlisted"} else "private"
        data["metadata"] = {"youtube": {
            "title": (title or text).strip()[:100], "categoryId": str(youtube_category_id or "22"),
            "privacy": privacy, "madeForKids": False, "isAiGenerated": False,
        }}
    return data


def _pending_path(state_dir: Path, clip_id: str, platform: str, account: str | None = None) -> Path:
    from app.services.publish_state import account_suffix

    return Path(state_dir) / f"{clip_id}__{platform}{account_suffix(account)}.buffer.json"


def publish_video(
    *,
    file_path: Path,
    text: str,
    platform: str,
    channel_id: str,
    api_key: str,
    public_dir: Path | str,
    public_base_url: str,
    state_dir: Path,
    clip_id: str,
    account: str | None = None,
    title: str = "",
    youtube_privacy: str = "public",
    youtube_category_id: str = "22",
    timeout_s: float = DEFAULT_TIMEOUT_S,
    poll_s: float = DEFAULT_POLL_S,
    client: httpx.Client | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    """Post one video now and return ``{"post_url", "buffer_post_id"}``."""
    if platform not in PLATFORMS:
        raise BufferPublishError(f"buffer platform {platform} not supported")
    missing = [
        name
        for name, value in (
            ("BUFFER_API_KEY", api_key),
            (f"BUFFER_CHANNEL_{platform.upper()}", channel_id),
            ("BUFFER_PUBLIC_DIR", str(public_dir or "")),
            ("BUFFER_PUBLIC_BASE_URL", public_base_url),
        )
        if not str(value or "").strip()
    ]
    if missing:
        raise BufferPublishError("buffer publish not configured: " + ", ".join(missing) + " missing")

    pending = _pending_path(state_dir, clip_id, platform, account)
    own_client = client is None
    http = client or httpx.Client(timeout=30.0)
    try:
        if pending.exists():
            state = json.loads(pending.read_text(encoding="utf-8"))
        else:
            name = f"{uuid.uuid4().hex}.mp4"
            public_path = Path(public_dir) / name
            shutil.copyfile(file_path, public_path)
            video_url = f"{public_base_url.rstrip('/')}/{name}"
            try:
                data = _graphql(http, api_key, CREATE_POST, {"input": _post_input(
                    platform, channel_id, text, video_url, title=title,
                    youtube_privacy=youtube_privacy, youtube_category_id=youtube_category_id)})
            except Exception:
                public_path.unlink(missing_ok=True)
                raise
            result = data.get("createPost") or {}
            post = result.get("post") if result.get("__typename") == "PostActionSuccess" else None
            if not post or not post.get("id"):
                public_path.unlink(missing_ok=True)
                reason = result.get("message") or result.get("__typename") or "no post"
                raise BufferPublishError(f"buffer createPost failed: {str(reason)[:300]}")
            state = {"post_id": post["id"], "public_path": str(public_path)}
            pending.parent.mkdir(parents=True, exist_ok=True)
            pending.write_text(json.dumps(state), encoding="utf-8")

        post_id = state["post_id"]
        deadline = time.monotonic() + timeout_s
        last = "unknown"
        while True:
            post = _graphql(http, api_key, POST_STATUS, {"id": post_id}).get("post") or {}
            last = str(post.get("status") or "unknown")
            link = str(post.get("externalLink") or "").strip()
            if last == "sent" and link:
                _finish(pending, state)
                return {"post_url": link, "buffer_post_id": post_id}
            if last in {"error", "draft", "needs_approval"}:
                _finish(pending, state)
                reason = (post.get("error") or {}).get("message") or last
                raise BufferPublishError(f"buffer post {post_id} {last}: {str(reason)[:300]}")
            if time.monotonic() >= deadline:
                # Keep the pending file: a retry resumes this post, never re-posts.
                raise BufferPublishError(
                    f"buffer post {post_id} not public after {int(timeout_s)}s (status={last}, link={'yes' if link else 'no'})"
                )
            sleep(poll_s)
    finally:
        if own_client:
            http.close()


def _finish(pending: Path, state: dict[str, Any]) -> None:
    """Buffer has fetched the video (sent or failed): drop the public copy."""
    public_path = state.get("public_path")
    if public_path:
        Path(public_path).unlink(missing_ok=True)
    pending.unlink(missing_ok=True)
