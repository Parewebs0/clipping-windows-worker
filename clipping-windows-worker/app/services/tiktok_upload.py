"""TikTok inbox upload (scope video.upload). Direct post needs video.publish."""
from __future__ import annotations

import time
from pathlib import Path

import httpx

OPEN = "https://open.tiktokapis.com"
DONE = {"PUBLISH_COMPLETE", "FAILED", "SEND_TO_USER_INBOX"}


def upload_video(
    *,
    file_path: Path,
    title: str,
    access_token: str,
    privacy_level: str = "SELF_ONLY",
) -> dict:
    if not file_path.exists():
        raise FileNotFoundError(str(file_path))
    if not access_token:
        raise RuntimeError("TIKTOK_ACCESS_TOKEN missing")

    size = file_path.stat().st_size
    raw = file_path.read_bytes()
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json; charset=UTF-8",
    }
    with httpx.Client(timeout=300.0, follow_redirects=True) as client:
        init = client.post(
            f"{OPEN}/v2/post/publish/inbox/video/init/",
            headers=headers,
            json={
                "source_info": {
                    "source": "FILE_UPLOAD",
                    "video_size": size,
                    "chunk_size": size,
                    "total_chunk_count": 1,
                }
            },
        )
        body = init.json() if init.content else {}
        err = (body.get("error") or {}).get("code")
        if init.status_code >= 400 or (err not in (None, "ok", "")):
            raise RuntimeError(f"tiktok inbox init {init.status_code}: {init.text[:800]}")
        data = body.get("data") or {}
        upload_url = data.get("upload_url")
        publish_id = data.get("publish_id")
        if not upload_url or not publish_id:
            raise RuntimeError(f"tiktok inbox missing upload_url: {body}")

        put = client.put(
            upload_url,
            headers={
                "Content-Type": "video/mp4",
                "Content-Length": str(size),
                "Content-Range": f"bytes 0-{size - 1}/{size}",
            },
            content=raw,
        )
        if put.status_code not in {200, 201, 206}:
            raise RuntimeError(f"tiktok upload {put.status_code}: {put.text[:500]}")

        status = ""
        payload: dict = {}
        for _ in range(90):
            st = client.post(
                f"{OPEN}/v2/post/publish/status/fetch/",
                headers=headers,
                json={"publish_id": publish_id},
            )
            payload = st.json() if st.content else {}
            status = ((payload.get("data") or {}).get("status") or "").upper()
            if status in DONE:
                break
            time.sleep(5)
        if status == "FAILED":
            raise RuntimeError(f"tiktok publish failed: {payload}")
        if status not in DONE:
            uploaded = (payload.get("data") or {}).get("uploaded_bytes")
            if uploaded == size or status in {"PROCESSING_UPLOAD", "PROCESSING_DOWNLOAD"}:
                return {
                    "publish_id": publish_id,
                    "status": status or "PROCESSING_UPLOAD",
                    "privacy_level": privacy_level,
                    "title": title,
                    "post_url": "https://www.tiktok.com/@clipeand22",
                }
            raise TimeoutError(f"tiktok status={status} publish_id={publish_id} {payload}")

    return {
        "publish_id": publish_id,
        "status": status,
        "privacy_level": privacy_level,
        "title": title,
        "post_url": "https://www.tiktok.com/@clipeand22",
    }
