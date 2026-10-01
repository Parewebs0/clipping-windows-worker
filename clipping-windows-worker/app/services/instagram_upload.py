"""Instagram Reels via Graph API resumable upload. Tokens from Worker .env."""
from __future__ import annotations

import time
from pathlib import Path

import httpx

GRAPH = "https://graph.facebook.com"
API_VERSION = "v21.0"


def _raise_http(resp: httpx.Response, step: str) -> None:
    body = (resp.text or "")[:800]
    raise RuntimeError(f"instagram {step} HTTP {resp.status_code}: {body}")


def upload_reel(
    *,
    file_path: Path,
    caption: str,
    access_token: str,
    ig_user_id: str,
) -> dict:
    if not file_path.exists():
        raise FileNotFoundError(str(file_path))
    if not access_token or not ig_user_id:
        raise RuntimeError("INSTAGRAM_ACCESS_TOKEN / INSTAGRAM_IG_USER_ID missing")

    size = file_path.stat().st_size
    raw = file_path.read_bytes()
    if len(raw) != size:
        raise RuntimeError("instagram file size mismatch")

    headers_auth = {"Authorization": f"Bearer {access_token}"}
    with httpx.Client(timeout=300.0, follow_redirects=True) as client:
        create = client.post(
            f"{GRAPH}/{API_VERSION}/{ig_user_id}/media",
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

        uris = []
        if created.get("uri"):
            uris.append(created["uri"])
        uris.append(f"https://rupload.facebook.com/ig-api-upload/{API_VERSION}/{container_id}")
        uris.append(f"https://rupload.facebook.com/ig-api-upload/v21.0/{container_id}")

        last_err = None
        uploaded = False
        for uri in uris:
            for auth in (
                f"OAuth {access_token}",
                f"Bearer {access_token}",
            ):
                up = client.post(
                    uri,
                    headers={
                        "Authorization": auth,
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
        payload = {}
        for _ in range(40):
            st = client.get(
                f"{GRAPH}/{API_VERSION}/{container_id}",
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
                f"{GRAPH}/{API_VERSION}/{ig_user_id}/media_publish",
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
            f"{GRAPH}/{API_VERSION}/{media_id}",
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
