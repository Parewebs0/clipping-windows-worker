"""YouTube Data API v3 upload. Tokens from Worker .env only."""
from __future__ import annotations

from pathlib import Path

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

SCOPES = (
    "https://www.googleapis.com/auth/youtube.upload",
    "https://www.googleapis.com/auth/youtube",
)
TOKEN_URI = "https://oauth2.googleapis.com/token"


def upload_short(
    *,
    file_path: Path,
    title: str,
    description: str,
    client_id: str,
    client_secret: str,
    refresh_token: str,
    privacy: str = "public",
) -> dict:
    if not file_path.exists():
        raise FileNotFoundError(str(file_path))
    if not client_id or not client_secret or not refresh_token:
        raise RuntimeError("YOUTUBE_CLIENT_ID / SECRET / REFRESH_TOKEN missing")

    creds = Credentials(
        token=None,
        refresh_token=refresh_token,
        token_uri=TOKEN_URI,
        client_id=client_id,
        client_secret=client_secret,
        scopes=list(SCOPES),
    )
    creds.refresh(Request())
    youtube = build("youtube", "v3", credentials=creds, cache_discovery=False)

    safe_title = (title or "clip")[:100]
    body = {
        "snippet": {
            "title": safe_title,
            "description": (description or safe_title)[:5000],
            "categoryId": "22",
        },
        "status": {
            "privacyStatus": privacy if privacy in {"public", "unlisted", "private"} else "public",
            "selfDeclaredMadeForKids": False,
        },
    }
    media = MediaFileUpload(str(file_path), mimetype="video/mp4", resumable=True, chunksize=8 * 1024 * 1024)
    request = youtube.videos().insert(part="snippet,status", body=body, media_body=media)
    response = None
    while response is None:
        _status, response = request.next_chunk()
    video_id = response["id"]
    return {
        "video_id": video_id,
        "post_url": f"https://www.youtube.com/shorts/{video_id}",
        "watch_url": f"https://www.youtube.com/watch?v={video_id}",
    }
