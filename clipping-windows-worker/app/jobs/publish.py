"""Publish job — YouTube, Instagram, TikTok."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from app.jobs.base import BaseJob
from app.utils.clip_storage import ClipStorage, ClipStorageError


class PublishJob(BaseJob):
    type = "publish"

    def execute(self) -> dict[str, Any]:
        payload = self.job.payload if isinstance(self.job.payload, dict) else {}
        dry_run = payload.get("dry_run", True)
        platform = (payload.get("platform") or "youtube").lower()
        clip_id = payload.get("clip_id")
        campaign_id = payload.get("campaign_id")
        file_path = payload.get("file_path")
        title = payload.get("title") or payload.get("caption") or f"clip-{clip_id}"
        caption = payload.get("caption") or title
        if not clip_id:
            raise ValueError("payload.clip_id is required")
        if campaign_id is None:
            raise ValueError("payload.campaign_id is required")
        if platform not in {"youtube", "instagram", "tiktok"}:
            raise NotImplementedError(f"platform {platform} not implemented")

        dest = self._move_to_uploaded(clip_id=str(clip_id), campaign_id=campaign_id, file_path=file_path)

        if dry_run:
            fake = f"https://{platform}.example/dry-run-{clip_id}"
            self.logger.info("publish dry-run", clip_id=clip_id, platform=platform, dest=str(dest))
            return {
                "dry_run": True,
                "source_moved": True,
                "final_path_worker": str(dest),
                "publications": [{"platform": platform, "status": "posted", "post_url": fake}],
            }

        extra: dict[str, Any] = {}
        if platform == "youtube":
            from app.services.youtube_upload import upload_short

            uploaded = upload_short(
                file_path=dest,
                title=str(title),
                description=str(caption),
                client_id=getattr(self.settings, "youtube_client_id", None) or "",
                client_secret=getattr(self.settings, "youtube_client_secret", None) or "",
                refresh_token=getattr(self.settings, "youtube_refresh_token", None) or "",
                privacy=getattr(self.settings, "youtube_privacy", None) or "public",
            )
            post_url = uploaded["post_url"]
            extra = {"video_id": uploaded.get("video_id")}
        elif platform == "instagram":
            from app.services.instagram_upload import upload_reel

            uploaded = upload_reel(
                file_path=dest,
                caption=str(caption),
                access_token=getattr(self.settings, "instagram_access_token", None) or "",
                ig_user_id=getattr(self.settings, "instagram_ig_user_id", None) or "",
            )
            post_url = uploaded["post_url"]
            extra = {"media_id": uploaded.get("media_id")}
        else:
            from app.services.tiktok_upload import upload_video

            uploaded = upload_video(
                file_path=dest,
                title=str(title)[:150],
                access_token=getattr(self.settings, "tiktok_access_token", None) or "",
                privacy_level=getattr(self.settings, "tiktok_privacy", None) or "SELF_ONLY",
            )
            post_url = uploaded["post_url"]
            extra = {"publish_id": uploaded.get("publish_id"), "privacy_level": uploaded.get("privacy_level")}

        self.logger.info("published", clip_id=clip_id, platform=platform)
        pub = {"platform": platform, "status": "posted", "post_url": post_url}
        pub.update(extra)
        return {
            "dry_run": False,
            "source_moved": True,
            "final_path_worker": str(dest),
            "publications": [pub],
        }

    def _move_to_uploaded(
        self,
        clip_id: str,
        campaign_id: int | str,
        file_path: str | None,
    ) -> Path:
        storage = ClipStorage(
            storage_root=self.settings.clip_storage_root,
            campaign_id=campaign_id,
        )
        already = storage.path_for(clip_id, "uploaded")
        if already.exists():
            return already
        src: Path | None = Path(file_path) if file_path else None
        if src is None or not src.exists():
            src = storage.path_for(clip_id, "pending_upload")
        if not src.exists():
            raise FileNotFoundError(f"clip not in pending_upload or payload path: {src}")
        try:
            return storage.move(src, clip_id, "uploaded")
        except ClipStorageError as e:
            raise ValueError(str(e)) from e
