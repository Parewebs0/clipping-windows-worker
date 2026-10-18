"""Publish job — YouTube, Instagram, TikTok.

MOLI-53: per-platform copy payloads from the VPS (title/caption/hashtags),
account selection (`payload.account` → per-account env credentials) and a
`scheduled_at` guard — a live publish refuses to run before its UTC hour.
Dry-run is never blocked by the schedule (it only moves the clip).
"""
from __future__ import annotations

import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.jobs.base import BaseJob
from app.services.publish_state import PublishStateStore
from app.utils.clip_storage import ClipStorage, ClipStorageError


def _account_env(settings, base_field: str, account_key: str) -> str:
    """Credential for one account (MOLI-53).

    `account_key` 'default' (or a key without an override) reads the legacy
    single-account field, so old deployments work unchanged. Other keys read
    ``<BASE_FIELD>_<KEY>`` from the environment, e.g.
    ``INSTAGRAM_ACCESS_TOKEN_IG2`` for account key ``ig2``."""
    key = re.sub(r"[^A-Za-z0-9]+", "_", str(account_key or "default")).strip("_").upper()
    override = ""
    if key and key != "DEFAULT":
        override = (os.environ.get(f"{base_field.upper()}_{key}") or "").strip()
    if override:
        return override
    return (getattr(settings, base_field, None) or "").strip()


def _assert_past_scheduled_at(payload: dict) -> None:
    """Live publishes must not run before their UTC hour (MOLI-53).

    The VPS already holds the job back via ``available_at``; this is the belt
    and braces on the worker (clock skew, a requeued job, manual runs)."""
    raw = payload.get("scheduled_at")
    if not raw:
        return
    try:
        scheduled = datetime.fromisoformat(str(raw))
    except ValueError:
        return
    if scheduled.tzinfo is None:
        scheduled = scheduled.replace(tzinfo=timezone.utc)
    now = datetime.now(timezone.utc)
    if now < scheduled:
        raise ValueError(
            f"publish scheduled_at {scheduled.isoformat()} not reached (now {now.isoformat()}); "
            "job must be claimed only after its UTC hour"
        )


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

        if not dry_run:
            _assert_past_scheduled_at(payload)
            # Idempotency: a previous attempt may have published already and then
            # failed (e.g. reporting the result). Never upload the same clip twice.
            store = PublishStateStore(self._publish_state_dir())
            prior = store.get(str(clip_id), platform)
            if prior is not None:
                return self._reuse_prior(prior, clip_id=str(clip_id), campaign_id=campaign_id, file_path=file_path)

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
                privacy=getattr(self.settings, "youtube_privacy", None) or "private",
                paid_promotion=bool(payload.get("paid_promotion")),
            )
            post_url = uploaded["post_url"]
            extra = {"video_id": uploaded.get("video_id")}
        elif platform == "instagram":
            from app.services.instagram_upload import upload_reel

            account = str(payload.get("account") or "default")
            uploaded = upload_reel(
                file_path=dest,
                caption=str(caption),
                access_token=_account_env(self.settings, "instagram_access_token", account),
                ig_user_id=_account_env(self.settings, "instagram_ig_user_id", account),
                account_key=account,
                working_directory=self.settings.working_directory,
                auth=getattr(self.settings, "instagram_auth", None),
                app_id=getattr(self.settings, "instagram_app_id", None),
                app_secret=getattr(self.settings, "instagram_app_secret", None),
                graph_version=getattr(self.settings, "instagram_graph_version", None),
            )
            post_url = uploaded["post_url"]
            extra = {"media_id": uploaded.get("media_id"), "account": account}
        else:
            from app.services.tiktok_upload import upload_video

            account = str(payload.get("account") or "default")
            # MOLI-53: TikTok's post text IS the per-platform copy the VPS built
            # (caption already carries mentions/FTC/hashtags, ≤2200). The old
            # title-only path dropped the hashtags entirely.
            tiktok_text = str(caption).strip() or str(title)
            handle_raw = payload.get("handle")
            uploaded = upload_video(
                file_path=dest,
                title=tiktok_text,
                access_token=_account_env(self.settings, "tiktok_access_token", account),
                privacy_level=getattr(self.settings, "tiktok_privacy", None) or "SELF_ONLY",
                mode=getattr(self.settings, "tiktok_mode", None) or "inbox",
                paid_promotion=bool(payload.get("paid_promotion")),
                handle=str(handle_raw).strip() if handle_raw else None,
                account=account,
                client_key=_account_env(self.settings, "tiktok_client_key", account),
                client_secret=_account_env(self.settings, "tiktok_client_secret", account),
                refresh_token=_account_env(self.settings, "tiktok_refresh_token", account),
                oauth_state_dir=Path(self.settings.working_directory) / "oauth_state",
            )
            post_url = uploaded["post_url"]
            extra = {
                "publish_id": uploaded.get("publish_id"),
                "privacy_level": uploaded.get("privacy_level"),
                "account": account,
            }
            if uploaded.get("post_id"):
                extra["post_id"] = uploaded["post_id"]
            if uploaded.get("publicaly_available_post_id"):
                extra["publicaly_available_post_id"] = uploaded["publicaly_available_post_id"]

        pub = {"platform": platform, "status": "posted", "post_url": post_url}
        pub.update(extra)
        # Persist BEFORE anything else can fail (reporting to the API happens later).
        try:
            store.record(str(clip_id), platform, pub, job_id=str(self.job.id), final_path_worker=str(dest))
        except Exception as e:  # noqa: BLE001
            # Do not fail the job here: failing would trigger a retry that
            # re-publishes. The API result is then the only record.
            self.logger.error(
                "publish state NOT persisted (retry could duplicate)",
                clip_id=clip_id, platform=platform, error=str(e),
            )
        self.logger.info("published", clip_id=clip_id, platform=platform, post_url=post_url)
        return {
            "dry_run": False,
            "source_moved": True,
            "final_path_worker": str(dest),
            "publications": [pub],
        }

    def _publish_state_dir(self) -> Path:
        configured = getattr(self.settings, "publish_state_dir", None)
        if configured:
            return Path(configured)
        return Path(self.settings.working_directory) / "publish_state"

    def _reuse_prior(
        self,
        prior: dict[str, Any],
        *,
        clip_id: str,
        campaign_id: int | str,
        file_path: str | None,
    ) -> dict[str, Any]:
        pub = dict(prior["publication"])
        pub["status"] = "posted"
        pub["reused_from_state"] = True
        try:
            dest = str(self._move_to_uploaded(clip_id=clip_id, campaign_id=campaign_id, file_path=file_path))
        except (FileNotFoundError, ValueError):
            dest = prior.get("final_path_worker") or ""
        self.logger.warning(
            "publish skipped: already published (reusing stored result)",
            clip_id=clip_id,
            platform=pub.get("platform"),
            post_url=pub.get("post_url"),
            first_job_id=prior.get("job_id"),
        )
        return {
            "dry_run": False,
            "source_moved": bool(dest),
            "final_path_worker": dest,
            "publications": [pub],
        }

    def _move_to_uploaded(
        self,
        clip_id: str,
        campaign_id: int | str,
        file_path: str | None,
    ) -> Path:
        storage = ClipStorage.for_settings(self.settings, campaign_id)
        already = storage.path_for(clip_id, "uploaded")
        if already.exists():
            return already
        src: Path | None = Path(file_path) if file_path else None
        if src is None or not src.exists():
            src = storage.path_for(clip_id, "pending_upload")
        if not src.exists() and getattr(self.settings, "storage_root", None):
            # Clip written before STORAGE_ROOT was set: legacy pending_upload/.
            legacy = ClipStorage(self.settings.clip_storage_root, campaign_id).path_for(clip_id, "pending_upload")
            if legacy.exists():
                src = legacy
        if not src.exists():
            raise FileNotFoundError(f"clip not in pending_upload or payload path: {src}")
        try:
            return storage.move(src, clip_id, "uploaded")
        except ClipStorageError as e:
            raise ValueError(str(e)) from e
