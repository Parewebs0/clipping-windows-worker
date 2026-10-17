"""Cleanup of STORAGE_ROOT driven by the server (POST /worker/storage/check).

The worker lists what it has on disk and the server says, from the DB, what
is safe to delete. Nothing is deleted when the server cannot be reached.

Order of a run:
1. campaign archived and idle          -> whole <root>/<campaign_id>/
2. asset decided and idle               -> sources/<asset>/ and work/<asset>/
                                           (transcript*.json is kept)
3. clip rejected or published, idle     -> its .mp4 (pending or done)
4. clip older than CLIP_RETENTION_DAYS  -> its .mp4, unless a job uses it
5. disk above DISK_HIGH_WATERMARK_PCT   -> oldest processed sources first
                                           (asset idle + transcript present)

Every delete goes through storage_layout.safe_remove: inside STORAGE_ROOT
only, never a symlink. Dry-run logs the same lines without deleting.

    python -m app.services.storage_cleanup            # dry-run
    python -m app.services.storage_cleanup --apply    # really delete
"""
from __future__ import annotations

import argparse
import shutil
import time
from pathlib import Path
from typing import Any, Callable

from app.config import Settings
from app.utils.logging import get_logger
from app.utils.storage_layout import (
    StorageLayout,
    StoragePathError,
    campaign_key,
    item_key,
    safe_remove,
)

logger = get_logger("storage_cleanup")


def _dirs(path: Path) -> list[Path]:
    if not path.is_dir() or path.is_symlink():
        return []
    return [p for p in path.iterdir() if p.is_dir() and not p.is_symlink()]


def scan(root: Path) -> dict[str, Any]:
    """What is on disk, by id. Names that are not valid ids are ignored."""
    campaigns: dict[str, Path] = {}
    assets: dict[str, list[dict[str, Any]]] = {}
    clips: dict[str, list[Path]] = {}
    for cdir in _dirs(root):
        try:
            cid = campaign_key(cdir.name)
        except StoragePathError:
            continue
        campaigns[cid] = cdir
        for kind in ("sources", "work"):
            for adir in _dirs(cdir / kind):
                try:
                    aid = item_key(adir.name)
                except StoragePathError:
                    continue
                entry = next((e for e in assets.setdefault(aid, []) if e["campaign_id"] == cid), None)
                if entry is None:
                    entry = {"campaign_id": cid, "sources": None, "work": None}
                    assets[aid].append(entry)
                entry[kind] = adir
        for sub in ("pending", "done"):
            d = cdir / "clips" / sub
            if not d.is_dir() or d.is_symlink():
                continue
            for f in d.iterdir():
                if f.is_file() and not f.is_symlink() and f.suffix.lower() == ".mp4":
                    clips.setdefault(f.stem, []).append(f)
    return {"campaigns": campaigns, "assets": assets, "clips": clips}


def _mtime(path: Path) -> float:
    try:
        return path.stat().st_mtime
    except OSError:
        return 0.0


def _clean_work(work: Path, root: Path, *, dry_run: bool, **ctx) -> int:
    """Delete work/<asset>/ content except the transcript."""
    freed = 0
    for child in sorted(work.iterdir()) if work.is_dir() else []:
        if child.is_file() and child.name.startswith("transcript") and child.suffix == ".json":
            continue
        freed += safe_remove(child, root, dry_run=dry_run, **ctx)
    return freed


def run_cleanup(
    settings: Settings,
    api,
    *,
    dry_run: bool | None = None,
    now: float | None = None,
    disk_usage: Callable[[str], Any] = shutil.disk_usage,
) -> dict[str, Any]:
    dry_run = settings.storage_cleanup_dry_run if dry_run is None else dry_run
    now = time.time() if now is None else now
    summary: dict[str, Any] = {"dry_run": dry_run, "deleted": {}, "freed_bytes": 0}
    root_cfg = getattr(settings, "storage_root", None)
    if not root_cfg or not Path(root_cfg).is_dir():
        summary["skipped"] = "no_storage_root"
        return summary
    root = Path(root_cfg)
    layout = StorageLayout(root)
    inv = scan(root)
    if not (inv["campaigns"] or inv["assets"] or inv["clips"]):
        summary["skipped"] = "empty"
        return summary
    try:
        answer = api.storage_check({
            "campaign_ids": sorted(inv["campaigns"]),
            "asset_ids": sorted(inv["assets"]),
            "clip_ids": sorted(inv["clips"]),
        }) or {}
    except Exception as exc:  # noqa: BLE001 - no answer, no deletes
        logger.warning("storage cleanup skipped: server check failed", error=str(exc))
        summary["skipped"] = "server_check_failed"
        return summary

    def count(reason: str, freed: int) -> None:
        if freed or dry_run:
            summary["deleted"][reason] = summary["deleted"].get(reason, 0) + 1
            summary["freed_bytes"] += freed

    gone: set[str] = set()
    camps = answer.get("campaigns") or {}
    for cid, cdir in inv["campaigns"].items():
        if (camps.get(cid) or {}).get("delete") is True:
            freed = safe_remove(cdir, root, dry_run=dry_run, reason="campaign_archived", campaign_id=cid)
            count("campaign_archived", freed)
            gone.add(cid)

    assets = answer.get("assets") or {}
    for aid, entries in inv["assets"].items():
        info = assets.get(aid) or {}
        if info.get("done") is not True:
            continue
        for e in entries:
            if e["campaign_id"] in gone:
                continue
            if info.get("campaign_id") is not None and str(info["campaign_id"]) != e["campaign_id"]:
                logger.warning("storage: asset under another campaign, left alone", asset_id=aid,
                               folder_campaign=e["campaign_id"], db_campaign=info.get("campaign_id"))
                continue
            freed = 0
            if e["sources"] is not None:
                freed += safe_remove(e["sources"], root, dry_run=dry_run, reason="asset_decided",
                                     campaign_id=e["campaign_id"], asset_id=aid)
            if e["work"] is not None:
                freed += _clean_work(e["work"], root, dry_run=dry_run, reason="asset_decided",
                                     campaign_id=e["campaign_id"], asset_id=aid)
            count("asset_decided", freed)
            e["sources"] = None

    clips = answer.get("clips") or {}
    ttl = settings.clip_retention_days * 86400
    for clip_id, paths in inv["clips"].items():
        info = clips.get(clip_id) or {}
        if info.get("active") is True:
            continue
        for path in paths:
            if path.parent.parent.parent.name in gone:
                continue
            if info.get("release") is True:
                reason = f"clip_{info.get('status') or 'released'}"
            elif now - _mtime(path) > ttl:
                reason = "clip_ttl"
            else:
                continue
            count(reason, safe_remove(path, root, dry_run=dry_run, reason=reason, clip_id=clip_id))

    try:
        usage = disk_usage(str(root))
        total, used = float(usage.total), float(usage.used)
    except OSError as exc:
        logger.warning("storage: disk usage unavailable", error=str(exc))
        total = used = 0.0
    if total > 0:
        used -= 0 if not dry_run else summary["freed_bytes"]
        pct = 100.0 * used / total
        summary["disk_pct"] = round(pct, 1)
        if pct > settings.disk_high_watermark_pct:
            pool = []
            for aid, entries in inv["assets"].items():
                info = assets.get(aid) or {}
                if info.get("idle") is not True or info.get("known") is False:
                    continue
                for e in entries:
                    src = e["sources"]
                    if src is None or e["campaign_id"] in gone:
                        continue
                    if e["work"] is None or not any(e["work"].glob("transcript*.json")):
                        continue  # not processed yet
                    pool.append((_mtime(src), aid, e))
            for _m, aid, e in sorted(pool, key=lambda t: t[0]):
                if 100.0 * used / total <= settings.disk_high_watermark_pct:
                    break
                freed = safe_remove(e["sources"], root, dry_run=dry_run, reason="disk_pressure",
                                    campaign_id=e["campaign_id"], asset_id=aid, disk_pct=round(100.0 * used / total, 1))
                count("disk_pressure", freed)
                used -= freed
    logger.info("storage cleanup summary", **{k: v for k, v in summary.items() if k != "deleted"},
                deleted=summary["deleted"])
    return summary


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="STORAGE_ROOT cleanup (dry-run unless --apply)")
    p.add_argument("--apply", action="store_true", help="really delete (default: dry-run)")
    args = p.parse_args(argv)
    from app.config import get_settings
    from app.services.api_client import APIClient

    settings = get_settings()
    api = APIClient(settings)
    try:
        summary = run_cleanup(settings, api, dry_run=not args.apply)
    finally:
        api.close()
    print(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
