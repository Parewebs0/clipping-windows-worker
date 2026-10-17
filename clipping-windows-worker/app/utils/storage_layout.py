"""Per-campaign storage layout (STORAGE_ROOT) and guarded deletes.

    <root>/<campaign_id>/
        sources/<asset_id>/source.<ext>     downloaded source
        work/<asset_id>/                    audio + transcript.json (transcript kept)
        clips/pending/<clip_id>.mp4         QA pass, waiting for upload/decision
        clips/done/<clip_id>.mp4            uploaded (or archived)

`safe_remove` refuses anything outside the root, the root itself, a campaign
id that is not numeric, and symlinks (it removes neither the link target nor
follows links inside a tree: shutil.rmtree does not follow symlinks).
"""
from __future__ import annotations

import os
import re
import shutil
from pathlib import Path

from app.utils.logging import get_logger

logger = get_logger("storage")

_CAMPAIGN_RE = re.compile(r"^\d{1,12}$")
_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")

CLIP_DIRS = {"pending_upload": ("clips", "pending"), "uploaded": ("clips", "done"), "archived": ("clips", "done")}


class StoragePathError(ValueError):
    """An id or path that must not be used under STORAGE_ROOT."""


def campaign_key(campaign_id: object) -> str:
    key = str(campaign_id).strip()
    if not _CAMPAIGN_RE.match(key):
        raise StoragePathError(f"bad campaign_id: {campaign_id!r}")
    return key


def item_key(value: object) -> str:
    key = str(value).strip()
    if not _ID_RE.match(key):
        raise StoragePathError(f"bad id: {value!r}")
    return key


class StorageLayout:
    def __init__(self, root: Path | str) -> None:
        self.root = Path(root)

    def campaign_dir(self, campaign_id: object) -> Path:
        return self.root / campaign_key(campaign_id)

    def sources_dir(self, campaign_id: object, asset_id: object) -> Path:
        return self.campaign_dir(campaign_id) / "sources" / item_key(asset_id)

    def work_dir(self, campaign_id: object, asset_id: object) -> Path:
        return self.campaign_dir(campaign_id) / "work" / item_key(asset_id)

    def clips_dir(self, campaign_id: object, location: str) -> Path:
        parts = CLIP_DIRS.get(location)
        if parts is None:
            raise StoragePathError(f"bad clip location: {location!r}")
        return self.campaign_dir(campaign_id).joinpath(*parts)


def is_within(path: Path, root: Path) -> bool:
    """True if `path` (not following a final symlink) is strictly inside `root`."""
    try:
        real_root = Path(os.path.realpath(root))
        parent = Path(os.path.realpath(path.parent))
        candidate = parent / path.name
    except (OSError, ValueError):
        return False
    if candidate == real_root:
        return False
    try:
        candidate.relative_to(real_root)
    except ValueError:
        return False
    return path.name not in ("", ".", "..")


def tree_size(path: Path) -> int:
    if path.is_symlink() or not path.exists():
        return 0
    if path.is_file():
        return path.stat().st_size
    total = 0
    for dirpath, _dirs, files in os.walk(path, followlinks=False):
        for f in files:
            fp = Path(dirpath) / f
            try:
                if not fp.is_symlink():
                    total += fp.stat().st_size
            except OSError:
                pass
    return total


def safe_remove(path: Path, root: Path, *, dry_run: bool, reason: str, **ctx) -> int:
    """Delete a file or tree under `root`. Returns bytes freed (estimated in dry-run).

    Refuses (logs and returns 0) symlinks, paths outside root and root itself.
    """
    path = Path(path)
    if not path.exists() and not path.is_symlink():
        return 0
    if path.is_symlink():
        logger.warning("storage: refusing symlink", path=str(path), reason=reason, **ctx)
        return 0
    if not is_within(path, Path(root)):
        logger.warning("storage: refusing path outside STORAGE_ROOT", path=str(path), root=str(root), reason=reason, **ctx)
        return 0
    size = tree_size(path)
    logger.info("storage: delete" + (" (dry-run)" if dry_run else ""), path=str(path), reason=reason,
                bytes=size, dry_run=dry_run, **ctx)
    if dry_run:
        return size
    try:
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()
    except OSError as exc:
        logger.warning("storage: cannot delete", path=str(path), error=str(exc), **ctx)
        return 0
    return size
