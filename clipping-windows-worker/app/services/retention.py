"""Media retention for the CPU fallback worker (MEDIA_RETENTION_ENABLED)."""
from __future__ import annotations

import time
from pathlib import Path

from app.config import Settings
from app.utils.logging import get_logger

logger = get_logger("retention")


def _roots(settings: Settings) -> list[tuple[Path, float]]:
    """(root, max_age_hours). Clips keep clip_retention_days, not 24 h."""
    hours = settings.media_retention_hours
    roots = [(settings.downloads_dir, hours), (settings.jobs_dir, hours),
             (settings.outputs_dir, hours), (settings.temp_dir, hours)]
    if settings.clip_storage_root:
        roots.append((Path(settings.clip_storage_root), settings.clip_retention_days * 24))
    return roots


def _inside(path: Path, other: Path | None) -> bool:
    if other is None:
        return False
    try:
        path.resolve().relative_to(Path(other).resolve())
        return True
    except (ValueError, OSError):
        return False


def sweep_media(settings: Settings, *, now: float | None = None) -> dict[str, int]:
    """Delete files older than their root's max age, then empty dirs.

    Legacy roots only: STORAGE_ROOT is handled by storage_cleanup (it must ask
    the server first). Logs and publish_state are never touched.
    """
    now = time.time() if now is None else now
    storage_root = getattr(settings, "storage_root", None)
    removed = freed = 0
    for root, max_age_h in _roots(settings):
        if not root.is_dir() or _inside(root, storage_root):
            continue
        cutoff = now - max_age_h * 3600
        nested = storage_root is not None and _inside(Path(storage_root), root)
        for path in sorted(root.rglob("*"), key=lambda p: len(p.parts), reverse=True):
            if nested and _inside(path, storage_root):
                continue
            try:
                if path.is_file() or path.is_symlink():
                    st = path.lstat()
                    if st.st_mtime < cutoff:
                        path.unlink()
                        removed += 1
                        freed += st.st_size
                elif path.is_dir() and not any(path.iterdir()):
                    path.rmdir()
            except OSError as exc:
                logger.warning("retention: cannot remove", path=str(path), error=str(exc))
    if removed:
        logger.info("retention sweep", removed=removed, freed_mb=round(freed / 1e6, 1))
    return {"removed": removed, "freed_bytes": freed}
