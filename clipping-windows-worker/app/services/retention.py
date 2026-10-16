"""Media retention for the CPU fallback worker (MEDIA_RETENTION_ENABLED)."""
from __future__ import annotations

import time
from pathlib import Path

from app.config import Settings
from app.utils.logging import get_logger

logger = get_logger("retention")


def _roots(settings: Settings) -> list[Path]:
    roots = [settings.downloads_dir, settings.jobs_dir, settings.outputs_dir, settings.temp_dir]
    if settings.clip_storage_root:
        roots.append(Path(settings.clip_storage_root))
    return roots


def sweep_media(settings: Settings, *, now: float | None = None) -> dict[str, int]:
    """Delete files older than media_retention_hours, then empty dirs.

    Logs and publish_state are never touched.
    """
    now = time.time() if now is None else now
    cutoff = now - settings.media_retention_hours * 3600
    removed = freed = 0
    for root in _roots(settings):
        if not root.is_dir():
            continue
        for path in sorted(root.rglob("*"), key=lambda p: len(p.parts), reverse=True):
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
