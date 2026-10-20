"""Persistent local record of successful publications (idempotency guard).

A publish job can fail *after* the real upload succeeded (e.g. the result
cannot be reported to the API, the worker crashes, the lease expires). The
API then re-queues the same job (up to ``max_attempts``) and, without this
guard, the retry would upload the clip again → duplicate post.

Right after a successful upload the worker writes one small JSON file per
``(clip_id, platform)`` under ``publish_state_dir``. Before uploading, the
publish job checks that file and, if present, reuses the stored publication
(post_url, ids) instead of uploading again.

The key is ``clip_id + platform + account`` (not ``job_id``) on purpose: it
also covers a new publish job created later for the same clip/platform/account.
The ``default`` account keeps the legacy ``<clip>__<platform>.json`` name; a
named account (``<clip>__<platform>__<account>.json``) is a different post, so a
clip already on the shared account can be published on a dedicated one. To
force a re-publish on purpose, delete the corresponding file.
"""
from __future__ import annotations

import json
import os
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_SAFE = re.compile(r"[^A-Za-z0-9._-]+")


def account_suffix(account: str | None) -> str:
    """'' for the default account (legacy file names), else '__<account>'."""
    key = _SAFE.sub("_", str(account or "").strip().lower())
    return "" if key in ("", "default") else f"__{key}"


class PublishStateError(RuntimeError):
    """The state file exists but cannot be read; refuse to guess."""


class PublishStateStore:
    def __init__(self, root: Path | str) -> None:
        self.root = Path(root)

    def path_for(self, clip_id: str, platform: str, account: str | None = None) -> Path:
        clip = _SAFE.sub("_", str(clip_id)) or "unknown"
        plat = _SAFE.sub("_", str(platform).lower()) or "unknown"
        return self.root / f"{clip}__{plat}{account_suffix(account)}.json"

    def get(self, clip_id: str, platform: str, account: str | None = None) -> dict[str, Any] | None:
        """Return the stored record or None if this clip/platform was never published.

        A present-but-unreadable file raises ``PublishStateError``: it means a
        publication probably happened, so we must not silently upload again.
        """
        path = self.path_for(clip_id, platform, account)
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as e:
            raise PublishStateError(f"unreadable publish state {path}: {e}") from e
        if not isinstance(data, dict) or not isinstance(data.get("publication"), dict):
            raise PublishStateError(f"invalid publish state {path}")
        return data

    def record(
        self,
        clip_id: str,
        platform: str,
        publication: dict[str, Any],
        *,
        job_id: str | None = None,
        final_path_worker: str | None = None,
        account: str | None = None,
    ) -> Path:
        """Atomically persist a successful publication (write tmp + fsync + replace)."""
        self.root.mkdir(parents=True, exist_ok=True)
        path = self.path_for(clip_id, platform, account)
        data = {
            "clip_id": str(clip_id),
            "platform": str(platform).lower(),
            "account": str(account or "default"),
            "job_id": job_id,
            "final_path_worker": final_path_worker,
            "published_at": datetime.now(timezone.utc).isoformat(),
            "publication": publication,
        }
        fd, tmp = tempfile.mkstemp(dir=self.root, prefix=path.name, suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(data, fh, ensure_ascii=False, indent=2)
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp, path)
        except BaseException:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise
        return path
