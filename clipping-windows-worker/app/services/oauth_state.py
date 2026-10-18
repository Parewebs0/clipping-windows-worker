"""Local OAuth token files shared by the platform uploaders.

One JSON file per provider and account:

    <working_directory>/oauth_state/<provider>_<account>.json

Writes are atomic (temp file in the same directory, fsync, replace) and the
file mode is ``0o600``. Callers must not log the payload: it holds tokens.
"""
from __future__ import annotations

import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any

_UNSAFE = re.compile(r"[^a-z0-9._-]+")


class OAuthStateError(RuntimeError):
    """The state file could not be written."""


def slug(value: str, *, fallback: str) -> str:
    """Filename-safe lowercase slug. Never empty, never a path."""
    raw = str(value or "").strip().lower()
    cleaned = _UNSAFE.sub("_", raw).strip("._")
    return cleaned or fallback


def oauth_state_path(working_directory: Path | str, provider: str, account: str) -> Path:
    provider_slug = slug(provider, fallback="unknown")
    account_slug = slug(account, fallback="default")
    return Path(working_directory) / "oauth_state" / f"{provider_slug}_{account_slug}.json"


def load_oauth_state(
    working_directory: Path | str,
    provider: str,
    account: str,
) -> dict[str, Any] | None:
    """Return the stored object, or None when the file is missing or unreadable.

    An unreadable file is treated as 'no state' so a publish can still try the
    env token. The caller decides whether that is safe for its platform.
    """
    path = oauth_state_path(working_directory, provider, account)
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(data, dict):
        return None
    return data


def save_oauth_state(
    working_directory: Path | str,
    provider: str,
    account: str,
    payload: dict[str, Any],
) -> Path:
    """Atomically write ``payload`` and force mode ``0o600``."""
    if not isinstance(payload, dict):
        raise TypeError("oauth state payload must be a dict")
    path = oauth_state_path(working_directory, provider, account)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(payload, fh, ensure_ascii=True, separators=(",", ":"))
                fh.flush()
                os.fsync(fh.fileno())
            os.chmod(tmp_name, 0o600)
            os.replace(tmp_name, path)
            os.chmod(path, 0o600)
        except BaseException:
            try:
                os.unlink(tmp_name)
            except OSError:
                pass
            raise
    except OSError as exc:
        raise OAuthStateError(f"could not write oauth state for {slug(provider, fallback='unknown')}") from exc
    return path
