"""Per-campaign clip storage (Step 18 of architecture_flow.md).

The Worker keeps every approved clip organised in per-campaign folders::

    C:\\CODIANT\\clipping\\storage\\clips\\<campaign_id>\\
        pending_upload\\<clip_id>.mp4   <- QA pass, awaiting social-media upload
        uploaded\\<clip_id>.mp4         <- already published (future)
        archived\\<clip_id>.mp4         <- out of rotation (future)

This module is a thin wrapper around `pathlib.Path` for:

- Lazy initialisation of the three sub-folders (idempotent).
- Copying a clip into ``pending_upload/`` after QA pass.
- Moving a clip from ``pending_upload/`` to ``uploaded/`` after publish.
- Moving a clip from any of the three to ``archived/``.

The VPS only stores the resulting path. It does NOT touch the filesystem.
On the Worker side, the QA job is responsible for ``pending_upload`` and a
not-yet-written uploader is responsible for the other two transitions.

Public API:

    ClipStorage(campaign_id)         -> context manager style helper
        .ensure() -> dict[str, Path]
        .copy_to_pending_upload(src, clip_id) -> Path
        .move_to(src_clip_id, target, dst_clip_id=None) -> Path
        .path_for(clip_id, location) -> Path

Locations are restricted to the canonical set; ``location`` strings from
outside are rejected with :class:`ValueError`.
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Literal

# Carpeta raíz del storage de clips en Windows.
# Configurable vía env CLIP_STORAGE_ROOT en .env (ver app.config.Settings).
DEFAULT_CLIP_STORAGE_ROOT = Path(r"C:\CODIANT\clipping\storage\clips")

# Localizaciones válidas (alineadas con VPS clip_storage_service.VALID_LOCATIONS).
Location = Literal["pending_upload", "uploaded", "archived"]
_LOCATIONS: tuple[str, ...] = ("pending_upload", "uploaded", "archived")

logger = logging.getLogger(__name__)


class ClipStorageError(ValueError):
    """Raised when the helper is called with bad arguments or bad state."""


def _safe_filename(name: str) -> str:
    """Sanitise a string so it's safe to use as a filename on Windows.

    Conserva el ``clip_id`` literal (UUID con guiones) como nombre de archivo,
    porque es lo que el VPS espera en ``clips.final_path_worker``. Sólo
    reemplaza por ``_`` los caracteres que Windows prohíbe explícitamente en
    nombres de archivo y los caracteres de control.
    """
    forbidden = set('<>:"/\\|?*') | set(chr(c) for c in range(0x00, 0x20))
    cleaned = "".join("_" if c in forbidden else c for c in name).strip()
    if not cleaned:
        raise ClipStorageError("filename is empty after sanitisation")
    # Defensiva: bloquear traversal aunque no haya separadores.
    if cleaned in (".", "..") or "/" in cleaned or "\\" in cleaned:
        raise ClipStorageError(f"unsafe filename: {name!r}")
    return cleaned


class ClipStorage:
    """Helper around the per-campaign clip storage layout.

    Usage::

        storage = ClipStorage(settings, campaign_id=5463)
        storage.ensure()
        final = storage.copy_to_pending_upload(
            src=Path("data/jobs/<id>/output/clip.mp4"),
            clip_id=clip_uuid_str,
        )
    """

    def __init__(
        self,
        storage_root: Path | str,
        campaign_id: int | str,
        *,
        campaign_layout: bool = False,
    ) -> None:
        if campaign_id is None:
            raise ClipStorageError("campaign_id is required")
        self._root = Path(storage_root)
        self._campaign_id = str(campaign_id)
        self._campaign_dir = self._root / self._campaign_id
        # STORAGE_ROOT layout: pending_upload -> clips/pending, uploaded and
        # archived -> clips/done. Location names (API contract) do not change.
        self._campaign_layout = campaign_layout
        if campaign_layout:
            from app.utils.storage_layout import campaign_key

            try:
                campaign_key(campaign_id)
            except ValueError as exc:
                raise ClipStorageError(str(exc)) from exc

    @classmethod
    def for_settings(cls, settings, campaign_id: int | str) -> "ClipStorage":
        """STORAGE_ROOT layout when configured, legacy clip_storage_root otherwise."""
        root = getattr(settings, "storage_root", None)
        if root:
            return cls(root, campaign_id, campaign_layout=True)
        return cls(settings.clip_storage_root, campaign_id)

    def _location_dir(self, location: str) -> Path:
        if self._campaign_layout:
            from app.utils.storage_layout import CLIP_DIRS

            return self._campaign_dir.joinpath(*CLIP_DIRS[location])
        return self._campaign_dir / location

    # ------------------------------------------------------------------
    # Paths
    # ------------------------------------------------------------------

    def path_for(
        self,
        clip_id: str,
        location: str,
    ) -> Path:
        """Devuelve la ruta esperada para ``clip_id`` en ``location``.

        No toca el sistema de archivos; sólo computa la ruta. Usar
        :meth:`ensure` antes de escribir.
        """
        if location not in _LOCATIONS:
            raise ClipStorageError(
                f"location must be one of {_LOCATIONS}, got {location!r}"
            )
        safe = _safe_filename(clip_id)
        return self._location_dir(location) / f"{safe}.mp4"

    def campaign_dir(self) -> Path:
        return self._campaign_dir

    def root(self) -> Path:
        return self._root

    # ------------------------------------------------------------------
    # Folder initialisation (lazy, idempotent)
    # ------------------------------------------------------------------

    def ensure(self) -> dict[str, Path]:
        """Crea las tres subcarpetas si no existen y devuelve sus paths."""
        paths = {
            location: self._location_dir(location) for location in _LOCATIONS
        }
        for p in paths.values():
            p.mkdir(parents=True, exist_ok=True)
        logger.debug(
            "clip storage ensured",
            campaign_id=self._campaign_id,
            root=str(self._campaign_dir),
        )
        return paths

    # ------------------------------------------------------------------
    # File moves
    # ------------------------------------------------------------------

    def copy_to_pending_upload(
        self,
        src: Path | str,
        clip_id: str,
    ) -> Path:
        """Copia ``src`` a ``<root>/<campaign_id>/pending_upload/<clip_id>.mp4``.

        - ``src`` puede ser una ruta absoluta o relativa al directorio del job.
        - Idempotente: si el destino ya existe, se sobreescribe (el clip
          nuevo siempre gana; el Worker no compara checksums en esta fase).
        - Si la copia falla, propaga la excepción; el caller decide si
          continuar (QA handler sí; otros callers deben fallar ruidosamente).
        """
        source = Path(src)
        if not source.exists():
            raise FileNotFoundError(f"source clip not found: {source}")
        if not source.is_file():
            raise ClipStorageError(f"source is not a regular file: {source}")

        self.ensure()
        destination = self.path_for(clip_id, "pending_upload")

        logger.info(
            "clip %s copying to pending_upload",
            clip_id,
            src=str(source),
            dst=str(destination),
        )
        shutil.copy2(source, destination)
        return destination

    def move(
        self,
        src: Path | str,
        clip_id: str,
        target: Location,
    ) -> Path:
        """Mueve ``src`` (o ``<campaign>/pending_upload/<clip_id>.mp4`` si
        ``src`` es None) a ``<campaign>/<target>/<clip_id>.mp4``.

        Usado por el futuro módulo de upload (``pending_upload`` ->
        ``uploaded``) y por la rotación (``uploaded`` -> ``archived``).
        """
        if target not in _LOCATIONS:
            raise ClipStorageError(
                f"target must be one of {_LOCATIONS}, got {target!r}"
            )

        if src is None:
            # Default: asume que viene de pending_upload/.
            source = self.path_for(clip_id, "pending_upload")
        else:
            source = Path(src)

        if not source.exists():
            raise FileNotFoundError(f"source clip not found: {source}")

        self.ensure()
        destination = self.path_for(clip_id, target)
        destination.parent.mkdir(parents=True, exist_ok=True)

        logger.info(
            "clip %s moving to %s",
            clip_id, target,
            src=str(source),
            dst=str(destination),
        )
        shutil.move(str(source), str(destination))
        return destination