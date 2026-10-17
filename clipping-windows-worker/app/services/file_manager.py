"""Gestión de archivos: descarga con streaming y hashing."""

from __future__ import annotations

import hashlib
from pathlib import Path
from urllib.parse import urlparse

import httpx

from app.config import Settings
from app.services.download_errors import (
    PermanentDownloadError,
    permanent_download_kind,
    strip_ansi,
)
from app.services.drive_resolver import (
    DriveAssetNotFound,
    DriveAuthRequired,
    DriveDocNotVideo,
    DriveResolverError,
    is_drive_url,
    resolve as resolve_drive_url,
    try_with_confirm_token,
)
from app.tools.ffprobe import FFprobeTool
from app.utils.logging import get_logger

logger = get_logger("file_manager")

# Tamaño de chunk para streaming (1 MB).
_CHUNK_SIZE = 1024 * 1024

# Hosts que requieren un extractor específico (yt-dlp).
# YouTube no expone el MP4 vía GET directo: responde con la watch page HTML.
# Lo mismo pasa con los VODs de Twitch/Kick (HLS) y las páginas de TikTok/Vimeo.
# Se compara con el hostname (exacto o subdominio: www., m., player., ...),
# no como subcadena, para que p.ej. "sidekick.com" no cuente como kick.com.
_YT_DLP_HOSTS: tuple[str, ...] = (
    "youtube.com",
    "youtu.be",
    "www.youtube.com",
    "m.youtube.com",
    "music.youtube.com",
    "twitch.tv",
    "kick.com",
    "tiktok.com",
    "vimeo.com",
)


def _needs_ytdlp(url: str) -> bool:
    """True si la URL apunta a un host que necesita yt-dlp."""
    try:
        host = (urlparse((url or "").strip()).hostname or "").lower()
    except ValueError:
        return False
    return any(host == h or host.endswith("." + h) for h in _YT_DLP_HOSTS)


class FileManager:
    """Descarga archivos grandes por streaming y calcula su SHA256."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def download(
        self,
        url: str,
        destination: str | Path,
        *,
        timeout: float = 3600.0,
    ) -> Path:
        """Descarga un archivo y devuelve la ruta local.

        Enruta automáticamente:
        - YouTube, Twitch, Kick, TikTok, Vimeo → ``yt_dlp.YoutubeDL`` (descarga el vídeo real).
        - Google Drive / Google Docs → ``drive_resolver`` reescribe la URL
          (view→uc, listado de carpetas, scraping de Docs) y aplica cookies
          de sesión si están configuradas.
        - Resto de URLs → ``httpx.stream`` (streaming directo).

        Para YouTube, ``destination`` actúa como **plantilla**: si no tiene
        ``%(ext)s`` se añade para que yt-dlp pueda estampar la extensión real.
        """
        cookies: dict[str, str] | None = None
        if is_drive_url(url):
            try:
                resolved = resolve_drive_url(url)
            except (DriveAuthRequired, DriveAssetNotFound, DriveDocNotVideo):
                # Re-raise con mensaje accionable: los tipos ya son
                # suficientemente explícitos para el operador.
                raise
            except DriveResolverError as exc:
                raise ValueError(f"Drive resolver error: {exc}") from exc

            logger.info(
                "drive url resolved",
                original=url,
                resolved=resolved.url,
                pattern=resolved.source_pattern,
            )
            url = resolved.url
            cookies = resolved.cookies or None

            # Si el resolver devolvió una URL de YouTube (Doc→YT), enrutar a yt-dlp.
            if _needs_ytdlp(url):
                path = self.download_with_ytdlp(url, destination)
            else:
                path = self.download_http(
                    url, destination, timeout=timeout, cookies=cookies
                )

            # Si tras la descarga todavía es HTML (página de confirmación de
            # virus scan), reintentar con confirm=t.
            try:
                self.validate_media_file(path, source_url=url, cookies=cookies)
            except ValueError as exc:
                if "HTML/XML page" not in str(exc):
                    raise
                recovered = self._retry_html_download(
                    url=url,
                    destination=destination,
                    cookies=cookies,
                    timeout=timeout,
                )
                if recovered is None:
                    raise
                path = recovered

            return path

        if _needs_ytdlp(url):
            path = self.download_with_ytdlp(url, destination)
        else:
            path = self.download_http(url, destination, timeout=timeout)

        self.validate_media_file(path, source_url=url)
        return path

    def download_http(
        self,
        url: str,
        destination: str | Path,
        *,
        timeout: float = 3600.0,
        cookies: dict[str, str] | None = None,
    ) -> Path:
        """Descarga un archivo por streaming HTTP sin cargarlo en memoria."""
        dest = Path(destination)
        dest.parent.mkdir(parents=True, exist_ok=True)

        logger.info("downloading file (http)", url=url, destination=str(dest))
        with httpx.stream(
            "GET",
            url,
            timeout=timeout,
            follow_redirects=True,
            cookies=cookies or None,
        ) as response:
            response.raise_for_status()
            with open(dest, "wb") as f:
                for chunk in response.iter_bytes(chunk_size=_CHUNK_SIZE):
                    f.write(chunk)

        logger.info("download complete", url=url, size=dest.stat().st_size)
        return dest

    def download_with_ytdlp(
        self,
        url: str,
        destination: str | Path,
    ) -> Path:
        """Descarga un vídeo usando ``yt-dlp`` (YouTube y similares).

        ``destination`` puede ser un directorio (se usa un nombre derivado
        del título) o una plantilla con ``%(ext)s``. Tras la descarga,
        se devuelve la ruta final del fichero con su extensión real.

        Si el destino pedido no existe tras la descarga, se busca el
        archivo más reciente en el directorio padre y, si es único y
        procede claramente de esta descarga, se renombra al destino.
        """
        # Importación local para que el resto del módulo no dependa de yt-dlp.
        import yt_dlp

        dest = Path(destination)
        dest.parent.mkdir(parents=True, exist_ok=True)
        search_dir = dest if dest.is_dir() else dest.parent

        # Si es un directorio, yt-dlp lo trata como tal y nombra el archivo.
        # Si es una ruta sin %(ext)s, se le añade para evitar .bin genérico.
        out_template = str(dest)
        if dest.is_dir() or "%(ext)s" not in out_template:
            if dest.is_dir():
                out_template = str(dest / "%(id)s.%(ext)s")
            else:
                # Sustituir la extensión (incluyendo .bin) por plantilla.
                out_template = str(dest.with_suffix("")) + ".%(ext)s"

        ydl_opts: dict = {
            "outtmpl": out_template,
            "quiet": True,
            "no_warnings": True,
            "noprogress": True,
            "merge_output_format": "mp4",
            "format": "bv*+ba/b",
        }

        logger.info("downloading file (yt-dlp)", url=url, template=out_template)

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            try:
                info = ydl.extract_info(url, download=True)
            except yt_dlp.utils.DownloadError as exc:
                text = strip_ansi(str(exc)).strip() or "yt-dlp download failed"
                kind = permanent_download_kind(text)
                if kind:
                    raise PermanentDownloadError(kind, text[:500]) from exc
                raise
            if isinstance(info, dict) and "entries" in info and info["entries"]:
                info = info["entries"][0]

            # Camino 1: yt-dlp nos dice exactamente dónde quedó.
            requested = info.get("requested_downloads") or []
            candidate: Path | None = None
            if requested:
                candidate = Path(requested[0].get("filepath") or "")

            # Camino 2: predecir por nombre, pero yt-dlp a veces cambia
            # la extensión tras el merge (webm → mp4/mkv).
            if candidate is None or not candidate.exists():
                predicted = Path(ydl.prepare_filename(info))
                if predicted.exists():
                    candidate = predicted

        # Camino 3: buscar el archivo más reciente en el directorio.
        # Esto cubre el caso de merges que dejan el archivo en otro nombre.
        if candidate is None or not candidate.exists():
            files = sorted(
                search_dir.glob("*"),
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )
            files = [f for f in files if f.is_file()]
            if len(files) == 1:
                candidate = files[0]
            elif files:
                # Tomar el más reciente que no sea .bin o .part si hay otros.
                non_partial = [
                    f for f in files
                    if not f.suffix in (".part", ".ytdl")
                ]
                candidate = non_partial[0] if non_partial else files[0]

        if candidate is None or not candidate.exists():
            raise FileNotFoundError(
                f"yt-dlp reported success but no file was found in {search_dir}"
            )

        final_path = candidate

        # Si el destino solicitado es un archivo concreto que no existe y
        # yt-dlp dejó otro nombre, renombrar para preservar el contrato.
        if (
            not dest.is_dir()
            and dest != final_path
            and not dest.exists()
        ):
            dest.parent.mkdir(parents=True, exist_ok=True)
            final_path.rename(dest)
            final_path = dest

        logger.info(
            "download complete (yt-dlp)",
            url=url,
            path=str(final_path),
            size=final_path.stat().st_size,
        )
        return final_path

    def validate_media_file(
        self,
        path: str | Path,
        *,
        source_url: str | None = None,
        cookies: dict[str, str] | None = None,
    ) -> None:
        """Reject HTML/error pages saved as successful media downloads.

        Si ``source_url`` y ``cookies`` están disponibles y la página es
        la de "can't scan for viruses" de Drive, reintenta in-place con
        ``confirm=t`` antes de marcar como fallo.
        """
        media_path = Path(path)
        if not media_path.is_file() or media_path.stat().st_size == 0:
            raise ValueError(f"Downloaded file is empty or missing: {media_path}")

        with media_path.open("rb") as file:
            prefix = file.read(512).lstrip().lower()

        if prefix.startswith((b"<!doctype html", b"<html", b"<?xml")):
            hint = (
                " (Drive devolvió página de confirmación/virus scan; "
                "asegúrate de que las cookies de sesión están configuradas)"
                if source_url and "drive.google.com" in source_url
                else ""
            )
            raise ValueError(
                f"Downloaded URL returned an HTML/XML page instead of media: "
                f"{media_path}{hint}"
            )

        try:
            probe = FFprobeTool(self.settings).probe(media_path)
        except Exception as exc:  # noqa: BLE001 - normalize tool errors at boundary
            raise ValueError(
                f"Downloaded file is not a readable media container: {media_path}"
            ) from exc

        streams = probe.get("streams", [])
        if not any(stream.get("codec_type") in {"video", "audio"} for stream in streams):
            raise ValueError(
                f"Downloaded file contains no audio or video stream: {media_path}"
            )

    def _retry_html_download(
        self,
        *,
        url: str,
        destination: str | Path,
        cookies: dict[str, str] | None,
        timeout: float,
    ) -> Path | None:
        """Si la primera descarga cayó en una página de confirmación de Drive,
        reintenta con ``confirm=t`` y reemplaza ``destination`` in-place.
        Devuelve la nueva ``Path`` si tuvo éxito, o ``None`` si no se pudo.
        """
        # Volver a hacer GET a la URL original propagando cookies y leyendo
        # el HTML, para luego extraer el token. (Más barato que re-parsear el
        # .bin ya guardado.)
        try:
            with httpx.Client(
                follow_redirects=True,
                timeout=60.0,
                cookies=cookies or None,
                headers={
                    "User-Agent": (
                        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                        "AppleWebKit/537.36 (KHTML, like Gecko) "
                        "Chrome/126.0.0.0 Safari/537.36"
                    ),
                },
            ) as client:
                probe_response = client.get(url)
                retry_response = try_with_confirm_token(probe_response, cookies)
        except httpx.HTTPError as exc:
            logger.warning("drive confirm-retry http error", error=str(exc))
            return None

        if retry_response is None or retry_response.status_code != 200:
            return None

        # Verificar que esta vez sí es binario.
        head = retry_response.content[:512].lstrip().lower()
        if head.startswith((b"<!doctype html", b"<html", b"<?xml")):
            logger.warning(
                "drive confirm-retry still returned HTML; giving up",
                url=url,
            )
            return None

        dest = Path(destination)
        dest.parent.mkdir(parents=True, exist_ok=True)
        with open(dest, "wb") as f:
            f.write(retry_response.content)

        logger.info(
            "drive confirm-retry recovered download",
            url=url,
            size=dest.stat().st_size,
        )

        try:
            self.validate_media_file(dest)
        except ValueError:
            return None
        return dest

    @staticmethod
    def sha256(path: str | Path) -> str:
        """Calcula el SHA256 de un archivo por streaming."""
        hasher = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(_CHUNK_SIZE), b""):
                hasher.update(chunk)
        return hasher.hexdigest()

    @staticmethod
    def file_size(path: str | Path) -> int:
        """Devuelve el tamaño en bytes de un archivo."""
        return Path(path).stat().st_size
