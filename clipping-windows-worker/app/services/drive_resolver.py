"""Resolución de URLs de Google Drive/Google Docs a URLs binarias descargables.

El VPS envía a veces en ``jobs.payload.url`` enlaces "de vista" o de
carpeta que NO son descargables como binario sin transformación:

- ``drive.google.com/file/d/<ID>/view?usp=sharing`` → hay que reescribir
  a ``drive.google.com/uc?export=download&id=<ID>``.
- ``drive.google.com/drive/folders/<ID>`` → hay que listar la carpeta
  (con sesión/cookies de Drive del usuario) y elegir el primer vídeo/
  .zip dentro.
- ``docs.google.com/document/d/<ID>/edit`` → NO es un vídeo. El Doc
  puede contener un enlace al vídeo real dentro de su HTML; si no, el
  job se marca como fallido con un mensaje claro.

Sin sesión de Drive, los archivos grandes devuelven HTML pidiendo login
o la página de confirmación "can't scan for viruses". Esta capa:

1. Detecta el patrón de la URL.
2. Resuelve la URL a una URL ``/uc?export=download&...`` cuando es
   posible.
3. Si tras la descarga sigue llegando HTML (página de login / virus
   scan), expone ``try_with_confirm_token`` para reintentar con
   ``confirm=t`` siguiendo los ``Location`` con cookies.

Las cookies de sesión se cargan desde:

- ``%APPDATA%\\clipping-windows-worker\\drive_session.json`` (formato:
  ``[{"name": "SID", "value": "...", "domain": ".google.com", ...}, ...]``).
- Variables de entorno ``DRIVE_COOKIES_JSON`` (mismo JSON literal),
  ``DRIVE_COOKIES_FILE`` (ruta alternativa), ``DRIVE_SID``, ``DRIVE_HSID``,
  ``DRIVE_SSID`` (atajo para los tres nombres estándar de Google).
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlencode, urlparse, urlunparse

import httpx

from app.utils.logging import get_logger

logger = get_logger("drive_resolver")

# Hosts gestionados por este resolver.
_DRIVE_HOSTS: tuple[str, ...] = (
    "drive.google.com",
    "docs.google.com",
)


# Patrones de URL ------------------------------------------------------

# drive.google.com/file/d/<ID>/view  (también /preview, /edit, etc.)
# Sin ``^`` porque también lo usamos con ``re.search`` dentro de HTML
# scrapeado de Docs (donde la URL viene precedida por ``href="...``).
_FILE_VIEW_RE = re.compile(
    r"https?://drive\.google\.com/file/d/(?P<file_id>[A-Za-z0-9_\-]+)",
    re.IGNORECASE,
)

# drive.google.com/open?id=<ID>
_FILE_OPEN_RE = re.compile(
    r"https?://drive\.google\.com/open",
    re.IGNORECASE,
)

# drive.google.com/uc?export=download&id=<ID>  /  /uc?id=<ID>
_FILE_UC_RE = re.compile(
    r"https?://drive\.google\.com/uc",
    re.IGNORECASE,
)

# drive.google.com/drive/u/N/folders/<ID>  o  /drive/folders/<ID>
_FOLDER_RE = re.compile(
    r"https?://drive\.google\.com/drive/(?:u/\d+/)?folders/(?P<folder_id>[A-Za-z0-9_\-]+)",
    re.IGNORECASE,
)

# docs.google.com/document/d/<ID>/...
_DOC_RE = re.compile(
    r"https?://docs\.google\.com/document/d/(?P<doc_id>[A-Za-z0-9_\-]+)",
    re.IGNORECASE,
)

# Extensiones que consideramos "asset" (vídeo/zip/audio/imagen).
_ASSET_EXTENSIONS: tuple[str, ...] = (
    ".mp4", ".mov", ".mkv", ".webm", ".m4v", ".avi",
    ".zip", ".rar", ".7z",
    ".mp3", ".m4a", ".wav", ".ogg", ".flac",
    ".png", ".jpg", ".jpeg", ".webp", ".gif",
)


# Errores -------------------------------------------------------------


class DriveResolverError(Exception):
    """Error genérico del resolver de Drive."""


class DriveAuthRequired(DriveResolverError):
    """Drive requiere autenticación (cookies de sesión)."""


class DriveAssetNotFound(DriveResolverError):
    """No se encontró ningún asset válido en la URL de Drive/Docs."""


class DriveDocNotVideo(DriveResolverError):
    """La URL apunta a un Google Doc y no contiene un vídeo scrapeable."""


# Tipos ---------------------------------------------------------------


@dataclass(frozen=True)
class ResolvedUrl:
    """URL ya transformada y lista para descarga binaria."""

    url: str
    cookies: dict[str, str]  # cookies de sesión para usar en la descarga
    source_pattern: str  # "file_view", "file_open", "file_uc", "folder", "doc_video"


# API pública ---------------------------------------------------------


def is_drive_url(url: str) -> bool:
    """True si la URL es de drive.google.com o docs.google.com."""
    host = (urlparse(url).hostname or "").lower()
    return host in _DRIVE_HOSTS


def resolve(url: str) -> ResolvedUrl:
    """Resuelve una URL de Drive/Docs a una URL descargable.

    Lanza ``DriveResolverError`` (o subclases) si no se puede resolver.
    """
    cookies = load_drive_cookies()

    parsed = urlparse(url)
    host = (parsed.hostname or "").lower()

    if host == "docs.google.com":
        return _resolve_doc(url, cookies)

    if host == "drive.google.com":
        m = _FILE_VIEW_RE.match(url)
        if m:
            return _resolve_file_view(url, m.group("file_id"), cookies)

        m = _FILE_OPEN_RE.match(url)
        if m:
            return _resolve_file_open(url, cookies)

        m = _FILE_UC_RE.match(url)
        if m:
            return ResolvedUrl(
                url=_strip_drive_confirm(url),
                cookies=cookies,
                source_pattern="file_uc",
            )

        m = _FOLDER_RE.match(url)
        if m:
            return _resolve_folder(url, m.group("folder_id"), cookies)

    raise DriveAssetNotFound(
        f"URL de Drive no soportada por el resolver: {url}"
    )


def try_with_confirm_token(
    response: httpx.Response,
    cookies: dict[str, str] | None,
) -> httpx.Response | None:
    """Si la respuesta es la página de confirmación "can't scan for viruses",
    reintenta con ``confirm=t`` propagando cookies y siguiendo Location.

    Devuelve el ``Response`` final si se pudo resolver, o ``None`` si la
    página de confirmación no es resoluble con esta estrategia.
    """
    if response.status_code != 200:
        return None

    body = response.text[:4096]
    is_confirm_page = (
        "can't scan for viruses" in body.lower()
        or "scan failed" in body.lower()
        or "confirm=" in body.lower() and "download" in body.lower()
    )
    if not is_confirm_page:
        return None

    # Intentar extraer el token "confirm=<XXXX>" del HTML.
    token = _extract_confirm_token(body)
    if not token:
        logger.warning("drive confirm page detected but no confirm token found")
        return None

    # Reconstruir la URL con confirm=t.
    parsed = urlparse(str(response.request.url))
    qs = parse_qs(parsed.query)
    qs["confirm"] = [token]
    qs.pop("uuid", None)
    qs.pop("at", None)
    new_query = urlencode(qs, doseq=True)
    new_url = urlunparse(parsed._replace(query=new_query))

    logger.info("retrying drive download with confirm token", url=new_url)
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
        return client.get(new_url)


# Helpers internos ----------------------------------------------------


def _resolve_file_view(
    url: str,
    file_id: str,
    cookies: dict[str, str],
) -> ResolvedUrl:
    """``/file/d/<ID>/view`` → ``/uc?export=download&id=<ID>``."""
    new_url = f"https://drive.google.com/uc?export=download&id={file_id}"
    return ResolvedUrl(
        url=new_url,
        cookies=cookies,
        source_pattern="file_view",
    )


def _resolve_file_open(
    url: str,
    cookies: dict[str, str],
) -> ResolvedUrl:
    """``/open?id=<ID>`` → ``/uc?export=download&id=<ID>``."""
    parsed = urlparse(url)
    qs = parse_qs(parsed.query)
    file_id = (qs.get("id") or [""])[0]
    if not file_id:
        raise DriveAssetNotFound(f"drive /open sin id: {url}")
    new_url = f"https://drive.google.com/uc?export=download&id={file_id}"
    return ResolvedUrl(
        url=new_url,
        cookies=cookies,
        source_pattern="file_open",
    )


def _strip_drive_confirm(url: str) -> str:
    """Quita ``confirm=`` para evitar doble confirmación en reintentos."""
    parsed = urlparse(url)
    qs = parse_qs(parsed.query)
    qs.pop("confirm", None)
    new_query = urlencode(qs, doseq=True)
    return urlunparse(parsed._replace(query=new_query))


def _resolve_folder(
    url: str,
    folder_id: str,
    cookies: dict[str, str],
) -> ResolvedUrl:
    """Lista una carpeta de Drive y devuelve el primer asset válido.

    Estrategia:
    1. Probar API JSON interna ``drive.google.com/drive/folders/<id>``
       con ``?layout=grid&hl=en`` y extraer ``data-id`` de cada item.
    2. Para cada ``data-id`` (mientras queden), pedir el ``/file/d/<id>/view``
       para quedarse con el primero cuya extensión esté en ``_ASSET_EXTENSIONS``.
    3. Si la API no devuelve nada (HTML de login), lanzar ``DriveAuthRequired``.

    Sin ``cookies``, Drive devuelve la página de login (HTML), no la lista.
    """
    if not cookies:
        raise DriveAuthRequired(
            "Listar carpetas de Drive requiere sesión. Configura "
            "%APPDATA%\\clipping-windows-worker\\drive_session.json o "
            "DRIVE_COOKIES_JSON / DRIVE_SID+DRIVE_HSID+DRIVE_SSID."
        )

    listing_url = (
        f"https://drive.google.com/drive/folders/{folder_id}"
        "?layout=grid&hl=en"
    )
    try:
        resp = httpx.get(
            listing_url,
            cookies=cookies,
            follow_redirects=True,
            timeout=30.0,
            headers={
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/126.0.0.0 Safari/537.36"
                ),
            },
        )
    except httpx.HTTPError as exc:
        raise DriveResolverError(f"Error listando carpeta Drive: {exc}") from exc

    if resp.status_code != 200:
        raise DriveAuthRequired(
            f"Drive devolvió HTTP {resp.status_code} al listar la carpeta; "
            "revisa las cookies de sesión."
        )

    head = resp.text[:1024].lower()
    looks_like_login = (
        "sign in" in head
        or "accounts.google.com" in head
        or "you must be signed in" in head
    )
    if looks_like_login:
        # Página de login real: no se puede listar sin sesión válida.
        raise DriveAuthRequired(
            "Drive devolvió página de login al listar la carpeta. "
            "Revisa las cookies de sesión."
        )

    file_ids = _extract_drive_data_ids(resp.text)
    if not file_ids:
        # A veces la carpeta está vacía o usa un layout distinto.
        raise DriveAssetNotFound(
            f"No se encontraron items en la carpeta Drive: {url}"
        )

    # Iterar candidatos: quedarnos con el primero cuya URL /file/d/<id>/view
    # tenga extensión de asset (suele venir como título del item).
    titles = _extract_drive_item_titles(resp.text)

    chosen_id: str | None = None
    for idx, file_id in enumerate(file_ids):
        title = titles[idx] if idx < len(titles) else ""
        if any(title.lower().endswith(ext) for ext in _ASSET_EXTENSIONS):
            chosen_id = file_id
            break

    if chosen_id is None:
        # Si ninguno tenía título con extensión reconocible, devolver el primero
        # igualmente — Drive puede haber listado el item sin nombre visible.
        chosen_id = file_ids[0]

    new_url = (
        f"https://drive.google.com/uc?export=download&id={chosen_id}"
    )
    logger.info(
        "drive folder resolved",
        folder_id=folder_id,
        chosen_file_id=chosen_id,
    )
    return ResolvedUrl(
        url=new_url,
        cookies=cookies,
        source_pattern="folder",
    )


def _resolve_doc(
    url: str,
    cookies: dict[str, str],
) -> ResolvedUrl:
    """Google Doc: scrapeamos el HTML publicado en busca de un vídeo.

    Si el Doc tiene un vídeo embebido (por ejemplo YouTube, Drive, o un
    MP4 directo), devolvemos la URL del vídeo. Si no, lanzamos
    ``DriveDocNotVideo``.
    """
    parsed = urlparse(url)
    qs = parse_qs(parsed.query)
    doc_id = _DOC_RE.match(url).group("doc_id") if _DOC_RE.match(url) else ""

    if not doc_id:
        raise DriveAssetNotFound(f"docs URL sin doc_id: {url}")

    # Endpoint "published" sirve HTML estático del Doc (no requiere sesión).
    published_url = (
        f"https://docs.google.com/document/d/{doc_id}/published"
    )

    try:
        resp = httpx.get(
            published_url,
            cookies=cookies or None,
            follow_redirects=True,
            timeout=30.0,
            headers={
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/126.0.0.0 Safari/537.36"
                ),
            },
        )
    except httpx.HTTPError as exc:
        raise DriveResolverError(f"Error scrapeando Doc: {exc}") from exc

    if resp.status_code != 200:
        raise DriveDocNotVideo(
            f"Google Doc devolvió HTTP {resp.status_code}; "
            "no se puede extraer vídeo (asset no es un vídeo, revisar en Whop)"
        )

    html = resp.text

    # Buscar primer MP4/WebM directo en el HTML.
    direct = re.search(
        r"https?://[^\"'<>\s]+\.(?:mp4|webm|mov|m4v)",
        html,
        re.IGNORECASE,
    )
    if direct:
        return ResolvedUrl(
            url=direct.group(0),
            cookies=cookies,
            source_pattern="doc_video",
        )

    # Buscar iframe de YouTube.
    yt = re.search(
        r"https?://(?:www\.)?youtube\.com/embed/([A-Za-z0-9_\-]+)",
        html,
        re.IGNORECASE,
    )
    if yt:
        return ResolvedUrl(
            url=f"https://www.youtube.com/watch?v={yt.group(1)}",
            cookies={},
            source_pattern="doc_video",
        )

    # Buscar link a un archivo de Drive dentro del Doc.
    drive_link = _FILE_VIEW_RE.search(html)
    if drive_link:
        return _resolve_file_view(
            drive_link.group(0),
            drive_link.group("file_id"),
            cookies,
        )

    raise DriveDocNotVideo(
        f"Google Doc {doc_id} no contiene un vídeo scrapeable "
        "(asset no es un vídeo, revisar en Whop)"
    )


def _extract_confirm_token(html: str) -> str | None:
    """Extrae el valor de ``confirm=XXXX`` de la página de virus scan."""
    m = re.search(
        r'confirm=([A-Za-z0-9_\-]+)',
        html,
        re.IGNORECASE,
    )
    return m.group(1) if m else None


def _extract_drive_data_ids(html: str) -> list[str]:
    """Extrae los ``data-id`` que Drive usa para identificar items."""
    return re.findall(r'data-id="([^"]+)"', html)


def _extract_drive_item_titles(html: str) -> list[str]:
    """Extrae títulos de items listados por Drive.

    Drive suele emitir ``<div class="Q5txwe" ... title="filename.mp4">``.
    Como el HTML cambia con frecuencia, esto es best-effort: si no se
    encuentran, el caller cae al primer ``data-id``.
    """
    return re.findall(r'title="([^"]+)"', html)


# Cookies -------------------------------------------------------------


def _drive_session_path() -> Path:
    """Devuelve la ruta estándar del JSON de cookies en Windows."""
    appdata = os.environ.get("APPDATA")
    if appdata:
        return Path(appdata) / "clipping-windows-worker" / "drive_session.json"
    # Fallback Linux/macOS (no se usa en producción, pero útil en CI).
    return Path.home() / ".config" / "clipping-windows-worker" / "drive_session.json"


def load_drive_cookies() -> dict[str, str]:
    """Carga cookies de sesión de Drive desde los lugares soportados.

    Prioridad:
    1. ``DRIVE_COOKIES_JSON`` (literal JSON en env).
    2. ``DRIVE_COOKIES_FILE`` (ruta a JSON).
    3. ``%APPDATA%\\clipping-windows-worker\\drive_session.json``.
    4. ``DRIVE_SID``/``DRIVE_HSID``/``DRIVE_SSID`` (atajo).

    Devuelve un dict ``{nombre: valor}``. Si no hay nada configurado,
    devuelve ``{}``.
    """
    raw: Any = None

    env_json = os.environ.get("DRIVE_COOKIES_JSON")
    if env_json:
        try:
            raw = json.loads(env_json)
        except json.JSONDecodeError:
            logger.warning("DRIVE_COOKIES_JSON no es JSON válido, ignorando")

    if raw is None:
        env_file = os.environ.get("DRIVE_COOKIES_FILE")
        candidates: list[Path] = []
        if env_file:
            candidates.append(Path(env_file))
        candidates.append(_drive_session_path())

        for candidate in candidates:
            if candidate.is_file():
                try:
                    raw = json.loads(candidate.read_text(encoding="utf-8"))
                    break
                except (OSError, json.JSONDecodeError) as exc:
                    logger.warning(
                        "no se pudo leer drive_session",
                        path=str(candidate),
                        error=str(exc),
                    )
                    raw = None

    cookies: dict[str, str] = {}
    if isinstance(raw, list):
        for entry in raw:
            if isinstance(entry, dict) and entry.get("name") and entry.get("value"):
                cookies[str(entry["name"])] = str(entry["value"])
    elif isinstance(raw, dict):
        cookies = {str(k): str(v) for k, v in raw.items() if v}

    if not cookies:
        # Atajo: SID/HSID/SSID como variables sueltas.
        for key in ("DRIVE_SID", "DRIVE_HSID", "DRIVE_SSID", "DRIVE_LSAP"):
            value = os.environ.get(key)
            if value:
                short = key.split("_", 1)[1]
                cookies[short] = value

    if cookies:
        logger.info("drive session cookies loaded", count=len(cookies))

    return cookies
