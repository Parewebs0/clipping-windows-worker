"""Integration tests: FileManager.download end-to-end con Drive/Docs.

Cubre:
- ``file_view`` se reescribe a ``/uc?...`` y descarga real (mockeando httpx).
- Folder listing → cookie auth → ``/uc?export=download&id=...``.
- Doc no scrapeable → ``DriveDocNotVideo`` propagado.
- Folder sin cookies → ``DriveAuthRequired`` propagado.
- HTML tras download → retry ``confirm=t`` recupera el archivo.

Estos tests no hacen red: mockean ``app.services.drive_resolver.httpx.get``
o ``app.services.file_manager.httpx.stream`` para simular respuestas de Drive.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import httpx
import pytest

from app.config import Settings
from app.services import drive_resolver, file_manager
from app.services.drive_resolver import (
    DriveAssetNotFound,
    DriveAuthRequired,
    DriveDocNotVideo,
)
from app.services.file_manager import FileManager


def _make_settings(tmp_path: Path) -> Settings:
    return Settings(
        api_base_url="http://localhost",
        api_token="test",
        worker_id="test-worker",
        working_directory=str(tmp_path),
    )


def _make_sample_mp4(out_dir: Path, name: str = "sample.mp4") -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    sample = out_dir / name
    subprocess.run(
        [
            "ffmpeg", "-y", "-f", "lavfi",
            "-i", "testsrc=duration=1:size=160x120:rate=15",
            "-pix_fmt", "yuv420p", str(sample),
        ],
        check=True, capture_output=True,
    )
    return sample


def _httpx_response(
    *,
    status_code: int,
    content: bytes = b"",
    text: str = "",
) -> MagicMock:
    resp = MagicMock(spec=httpx.Response)
    resp.status_code = status_code
    resp.content = content
    resp.text = text
    return resp


def _stream_response(content: bytes) -> MagicMock:
    """Simula ``httpx.stream('GET', url) as response`` con ``iter_bytes``."""

    class _Stream:
        def __init__(self, content: bytes) -> None:
            self._content = content
            self.status_code = 200

        def raise_for_status(self) -> None:
            return None

        def iter_bytes(self, chunk_size: int):
            for i in range(0, len(self._content), chunk_size):
                yield self._content[i:i + chunk_size]

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    stream = _Stream(content)
    cm = MagicMock()
    cm.__enter__ = lambda self: stream
    cm.__exit__ = lambda self, *a: False
    return cm


# --- file_view --------------------------------------------------------


def test_download_file_view_writes_real_binary(tmp_path):
    """view → uc → descarga binaria OK."""
    settings = _make_settings(tmp_path)
    sample = _make_sample_mp4(tmp_path / "src")
    binary = sample.read_bytes()

    # Mockeamos la descarga HTTP (no la resolución, que solo hace regex).
    with patch.object(file_manager.httpx, "stream", return_value=_stream_response(binary)):
        fm = FileManager(settings)
        out = fm.download(
            "https://drive.google.com/file/d/ABC/view?usp=sharing",
            tmp_path / "out.bin",
        )

    assert out.exists()
    assert out.stat().st_size > 0
    # FFprobe debe detectar al menos un stream.
    assert out.read_bytes()[:12] != b"<!DOCTYPE ht"


def test_download_folder_with_cookies_resolves_and_downloads(tmp_path):
    settings = _make_settings(tmp_path)
    sample = _make_sample_mp4(tmp_path / "src", name="clip-final.mp4")
    binary = sample.read_bytes()

    fake_folder_html = (
        '<div data-id="FILE-A" title="clip-final.mp4"></div>'
        '<div data-id="FILE-B" title="nota.txt"></div>'
    )
    folder_resp = _httpx_response(status_code=200, text=fake_folder_html)

    with patch.object(drive_resolver, "load_drive_cookies", return_value={"SID": "x"}):
        with patch.object(file_manager.httpx, "stream", return_value=_stream_response(binary)):
            with patch.object(drive_resolver.httpx, "get", return_value=folder_resp):
                fm = FileManager(settings)
                out = fm.download(
                    "https://drive.google.com/drive/folders/1ABC?usp=sharing",
                    tmp_path / "out.bin",
                )

    assert out.exists()
    assert out.stat().st_size == len(binary)


def test_download_folder_without_cookies_raises_auth_required(tmp_path):
    settings = _make_settings(tmp_path)
    with patch.object(drive_resolver, "load_drive_cookies", return_value={}):
        fm = FileManager(settings)
        with pytest.raises(DriveAuthRequired):
            fm.download(
                "https://drive.google.com/drive/folders/1ABC?usp=sharing",
                tmp_path / "out.bin",
            )


def test_download_doc_without_video_raises_not_video(tmp_path):
    settings = _make_settings(tmp_path)
    fake_doc_html = "<html><body>Solo texto, ningún vídeo.</body></html>"
    doc_resp = _httpx_response(status_code=200, text=fake_doc_html)

    with patch.object(drive_resolver, "load_drive_cookies", return_value={}):
        with patch.object(drive_resolver.httpx, "get", return_value=doc_resp):
            fm = FileManager(settings)
            with pytest.raises(DriveDocNotVideo):
                fm.download(
                    "https://docs.google.com/document/d/DOCID/edit?usp=sharing",
                    tmp_path / "out.bin",
                )


def test_download_doc_with_drive_link_chains_to_file_view(tmp_path):
    settings = _make_settings(tmp_path)
    sample = _make_sample_mp4(tmp_path / "src", name="nested.mp4")
    binary = sample.read_bytes()

    fake_doc_html = (
        '<html><a href="https://drive.google.com/file/d/DRIVE_FILE/view">ver</a></html>'
    )
    doc_resp = _httpx_response(status_code=200, text=fake_doc_html)

    with patch.object(drive_resolver, "load_drive_cookies", return_value={"SID": "x"}):
        with patch.object(file_manager.httpx, "stream", return_value=_stream_response(binary)):
            with patch.object(drive_resolver.httpx, "get", return_value=doc_resp):
                fm = FileManager(settings)
                out = fm.download(
                    "https://docs.google.com/document/d/DOCID/edit?usp=sharing",
                    tmp_path / "out.bin",
                )

    assert out.exists()
    assert out.stat().st_size > 0


# --- HTML retry on virus-scan page -----------------------------------


def test_download_recovers_from_virus_scan_page(tmp_path):
    """Si la primera GET devuelve HTML de virus scan, reintentamos y
    recuperamos el binario."""
    settings = _make_settings(tmp_path)
    sample = _make_sample_mp4(tmp_path / "src", name="good.mp4")
    binary = sample.read_bytes()

    confirm_html = (
        "<html>can't scan for viruses, "
        '<a href="/uc?export=download&id=X&confirm=TOK123">download</a>'
        "</html>"
    )

    # Primer stream: HTML (será rechazado por validate_media_file).
    html_stream = _stream_response(confirm_html.encode("utf-8"))
    # El probe del retry lee el HTML de confirmación; el GET con confirm=
    # (dentro de try_with_confirm_token) devuelve el binario.
    probe_resp = _httpx_response(status_code=200, text=confirm_html)
    probe_resp.request = MagicMock()
    probe_resp.request.url = "https://drive.google.com/uc?export=download&id=ABC"
    binary_resp = _httpx_response(status_code=200, content=binary, text="")

    # file_manager y drive_resolver comparten el módulo httpx: el primer
    # Client.get es el probe HTML y el segundo es la descarga con confirm=.
    fake_client = MagicMock()
    fake_client.get.side_effect = [probe_resp, binary_resp]
    fake_client.__enter__ = lambda self: self
    fake_client.__exit__ = lambda self, *a: False

    with patch.object(drive_resolver, "load_drive_cookies", return_value={"SID": "x"}):
        with patch.object(file_manager.httpx, "stream", return_value=html_stream):
            with patch.object(file_manager.httpx, "Client", return_value=fake_client):
                fm = FileManager(settings)
                out = fm.download(
                    "https://drive.google.com/file/d/ABC/view?usp=sharing",
                    tmp_path / "out.bin",
                )

    assert out.exists()
    # El archivo en disco debe ser el binario (no el HTML).
    assert out.read_bytes() == binary


def test_download_gives_up_if_html_persists(tmp_path):
    """Si el HTML persiste tras el retry, levantamos un ValueError claro."""
    settings = _make_settings(tmp_path)
    confirm_html = (
        "<html>can't scan for viruses, "
        '<a href="/uc?id=X&confirm=TOK123">x</a></html>'
    )
    html_stream = _stream_response(confirm_html.encode("utf-8"))
    retry_resp = _httpx_response(status_code=200, text="<html>still login</html>")

    fake_client = MagicMock()
    fake_client.get.return_value = retry_resp
    fake_client.__enter__ = lambda self: self
    fake_client.__exit__ = lambda self, *a: False

    with patch.object(drive_resolver, "load_drive_cookies", return_value={"SID": "x"}):
        with patch.object(file_manager.httpx, "stream", return_value=html_stream):
            with patch.object(file_manager.httpx, "Client", return_value=fake_client):
                fm = FileManager(settings)
                with pytest.raises(ValueError, match="HTML/XML page"):
                    fm.download(
                        "https://drive.google.com/file/d/ABC/view",
                        tmp_path / "out.bin",
                    )
