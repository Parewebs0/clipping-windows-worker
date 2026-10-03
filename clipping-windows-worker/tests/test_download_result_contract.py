"""Regression: download job emits canonical VPS fields (file_path, file_size)
plus legacy aliases (filename, size).
"""

from __future__ import annotations

import threading
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path

import pytest

from app.config import Settings
from app.jobs.download import DownloadJob
from app.models.job import Job
from unittest.mock import MagicMock


class _SilentHandler(SimpleHTTPRequestHandler):
    def log_message(self, *_args, **_kwargs):  # noqa: D401
        return


def _start_http_server(directory: Path) -> tuple[HTTPServer, str]:
    handler = lambda *a, **kw: _SilentHandler(*a, directory=str(directory), **kw)
    httpd = HTTPServer(("127.0.0.1", 0), handler)
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    host, port = httpd.server_address
    return httpd, f"http://{host}:{port}"


def _make_settings(tmp_path: Path) -> Settings:
    return Settings(
        api_base_url="http://localhost",
        api_token="test",
        worker_id="test-worker",
        working_directory=str(tmp_path),
        output_dir=str(tmp_path / "out"),
    )


def _make_job(job_id: str, url: str) -> Job:
    j = MagicMock(spec=Job)
    j.id = job_id
    j.payload = {"url": url}
    j.job_type = "download"
    return j


def _make_sample_mp4(out_dir: Path) -> Path:
    """Produce a tiny real MP4 with ftyp atom."""
    import subprocess
    out_dir.mkdir(parents=True, exist_ok=True)
    sample = out_dir / "sample.mp4"
    subprocess.run(
        [
            "ffmpeg", "-y", "-f", "lavfi",
            "-i", "testsrc=duration=1:size=160x120:rate=15",
            "-pix_fmt", "yuv420p", str(sample),
        ],
        check=True, capture_output=True,
    )
    return sample


def test_download_emits_canonical_and_alias_keys(tmp_path):
    settings = _make_settings(tmp_path)
    asset_dir = tmp_path / "asset"
    asset_dir.mkdir()
    sample = _make_sample_mp4(asset_dir)
    httpd, base = _start_http_server(asset_dir)
    try:
        job = _make_job("dl-contract-001", f"{base}/sample.mp4")
        dj = DownloadJob(settings=settings, job=job)
        result = dj.execute()

        # Canonical (VPS consumes these)
        assert "file_path" in result
        assert "file_size" in result
        assert Path(result["file_path"]).exists()
        assert result["file_size"] > 0

        # Aliases for backward compat
        assert "filename" in result
        assert "size" in result
        assert result["size"] == result["file_size"]
    finally:
        httpd.shutdown()


def test_download_uses_job_id_in_filename(tmp_path):
    """Stable path is downloads/{job.id}/source.ext, never the URL.

    Transcribe looks up the asset directory. The job id is the directory
    when payload.asset_id is absent; the file name stays ``source``.
    """
    settings = _make_settings(tmp_path)
    asset_dir = tmp_path / "asset"
    asset_dir.mkdir()
    _make_sample_mp4(asset_dir)
    httpd, base = _start_http_server(asset_dir)
    try:
        job = _make_job("dl-job-id-abc-123", f"{base}/sample.mp4?token=1&x=2")
        dj = DownloadJob(settings=settings, job=job)
        result = dj.execute()
        path = Path(result["file_path"])
        assert path.parent.name == "dl-job-id-abc-123"
        assert path.stem == "source"
        assert "?" not in path.name and "&" not in path.name and "=" not in path.name
    finally:
        httpd.shutdown()


def test_download_rejects_html_response(tmp_path):
    settings = _make_settings(tmp_path)
    asset_dir = tmp_path / "asset"
    asset_dir.mkdir()
    (asset_dir / "page.html").write_text(
        "<!DOCTYPE html><html><title>Google Drive</title></html>",
        encoding="utf-8",
    )
    httpd, base = _start_http_server(asset_dir)
    try:
        job = _make_job("dl-html-001", f"{base}/page.html")
        with pytest.raises(ValueError, match="HTML/XML page instead of media"):
            DownloadJob(settings=settings, job=job).execute()
    finally:
        httpd.shutdown()
