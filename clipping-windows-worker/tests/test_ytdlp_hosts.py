"""Twitch/Kick VODs (y TikTok/Vimeo) se descargan con yt-dlp, no por HTTP.

yt-dlp, HTTP, gog y ffprobe están falseados: sin red.
"""
from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

from app.config import Settings
from app.jobs import download as download_mod
from app.models.job import Job
from app.services import file_manager as fm
from app.services.file_manager import FileManager, _needs_ytdlp

KICK_VOD = "https://kick.com/gattouz0/videos/21e34232-613c-4c6b-9e5e-413ea2bb1392"
TWITCH_VOD = "https://www.twitch.tv/videos/2898980714"


def _settings(tmp_path: Path) -> Settings:
    return Settings(
        worker_id="test-worker",
        api_base_url="http://api.example",
        api_token="test-token",
        working_directory=tmp_path,
    )


@pytest.mark.parametrize("url", [
    "https://www.youtube.com/watch?v=abc",
    "https://youtu.be/abc",
    "https://m.youtube.com/watch?v=abc",
    "https://music.youtube.com/watch?v=abc",
    TWITCH_VOD,
    "https://twitch.tv/videos/123",
    "https://m.twitch.tv/videos/123",
    KICK_VOD,
    "https://www.kick.com/x/videos/abc",
    "https://www.tiktok.com/@user/video/7300000000000000000",
    "https://vimeo.com/123456",
    "https://player.vimeo.com/video/123456",
    "HTTPS://WWW.TWITCH.TV/videos/1",
])
def test_needs_ytdlp_true(url):
    assert _needs_ytdlp(url)


@pytest.mark.parametrize("url", [
    "https://drive.google.com/uc?export=download&id=abc",
    "https://www.dropbox.com/s/abc/file.mp4?dl=1",
    "https://cdn.example.com/video.mp4",
    "https://sidekick.com/video.mp4",
    "https://notyoutube.com/watch?v=abc",
    "https://cdn.example.com/x.mp4?ref=https://kick.com/x",
    "",
    "not a url",
])
def test_needs_ytdlp_false(url):
    assert not _needs_ytdlp(url)


def test_download_routes_kick_and_twitch_to_ytdlp(monkeypatch, tmp_path: Path):
    calls = []

    def fake_ytdlp(self, url, destination):
        calls.append(("ytdlp", url))
        out = tmp_path / "v.mp4"
        out.write_bytes(b"x")
        return out

    def fake_http(self, url, destination, **kw):
        calls.append(("http", url))
        raise AssertionError("HTTP path used for a yt-dlp host")

    monkeypatch.setattr(FileManager, "download_with_ytdlp", fake_ytdlp)
    monkeypatch.setattr(FileManager, "download_http", fake_http)
    monkeypatch.setattr(FileManager, "validate_media_file", lambda self, path, **kw: None)
    mgr = FileManager(_settings(tmp_path))
    mgr.download(KICK_VOD, tmp_path)
    mgr.download(TWITCH_VOD, tmp_path)
    assert calls == [("ytdlp", KICK_VOD), ("ytdlp", TWITCH_VOD)]


def _install_fake_ytdlp(monkeypatch, seen: list):
    module = types.ModuleType("yt_dlp")
    utils = types.ModuleType("yt_dlp.utils")

    class DownloadError(Exception):
        pass

    utils.DownloadError = DownloadError

    class YoutubeDL:
        def __init__(self, opts):
            self.opts = opts

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def extract_info(self, url, download=True):
            seen.append(url)
            path = Path(self.opts["outtmpl"].replace("%(id)s", "vod123").replace("%(ext)s", "mp4"))
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"\x00\x00\x00\x18ftypisom" + b"\x00" * 64)
            return {"id": "vod123", "requested_downloads": [{"filepath": str(path)}]}

        def prepare_filename(self, info):
            return ""

    module.utils = utils
    module.YoutubeDL = YoutubeDL
    monkeypatch.setitem(sys.modules, "yt_dlp", module)
    monkeypatch.setitem(sys.modules, "yt_dlp.utils", utils)


def test_download_job_kick_vod_uses_ytdlp_not_gog(monkeypatch, tmp_path: Path):
    """El servidor manda source_id='kick:<uuid>'. Antes eso caía en gog (Drive)."""
    seen: list = []
    _install_fake_ytdlp(monkeypatch, seen)

    def no_gog(*a, **k):
        raise AssertionError("gog used for a Kick VOD")

    monkeypatch.setattr(download_mod, "_gog_download", no_gog)
    monkeypatch.setattr(fm.httpx, "stream", lambda *a, **k: (_ for _ in ()).throw(AssertionError("http")))
    monkeypatch.setattr(FileManager, "validate_media_file", lambda self, path, **kw: None)

    class FakeProbe:
        def __init__(self, settings):
            pass

        def get_duration(self, path):
            return 3600.0

    monkeypatch.setattr(download_mod, "FFprobeTool", FakeProbe)
    job = Job(id="job-kick-1", type="download", payload={
        "url": KICK_VOD, "source_id": "kick:21e34232-613c-4c6b-9e5e-413ea2bb1392", "asset_id": "a1",
    })
    result = download_mod.DownloadJob(_settings(tmp_path), job).execute()
    assert seen == [KICK_VOD]
    assert result["filename"] == "source.mp4" and result["duration_seconds"] == 3600.0
