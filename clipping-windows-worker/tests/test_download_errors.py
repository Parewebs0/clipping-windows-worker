"""yt-dlp permanent failures and the body posted to /worker/jobs/{id}/fail."""
from __future__ import annotations

import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from app.config import Settings
from app.models.job import Job
from app.services.api_client import APIClient
from app.services.download_errors import (
    FAIL_MESSAGE_LIMIT,
    PermanentDownloadError,
    permanent_download_kind,
    worker_fail_body,
)
from app.services.file_manager import FileManager
from app.services.job_runner import JobRunner


def _settings(tmp_path: Path) -> Settings:
    return Settings(
        worker_id="windows-gpu-worker-01",
        api_base_url="http://api.example",
        api_token="test-token",
        working_directory=tmp_path,
    )


PRIVATE = (
    "ERROR: [youtube] 7rX5cPQrx_s: Private video. "
    "Sign in if you've been granted access to this video"
)


def test_fail_body_matches_the_api_contract():
    raw = "\x1b[0;31m" + ("frame\n" * 400) + PRIVATE + "\x1b[0m"
    assert len(raw) > FAIL_MESSAGE_LIMIT
    body = worker_fail_body(raw, permanent=True)
    assert set(body) == {"error_message", "permanent"}
    assert body["permanent"] is True
    message = body["error_message"]
    assert isinstance(message, str)
    assert 1 <= len(message) <= FAIL_MESSAGE_LIMIT
    assert "\x1b" not in message
    assert "Private video" in message


def test_transient_error_is_not_permanent_and_omits_the_flag():
    body = worker_fail_body("ERROR: HTTP Error 503")
    assert body == {"error_message": "ERROR: HTTP Error 503"}
    assert permanent_download_kind(body["error_message"]) is None


@pytest.mark.parametrize(
    "text,kind",
    [
        ("Private video. Sign in if you've been granted access", "video_private"),
        ("This video is unavailable", "video_unavailable"),
        ("Video unavailable", "video_unavailable"),
        ("This video has been removed by the uploader", "video_removed"),
        ("Join this channel to get access to members-only content", "members_only"),
        ("Sign in to confirm your age. This video may be inappropriate", "age_restricted"),
        ("ERROR: login required", "login_required"),
    ],
)
def test_permanent_markers(text, kind):
    assert permanent_download_kind(text) == kind


def test_bot_check_is_not_a_permanent_login_error():
    assert permanent_download_kind("Sign in to confirm you're not a bot") is None


def test_api_client_posts_a_bounded_body(monkeypatch, tmp_path: Path):
    posted = {}

    class _Response:
        def raise_for_status(self):
            return None

    class _Client:
        def post(self, url, params=None, json=None):
            posted["url"] = url
            posted["params"] = params
            posted["json"] = json
            return _Response()

        def close(self):
            return None

    monkeypatch.setattr("app.services.api_client.httpx.Client", lambda **kwargs: _Client())
    client = APIClient(_settings(tmp_path))
    raw = "\x1b[31m" + ("x" * 3000) + "Private video"
    assert client.fail_job("abc", raw, permanent=True) is True
    body = posted["json"]
    assert posted["url"] == "/worker/jobs/abc/fail"
    assert posted["params"]["worker_id"] == "windows-gpu-worker-01"
    assert body["permanent"] is True
    assert "\x1b" not in body["error_message"]
    assert len(body["error_message"]) <= FAIL_MESSAGE_LIMIT
    assert "Private video" in body["error_message"]


def _install_ytdlp(monkeypatch, message: str):
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

        def __exit__(self, exc_type, exc, tb):
            return False

        def extract_info(self, url, download=True):
            raise DownloadError(message)

    module.utils = utils
    module.YoutubeDL = YoutubeDL
    monkeypatch.setitem(sys.modules, "yt_dlp", module)
    monkeypatch.setitem(sys.modules, "yt_dlp.utils", utils)
    return DownloadError


def test_private_video_raises_permanent_download_error(monkeypatch, tmp_path: Path):
    _install_ytdlp(monkeypatch, "\x1b[0;31m" + PRIVATE + "\x1b[0m")
    with pytest.raises(PermanentDownloadError) as caught:
        FileManager(_settings(tmp_path)).download_with_ytdlp(
            "https://youtu.be/7rX5cPQrx_s",
            tmp_path,
        )
    assert caught.value.kind == "video_private"
    assert "Private video" in str(caught.value)
    assert "\x1b" not in str(caught.value)


def test_http_503_stays_a_retryable_download_error(monkeypatch, tmp_path: Path):
    error_type = _install_ytdlp(monkeypatch, "ERROR: unable to download video data: HTTP Error 503")
    with pytest.raises(error_type):
        FileManager(_settings(tmp_path)).download_with_ytdlp(
            "https://youtu.be/7rX5cPQrx_s",
            tmp_path,
        )


def test_job_runner_reports_permanent_failure_without_the_traceback(tmp_path: Path):
    api = MagicMock()
    api.start_job.return_value = True
    manager = MagicMock()

    class _Handler:
        def execute(self):
            raise PermanentDownloadError("video_private", "\x1b[31mPrivate video. Sign in\x1b[0m")

        def cleanup(self):
            return None

    manager.create_handler.return_value = _Handler()
    runner = JobRunner(_settings(tmp_path), api, manager)
    with pytest.raises(PermanentDownloadError):
        runner.run(Job(id="job-1", type="download", payload={}))
    api.fail_job.assert_called_once()
    args, kwargs = api.fail_job.call_args
    assert args[0] == "job-1"
    assert kwargs["permanent"] is True
    assert "Traceback" not in args[1]
    assert "Private video" in args[1]
    assert "\x1b" not in args[1]
