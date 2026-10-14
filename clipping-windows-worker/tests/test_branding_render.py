"""MOLI-54 — branding render contract: fontsdir filter, concat, applied block."""

from pathlib import Path
from unittest.mock import MagicMock

from app.config import Settings
from app.jobs.render import RenderJob, suffix_for_content_type
from app.models.job import Job
from app.tools.ffmpeg import FFmpegTool
from app.utils.subprocess import CommandError


class _Settings:
    ffmpeg_path = None
    ffprobe_path = None


def test_subtitle_filter_injects_fontsdir():
    flt = FFmpegTool._subtitle_filter("overlay.ass", fontsdir="/job/input")
    assert flt.startswith("ass=overlay.ass")
    assert "fontsdir=/job/input" in flt
    # escaped colon on Windows-style paths (C: → C\:)
    flt2 = FFmpegTool._subtitle_filter("overlay.ass", fontsdir="C:\\job\\input")
    assert "fontsdir=C\\:" in flt2 and "/job/input" in flt2


def test_subtitle_filter_without_fontsdir_unchanged():
    assert FFmpegTool._subtitle_filter("overlay.ass") == "ass=overlay.ass"
    assert FFmpegTool._subtitle_filter("subs.srt") == "subtitles=subs.srt"


def _ffmpeg_bin() -> str | None:
    import os
    import shutil

    sandbox = "/srv/datos/hermes/work/multica/molina-bfa7372bd6af/moli-54-1a281e93522a/workdir/ffbin/ffmpeg"
    if os.path.exists(sandbox):
        lib = "/srv/datos/hermes/work/multica/molina-bfa7372bd6af/moli-54-1a281e93522a/workdir/fflib"
        os.environ["LD_LIBRARY_PATH"] = lib
        return sandbox
    return shutil.which("ffmpeg")


def _make_lavfi_video(out: Path, duration: float = 2.0, size: str = "320x240") -> Path:
    import subprocess

    ffmpeg = _ffmpeg_bin()
    subprocess.run([ffmpeg, "-y", "-f", "lavfi",
                    "-i", f"testsrc=duration={duration}:size={size}:rate=30",
                    "-pix_fmt", "yuv420p", str(out)],
                   check=True, capture_output=True)
    return out


def test_concat_prepends_and_appends(tmp_path):
    """With cards → duration grows, order head/body/tail."""
    import subprocess

    ffmpeg = _ffmpeg_bin()
    if not ffmpeg:
        import pytest
        pytest.skip("ffmpeg not available")

    body = _make_lavfi_video(tmp_path / "body.mp4", duration=3.0)
    # build a 5s intro video and a 1-frame outro image
    intro = tmp_path / "intro.mp4"
    subprocess.run([ffmpeg, "-y", "-f", "lavfi",
                    "-i", "testsrc=duration=5:size=320x240:rate=30",
                    "-pix_fmt", "yuv420p", str(intro)], check=True, capture_output=True)
    outro_png = tmp_path / "outro.png"
    subprocess.run([ffmpeg, "-y", "-f", "lavfi",
                    "-i", "testsrc=duration=0.5:size=320x240:rate=30",
                    "-frames:v", "1", str(outro_png)], check=True, capture_output=True)

    from app.tools.ffprobe import FFprobeTool

    settings = _Settings()
    tool = FFmpegTool(settings)
    out = tmp_path / "joined.mp4"
    tool.concat_clips(head=intro, body=body, tail=outro_png,
                      output_path=out, max_side_seconds=1.0)
    assert out.exists() and out.stat().st_size > 0
    # intro 1s + body 3s + outro 1s ≈ 5s (±1s tolerance for encoders)
    duration = FFprobeTool(settings).get_duration(out)
    assert abs(float(duration) - 5.0) < 1.0, f"duration {duration} != ~5.0"


def test_concat_no_cards_returns_body(tmp_path):
    settings = _Settings()
    tool = FFmpegTool(settings)
    body = tmp_path / "body.mp4"
    body.write_bytes(b"x")
    assert tool.concat_clips(body=body, output_path=tmp_path / "out.mp4") == body


def test_suffix_for_content_type_ignores_parameters():
    assert suffix_for_content_type("video/mp4; charset=binary") == ".mp4"
    assert suffix_for_content_type("image/jpeg") == ".jpg"
    assert suffix_for_content_type("image/png") == ".png"
    assert suffix_for_content_type("video/quicktime") == ".mov"
    assert suffix_for_content_type("application/octet-stream") == ""
    assert suffix_for_content_type("") == ""


def _brand_job(tmp_path: Path, job_id: str, payload: dict) -> RenderJob:
    settings = Settings(
        api_base_url="http://vps",
        api_token="test",
        worker_id="test-worker",
        working_directory=str(tmp_path),
    )
    job = MagicMock(spec=Job)
    job.id = job_id
    job.payload = payload
    job.job_type = "render"
    return RenderJob(settings=settings, job=job)


class _Response:
    def __init__(self, body: bytes, content_type: str):
        self.content = body
        self.headers = {"content-type": content_type}

    def raise_for_status(self):
        return None


def _patch_httpx(monkeypatch, body: bytes, content_type: str):
    import httpx

    class _Client:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def get(self, url, headers=None):
            return _Response(body, content_type)

    monkeypatch.setattr(httpx, "Client", _Client)


def test_brand_card_without_url_suffix_uses_content_type(tmp_path, monkeypatch):
    """`/branding/intro` has no extension. video/mp4 must be saved as .mp4, not .bin."""
    _patch_httpx(monkeypatch, b"\x00\x00\x00\x18ftypmp42", "video/mp4")
    job = _brand_job(tmp_path, "card-mp4", {})
    path = job._download_brand("/worker/campaigns/1/branding/intro", ("intro",))
    assert path.name == "intro.mp4"
    assert path.suffix == ".mp4"
    assert path.read_bytes().startswith(b"\x00\x00\x00\x18ftyp")


def test_brand_card_image_content_type(tmp_path, monkeypatch):
    _patch_httpx(monkeypatch, b"\x89PNG\r\n", "image/png")
    job = _brand_job(tmp_path, "card-png", {})
    path = job._download_brand("/worker/campaigns/1/branding/outro", ("outro",))
    assert path.name == "outro.png"


def test_brand_url_suffix_wins_over_content_type(tmp_path, monkeypatch):
    _patch_httpx(monkeypatch, b"webm", "application/octet-stream")
    job = _brand_job(tmp_path, "card-url", {})
    path = job._download_brand("http://cdn.example/cards/intro.webm", ("intro",))
    assert path.name == "intro.webm"


def test_concat_failure_keeps_clip_and_records_cards_error(tmp_path, monkeypatch):
    src = tmp_path / "src.mp4"
    src.write_bytes(b"src")
    job = _brand_job(tmp_path, "brand-concat", {
        "input_video": str(src),
        "format": "9:16",
        "start_time": 0.0,
        "end_time": 4.0,
        "branding": {"intro": {"url": "/worker/campaigns/1/branding/intro", "max_seconds": 1.5}},
    })

    def fake_download(self, url, hint):
        path = self.directory.input / "intro.mp4"
        path.write_bytes(b"intro")
        return path

    def fake_render(self, *, output_path, **kwargs):
        Path(output_path).write_bytes(b"clip-body")
        return output_path

    def fake_concat(self, *, output_path, **kwargs):
        Path(output_path).write_bytes(b"partial-join")
        raise CommandError(["ffmpeg", "concat"], 1, "Invalid data found when processing input")

    monkeypatch.setattr(RenderJob, "_download_brand", fake_download)
    monkeypatch.setattr("app.tools.ffmpeg.FFmpegTool.render_clip", fake_render)
    monkeypatch.setattr(
        "app.tools.ffmpeg.FFmpegTool.prepare_loudnorm",
        lambda self, *a, **k: ({"applied": False}, None),
    )
    monkeypatch.setattr("app.tools.ffprobe.FFprobeTool.get_audio_stream", lambda self, source: None)
    monkeypatch.setattr("app.tools.ffmpeg.FFmpegTool.concat_clips", fake_concat)
    monkeypatch.setattr(RenderJob, "_probe", lambda self, path: {})
    monkeypatch.setattr(RenderJob, "_frame_checks", lambda *a, **k: {})

    result = job.execute()
    clip = Path(result["file_path"])
    assert clip.read_bytes() == b"clip-body"
    assert result["duration_seconds"] == 4.0
    branding = result["applied"]["branding"]
    assert branding["cards"] == ["intro"]
    assert "Invalid data found" in branding["cards_error"]
    assert not clip.with_name(f"_branded_{clip.name}").exists()