"""MOLI-54 — branding render contract: fontsdir filter, concat, applied block."""

from pathlib import Path

from app.tools.ffmpeg import FFmpegTool


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


def _make_lavfi_video(out: Path, duration: float = 2.0, size: str = "320x240") -> Path:
    import subprocess

    ffmpeg = "/srv/datos/hermes/work/multica/molina-bfa7372bd6af/moli-54-1a281e93522a/workdir/ffbin/ffmpeg"
    subprocess.run([ffmpeg, "-y", "-f", "lavfi",
                    "-i", f"testsrc=duration={duration}:size={size}:rate=30",
                    "-pix_fmt", "yuv420p", str(out)],
                   check=True, capture_output=True)
    return out


def test_concat_prepends_and_appends(tmp_path):
    """With cards → duration grows, order head/body/tail."""
    import os

    if not os.path.exists("/srv/datos/hermes/work/multica/molina-bfa7372bd6af/moli-54-1a281e93522a/workdir/ffbin/ffmpeg"):
        import pytest
        pytest.skip("local sandbox ffmpeg not available")
    os.environ["LD_LIBRARY_PATH"] = "/srv/datos/hermes/work/multica/molina-bfa7372bd6af/moli-54-1a281e93522a/workdir/fflib"

    body = _make_lavfi_video(tmp_path / "body.mp4", duration=3.0)
    # build a 5s intro video and a 1-frame outro image
    import subprocess

    ffmpeg = "/srv/datos/hermes/work/multica/molina-bfa7372bd6af/moli-54-1a281e93522a/workdir/ffbin/ffmpeg"
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