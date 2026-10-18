"""MOLI-73: captions_visible follows burned pixels, not a fixed bottom luma band.

The old check compared mean luma of y=height-540 against a mid-frame patch.
A black outline on a light plate lowers that mean, so 34/46 campaign clips
were reported invisible while the ASS was on the frame. Top and center styles
were outside the band, and only the first segment was sampled.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from app.config import Settings
from app.jobs.render import RenderJob, _control_band_y, caption_sample_times
from app.models.job import Job
from app.tools.captions import caption_search_window, write_ass

W, H = 540, 960


def _settings(tmp_path: Path) -> Settings:
    return Settings(
        api_base_url="http://localhost", api_token="test", worker_id="t",
        working_directory=str(tmp_path), output_dir=str(tmp_path / "out"),
    )


def _job(job_id: str, payload: dict | None = None) -> Job:
    job = MagicMock(spec=Job)
    job.id = job_id
    job.payload = payload or {}
    job.job_type = "render"
    return job


def _checker(tmp_path: Path) -> RenderJob:
    return RenderJob(settings=_settings(tmp_path), job=_job("cap-vis"))


def _burn(tmp_path: Path, lavfi: str, captions: dict | None, *, duration: float) -> Path:
    src = tmp_path / "src.mp4"
    subprocess.run(
        ["ffmpeg", "-y", "-f", "lavfi", "-i", lavfi, "-pix_fmt", "yuv420p", "-an", str(src)],
        check=True, capture_output=True,
    )
    out = tmp_path / "burned.mp4"
    if not captions:
        src.replace(out)
        return out
    ass = tmp_path / "overlay.ass"
    write_ass(ass, width=W, height=H, captions=captions, on_screen_text={}, duration=duration)
    subprocess.run(
        ["ffmpeg", "-y", "-i", src.name, "-vf", f"ass={ass.name}",
         "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p", "-an", out.name],
        check=True, capture_output=True, cwd=tmp_path,
    )
    return out


def _visible(tmp_path: Path, video: Path, captions: dict, *, duration: float) -> dict:
    probe = {"duration": duration, "width": W, "height": H}
    applied = {"captions": {"applied": True, "events": 1}}
    return _checker(tmp_path)._frame_checks(video, applied, {"captions": captions}, probe, {})


def _captions(position: str, text: str = "BOXABL foldable home", **extra) -> dict:
    style = {"position": position}
    style.update(extra)
    return {
        "enabled": True,
        "style": style,
        "segments": [{"start": 0.15, "end": 1.05, "text": text}],
    }


def test_search_window_covers_where_libass_paints_default_style():
    """Rows measured on a 1080x1920 DejaVu burn of the default style (margin 22%, size 3.8%)."""
    # Two-line bottom caption occupied y=1356..1491. Top y=426..561. Middle y=891..1026.
    bottom = caption_search_window(1080, 1920, {"position": "bottom"})
    assert bottom["y"] <= 1356 and bottom["y"] + bottom["h"] >= 1491
    top = caption_search_window(1080, 1920, {"position": "top"})
    assert top["y"] <= 426 and top["y"] + top["h"] >= 561
    for name in ("middle", "center"):
        mid = caption_search_window(1080, 1920, {"position": name})
        assert mid["y"] <= 891 and mid["y"] + mid["h"] >= 1026
    assert caption_search_window(1080, 1920, {})["position"] == "bottom"


def test_sample_times_skip_the_pause_and_spread_segments():
    gapped = [{
        "start": 0.0, "end": 1.8,
        "words": [
            {"start": 0.05, "end": 0.22, "word": "no"},
            {"start": 1.40, "end": 1.70, "word": "BOXABL"},
        ],
    }]
    # Segment midpoint is 0.9, inside the pause. The hit is the later word.
    assert caption_sample_times(gapped, 2.0) == pytest.approx([1.55])
    many = [{"start": i, "end": i + 0.4, "text": "word"} for i in (0.2, 1.2, 2.2, 3.2, 4.2)]
    times = caption_sample_times(many, 5.0)
    assert len(times) == 3
    assert times[0] == pytest.approx(0.4)
    assert times[-1] == pytest.approx(4.4)
    assert caption_sample_times([], 2.0) == []
    assert caption_sample_times(gapped, 0.0) == []


def test_invalid_samples_omit_the_flag(tmp_path):
    job = _checker(tmp_path)
    video = tmp_path / "unused.mp4"
    applied = {"captions": {"applied": True, "events": 3}}
    payload = {"captions": _captions("bottom")}
    zero = job._frame_checks(video, applied, payload, {"duration": 0, "width": W, "height": H}, {})
    assert "captions_visible" not in zero
    tiny = job._frame_checks(video, applied, payload, {"duration": 2, "width": 8, "height": 8}, {})
    assert "captions_visible" not in tiny
    no_segs = job._frame_checks(
        video, applied, {"captions": {"enabled": True, "segments": [], "style": {"position": "bottom"}}},
        {"duration": 2, "width": W, "height": H}, {},
    )
    assert "captions_visible" not in no_segs


@pytest.mark.parametrize("position", ["bottom", "middle", "center", "top"])
def test_light_plate_captions_visible_at_style_position(tmp_path, position):
    duration = 1.2
    captions = _captions(position)
    video = _burn(
        tmp_path, f"color=c=0xF2F2F2:size={W}x{H}:rate=15:duration={duration}",
        captions, duration=duration,
    )
    checks = _visible(tmp_path, video, captions, duration=duration)
    assert checks["captions_visible"] is True
    assert checks["samples"][-1]["position"] == position


def test_dark_plate_captions_stay_visible(tmp_path):
    duration = 1.2
    captions = _captions("bottom")
    video = _burn(
        tmp_path, f"color=c=black:size={W}x{H}:rate=15:duration={duration}",
        captions, duration=duration,
    )
    assert _visible(tmp_path, video, captions, duration=duration)["captions_visible"] is True


def test_no_burned_captions_is_false(tmp_path):
    duration = 1.2
    captions = _captions("bottom")
    video = _burn(
        tmp_path, f"color=c=0xF2F2F2:size={W}x{H}:rate=15:duration={duration}",
        None, duration=duration,
    )
    assert _visible(tmp_path, video, captions, duration=duration)["captions_visible"] is False


def _controls(checks: dict) -> list[float]:
    return [
        float(row["control"])
        for row in checks["samples"]
        if row.get("what") == "captions" and "control" in row
    ]


def test_noise_with_captions_does_not_block(tmp_path):
    """Fine detail saturates the control band. That must not become false."""
    duration = 1.2
    captions = _captions("bottom")
    # 8px checkerboard: every stroke window in the control band scores high.
    lavfi = (
        f"color=c=black:size={W}x{H}:rate=15:duration={duration},"
        "format=gray,geq=lum='if(mod(floor(X/8)+floor(Y/8),2),16,235)'"
    )
    video = _burn(tmp_path, lavfi, captions, duration=duration)
    checks = _visible(tmp_path, video, captions, duration=duration)
    assert checks.get("captions_visible") is not False
    assert _controls(checks) and min(_controls(checks)) >= 0.3
    assert "captions_visible" not in checks


def test_source_text_in_control_band_does_not_block(tmp_path):
    """Chat or a title already in the picture must not fail the caption check."""
    duration = 1.2
    captions = _captions("bottom")
    window = caption_search_window(W, H, captions["style"])
    control_y = _control_band_y(H, int(window["y"]), int(window["h"]))
    assert control_y is not None
    src = tmp_path / "src.mp4"
    font = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
    # Outlined lines fill the control band. No ':' in the strings: ffmpeg
    # splits a filtergraph on that character even inside text='...'.
    band_h = int(window["h"])
    draws = []
    step = 40
    for i, line in enumerate(("LIVE CHAT", "viewer lets go", "BOXABL now", "clip this")):
        y = int(control_y) + 4 + i * step
        if y + 36 > int(control_y) + band_h:
            break
        draws.append(
            "drawtext=fontfile=" + font
            + f":text='{line}':x=24:y={y}:fontsize=36:fontcolor=white:borderw=4:bordercolor=black"
        )
    assert draws
    subprocess.run(
        ["ffmpeg", "-y", "-f", "lavfi", "-i",
         f"color=c=0xF2F2F2:size={W}x{H}:rate=15:duration={duration}",
         "-vf", ",".join(draws), "-pix_fmt", "yuv420p", "-an", str(src)],
        check=True, capture_output=True,
    )
    ass = tmp_path / "overlay.ass"
    write_ass(ass, width=W, height=H, captions=captions, on_screen_text={}, duration=duration)
    out = tmp_path / "burned.mp4"
    subprocess.run(
        ["ffmpeg", "-y", "-i", src.name, "-vf", f"ass={ass.name}",
         "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p", "-an", out.name],
        check=True, capture_output=True, cwd=tmp_path,
    )
    checks = _visible(tmp_path, out, captions, duration=duration)
    controls = _controls(checks)
    assert checks.get("captions_visible") is not False
    assert controls and min(controls) >= 0.3, controls
    assert "captions_visible" not in checks


def test_busy_frame_without_captions_is_false(tmp_path):
    duration = 1.2
    captions = _captions("bottom")
    video = _burn(
        tmp_path, f"testsrc=size={W}x{H}:rate=15:duration={duration}",
        None, duration=duration,
    )
    assert _visible(tmp_path, video, captions, duration=duration)["captions_visible"] is False


def test_pause_in_the_first_segment_still_sees_the_later_word(tmp_path):
    duration = 2.0
    captions = {
        "enabled": True,
        "style": {"position": "bottom"},
        "segments": [{
            "start": 0.0, "end": 1.8, "text": "no BOXABL",
            "words": [
                {"start": 0.05, "end": 0.22, "word": "no"},
                {"start": 1.40, "end": 1.70, "word": "BOXABL"},
            ],
        }],
    }
    video = _burn(
        tmp_path, f"color=c=0xF2F2F2:size={W}x{H}:rate=15:duration={duration}",
        captions, duration=duration,
    )
    checks = _visible(tmp_path, video, captions, duration=duration)
    assert checks["captions_visible"] is True
    cap_rows = [row for row in checks["samples"] if row["what"] == "captions"]
    assert cap_rows[0]["t"] == pytest.approx(1.55, abs=0.02)


def test_render_job_light_background_reports_captions_visible(tmp_path):
    src = tmp_path / "src.mp4"
    subprocess.run(
        ["ffmpeg", "-y", "-f", "lavfi", "-i", "color=c=0xF2F2F2:size=640x360:rate=30:duration=2",
         "-f", "lavfi", "-i", "sine=frequency=440:duration=2", "-shortest", "-pix_fmt", "yuv420p", str(src)],
        check=True, capture_output=True,
    )
    payload = {
        "input_video": str(src), "format": "9:16", "start_time": 0.0, "end_time": 2.0,
        "captions": {"enabled": True, "segments": [
            {"start": 0.2, "end": 1.7, "text": "BOXABL foldable home now"},
        ]},
    }
    result = RenderJob(settings=_settings(tmp_path), job=_job("r-light", payload)).execute()
    checks = result["applied"]["frame_checks"]
    assert checks["captions_visible"] is True
    assert result["applied"]["captions"]["applied"] is True
    assert {row["what"] for row in checks["samples"]} >= {"captions", "background"}
