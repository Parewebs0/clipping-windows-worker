"""Layout inset (campaign 58): screen on top + webcam window below, per shot, static."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from app.config import Settings
from app.jobs.render import RenderJob
from app.models.job import Job
from app.tools import inset_layout as il
from app.tools import reframe as rf
from app.tools.reframe import Det, inset_for_campaign, normalize_reframe, plan_reframe

W, H = 1920, 1080
WIN = (1447, 752, 395, 219)       # PBInvesting's webcam window at 1920x1080
FACE = (1565, 800, 75, 75)        # x, y, w, h inside it


def _frame(*, window: bool = True, face=FACE) -> np.ndarray:
    img = np.zeros((H, W, 3), np.uint8)
    img[:] = (90, 10, 10)                               # dark red backdrop
    img[100:820, 200:1700] = (15, 15, 20)               # chart window
    for x in range(260, 1680, 120):                     # chart grid: vertical lines left of the cam
        img[120:800, x:x + 2] = (60, 60, 60)
    if window:
        x, y, w, h = WIN
        img[y:y + h, x:x + w] = (200, 30, 30)           # red border
        img[y + 5:y + h - 5, x + 5:x + w - 5] = (205, 205, 200)  # bright room
        img[y + 120:y + h - 5, x + 60:x + 300] = (25, 25, 25)    # dark shirt
    fx, fy, fw, fh = face
    img[fy:fy + fh, fx:fx + fw] = (250, 240, 20)        # "face" (yellow)
    return img


def _yellow_faces(img) -> list[Det]:
    """Stub BlazeFace: the yellow square, as a centre-coordinate Det."""
    mask = (img[..., 0] > 230) & (img[..., 1] > 220) & (img[..., 2] < 60)
    ys, xs = np.nonzero(mask)
    if len(xs) < 50:
        return []
    x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
    return [Det(t=0.0, x=(x0 + x1) / 2, y=(y0 + y1) / 2, w=float(x1 - x0), h=float(y1 - y0))]


def test_window_is_found_from_its_border():
    img = _frame()
    face = il.Box(*FACE)
    box = il.window_around(img, face)
    assert box is not None
    for got, want in zip(box.as_list(), WIN):
        assert abs(got - want) <= 6


def test_corner_faces_only_reports_cornered_faces():
    assert len(il.corner_faces(_yellow_faces, _frame())) == 1
    centred = _frame(window=False, face=(900, 400, 140, 140))
    assert il.corner_faces(_yellow_faces, centred) == []


def test_no_border_no_window():
    # A face in a corner with no window around it (a person in a room): no inset.
    img = np.full((H, W, 3), 128, np.uint8)
    img[800:875, 1565:1640] = (250, 240, 20)
    assert il.window_around(img, il.Box(1565, 800, 75, 75)) is None


def test_stable_box_needs_agreement():
    a = il.Box(1447, 752, 395, 219)
    assert il.stable_box([a, il.Box(1450, 750, 392, 220), a, None], W) is not None
    assert il.stable_box([a, None, None, None], W) is None
    assert il.stable_box([a, il.Box(200, 100, 395, 219), il.Box(900, 400, 300, 200)], W) is None


def test_classify_shots_and_short_shot_inherits():
    shots = [(0.0, 4.0), (4.0, 4.5), (4.5, 8.0)]
    frames = [[_frame() for _ in range(4)], [], [_frame(window=False, face=(900, 400, 140, 140)) for _ in range(4)]]
    out = il.classify_shots(shots, frames, _yellow_faces, scale=2.0, frame_w=3840)
    assert out[0]["inset"] is not None and out[0]["found"] == 4
    assert out[0]["inset"].as_list()[0] == pytest.approx(WIN[0] * 2, abs=12)
    assert out[1]["short"] and out[1]["inset"] is out[0]["inset"]
    assert out[2]["inset"] is None
    spans = il.spans_from(out)
    assert [s["layout"] for s in spans] == ["inset", "center"] and spans[0]["end"] == 4.5


def test_geometry_keeps_aspect_and_fits():
    geo = il.geometry(3840, 2160, il.Box(2894, 1504, 790, 438))
    cx, cy, cw, ch = geo["cam_crop"]
    _x, cam_y, camw, camh = geo["cam"]
    _x, sy, sw, sh = geo["screen"]
    assert (sw, sh) == (1080, 608) and sy >= 0
    assert abs(cw / ch - camw / camh) < 0.01           # no stretch
    assert cam_y >= sy + sh and cam_y + camh <= 1920
    assert 2894 <= cx and cx + cw <= 2894 + 790 and 1504 <= cy and cy + ch <= 1504 + 438
    assert all(v % 2 == 0 for v in geo["cam_crop"])


def test_build_graph_all_inset_and_mixed():
    geo = il.geometry(1920, 1080, il.Box(*WIN))
    whole = il.build_graph([{"start": 0, "end": 5, "layout": "inset"}], geo, 1920, 1080)
    assert whole.startswith("[0:v]split=3") and whole.endswith("[vmain]") and "enable" not in whole
    mixed = il.build_graph([{"start": 0, "end": 2, "layout": "center"}, {"start": 2, "end": 5, "layout": "inset"}],
                           geo, 1920, 1080)
    assert "overlay=enable='between(t,2.000,5.000)'[vmain]" in mixed
    assert "crop=608:1080:656:0,scale=1080:1920" in mixed


def test_inset_for_campaign_flag():
    assert inset_for_campaign({}, 58, "58")["layout"] == "inset"
    assert inset_for_campaign({"mode": "center", "layout": "single"}, "58", " 39, 58 ")["layout"] == "inset"
    assert "layout" not in inset_for_campaign({}, 63, "58")
    assert "layout" not in inset_for_campaign({}, 58, "")                         # off by default
    assert inset_for_campaign({"layout": "split"}, 58, "58")["layout"] == "split"  # explicit choice wins
    assert "layout" not in inset_for_campaign({"inset": False}, 58, "58")
    assert inset_for_campaign({"inset": True}, 63, "")["layout"] == "inset"
    assert normalize_reframe({"layout": "inset"})["layout"] == "inset"


def _source(tmp_path: Path) -> Path:
    """2 s screen + webcam window, hard cut, 2 s plain grey (no inset), 1920x1080."""
    frame = tmp_path / "f.png"
    from PIL import Image
    Image.fromarray(_frame()).save(frame)
    out = tmp_path / "src.mp4"
    subprocess.run([
        "ffmpeg", "-y", "-v", "error",
        "-loop", "1", "-t", "2", "-r", "30", "-i", str(frame),
        "-f", "lavfi", "-i", "color=c=0x808080:size=1920x1080:rate=30:duration=2",
        "-f", "lavfi", "-i", "sine=frequency=440:duration=4",
        "-filter_complex", "[0:v]format=yuv420p[a];[1:v]format=yuv420p[b];[a][b]concat=n=2:v=1[v]",
        "-map", "[v]", "-map", "2:a", "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
        "-c:a", "aac", str(out),
    ], check=True)
    return out


def _stub_detector(monkeypatch):
    stub = MagicMock(side_effect=_yellow_faces)
    monkeypatch.setattr(rf, "_inset_detector", lambda model_dir: stub)
    return stub


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")
def test_plan_inset_per_shot_and_center_elsewhere(tmp_path, monkeypatch):
    _stub_detector(monkeypatch)
    src = _source(tmp_path)
    plan = plan_reframe(input_path=src, start=0.0, end=4.0, output_format="9:16", reframe={"layout": "inset"},
                        src_w=W, src_h=H, output_fps=30.0, work_dir=tmp_path)
    meta = plan.metadata
    assert plan.legacy is False and plan.complex_graph.endswith("[vmain]")
    assert [s["layout"] for s in meta["layouts"]] == ["inset", "center"]
    assert abs(meta["layouts"][0]["end"] - 2.0) < 0.1
    assert meta["inset"]["detected_shots"] == 1 and meta["inset"]["sampled_shots"] == 2
    for got, want in zip(meta["inset"]["window"], WIN):
        assert abs(got - want) <= 6


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")
def test_low_confidence_keeps_the_centre_crop(tmp_path, monkeypatch):
    monkeypatch.setattr(rf, "_inset_detector", lambda model_dir: MagicMock(return_value=[]))
    src = _source(tmp_path)
    plan = plan_reframe(input_path=src, start=0.0, end=4.0, output_format="9:16", reframe={"layout": "inset"},
                        src_w=W, src_h=H, output_fps=30.0, work_dir=tmp_path)
    assert plan.legacy is True and plan.metadata["fallback"] == "no_inset"
    other = plan_reframe(input_path=src, start=0.0, end=4.0, output_format="16:9", reframe={"layout": "inset"},
                         src_w=W, src_h=H, output_fps=30.0, work_dir=tmp_path)
    assert other.legacy is True and other.metadata["fallback"] == "inset_needs_9_16"


def _pixel(video: Path, t: float, x: int, y: int) -> tuple[int, int, int]:
    raw = subprocess.run(["ffmpeg", "-v", "error", "-ss", f"{t}", "-i", str(video), "-frames:v", "1",
                          "-vf", f"crop=2:2:{x - x % 2}:{y - y % 2}", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                         capture_output=True, check=True).stdout
    return raw[0], raw[1], raw[2]


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")
def test_render_job_with_campaign_flag_and_captions(tmp_path, monkeypatch):
    _stub_detector(monkeypatch)
    src = _source(tmp_path)
    settings = Settings(api_base_url="http://localhost", api_token="test", worker_id="t",
                        working_directory=str(tmp_path), reframe_inset_campaigns="58")
    job = MagicMock(spec=Job)
    job.id, job.job_type = "inset", "render"
    job.payload = {
        "input_video": str(src), "format": "9:16", "start_time": 0, "end_time": 4.0, "campaign_id": 58,
        "reframe": {"mode": "center", "layout": "single"},
        "captions": {"enabled": True, "segments": [{"start": 0.2, "end": 3.8, "text": "high of day and"}]},
    }
    result = RenderJob(settings=settings, job=job).execute()
    assert result["probe"]["width"] == 1080 and result["probe"]["height"] == 1920
    meta = result["applied"]["reframe"]
    assert meta["layout"] == "inset" and meta["legacy"] is False
    assert result["applied"]["captions"]["applied"] is True
    out = Path(result["file_path"])
    geo = meta["inset"]["geometry"]
    _x, cam_y, _w, cam_h = geo["cam"]
    # Inset shot: the enlarged webcam (bright room) sits right under the screen band.
    r, g, b = _pixel(out, 1.0, 1000, cam_y + 20)
    assert r > 150 and g > 150 and b > 150
    # Plain shot after the cut: the centre crop (grey) fills the frame.
    r, g, b = _pixel(out, 3.0, 540, 300)
    assert abs(r - 128) < 25 and abs(g - 128) < 25
    json.dumps(meta)  # metadata stays JSON-serialisable for the API
