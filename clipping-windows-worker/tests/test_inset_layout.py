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
    cam_x, cam_y, _w, cam_h = geo["cam"]
    # Inset shot: the enlarged webcam (bright room) sits right under the screen band.
    r, g, b = _pixel(out, 1.0, cam_x + 20, cam_y + 20)
    assert r > 150 and g > 150 and b > 150
    # Captions moved to the zone under the webcam (never over the face or the chart).
    zones = meta["inset"]["text_zones"]
    assert zones["applied"] and zones["caption_margin_v"] >= cam_y + cam_h
    ass = (out.parent / "overlay.ass").read_text()
    caption_style = next(line for line in ass.splitlines() if line.startswith("Style: Caption"))
    assert caption_style.split(",")[-5] == "8" and caption_style.split(",")[-2] == str(zones["caption_margin_v"])
    # Plain shot after the cut: the centre crop (grey) fills the frame.
    r, g, b = _pixel(out, 3.0, 540, 300)
    assert abs(r - 128) < 25 and abs(g - 128) < 25
    json.dumps(meta)  # metadata stays JSON-serialisable for the API


def test_content_box_finds_the_window_inside_a_decorative_frame():
    frames = [_frame() for _ in range(3)]
    box = il.content_box(frames, exclude=il.Box(*WIN))
    assert box is not None
    x, y, w, h = box.as_list()
    assert 200 <= x <= 215 and 100 <= y <= 115          # inside the chart window border, not on a grid line
    assert 1685 <= x + w <= 1700 and 805 <= y + h <= 820


def test_content_box_trims_letterbox_and_keeps_plain_frames():
    img = np.zeros((H, W, 3), np.uint8)
    rng = np.random.default_rng(1)
    img[140:940] = rng.integers(0, 255, (800, W, 3), dtype=np.uint8)   # letterboxed busy content
    box = il.content_box([img])
    assert box is not None and 140 <= box.as_list()[1] <= 150 and 780 <= box.as_list()[3] <= 800
    assert box.as_list()[0] == 0 and box.as_list()[2] == W
    busy = rng.integers(0, 255, (H, W, 3), dtype=np.uint8)
    assert il.content_box([busy]) is None                               # nothing to trim: whole frame


def test_thumbnail_is_cut_or_masked():
    band = il.Box(200, 100, 1500, 720)
    small = il.Box(1600, 700, 100, 120)                                 # cheap to cut: right side
    cut, mask, how = il.band_and_mask(band, small, W)
    assert how == "cut_right" and mask is None and cut.x + cut.w <= 1600
    big = il.Box(1447, 752, 395, 219)                                   # cutting loses too much: mask
    same, mask, how = il.band_and_mask(band, big, W)
    assert how == "mask" and same == band and mask.x < 1447 and mask.y < 752
    assert il.band_and_mask(band, il.Box(1750, 900, 100, 100), W)[2] == "none"


def test_geometry_v2_zones_and_upscale_cap():
    cam = il.Box(2894, 1504, 790, 438)
    geo = il.geometry(3840, 2160, cam, band=il.Box(430, 318, 2972, 1406), mask=il.Box(2860, 1470, 542, 254),
                      fill="101010")
    _x, sy, sw, sh = geo["screen"]
    cx, cam_y, camw, camh = geo["cam"]
    z = geo["text_zones"]
    assert sw == 1080 and sh < 608                       # cropped band at full width
    assert z["top"][1] == sy and sy >= il.TOP_TEXT and cam_y >= sy + sh
    assert z["caption"][0] == cam_y + camh and 1920 - z["caption"][0] >= il.CAPTION_ZONE
    assert geo["upscale"] <= il.CAM_UPSCALE_MAX + 1e-6 and geo["sharpen"]
    assert geo["mask"] == [2860 - 430, 1470 - 318, 542, 1724 - 1470]
    graph = il.build_graph([{"start": 0, "end": 5, "layout": "inset"}], geo, 3840, 2160)
    assert "crop=2972:1406:430:318,drawbox=" in graph and "color=0x101010" in graph and "unsharp" in graph
    small = il.geometry(1920, 1080, il.Box(*WIN))        # 1080p source: the webcam is not blown up 2.9x
    assert small["upscale"] <= il.CAM_UPSCALE_MAX + 1e-6 and small["cam"][2] < 1080


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")
def test_probe_aborts_early_without_a_window(tmp_path, monkeypatch):
    stub = MagicMock(return_value=[])
    monkeypatch.setattr(rf, "_inset_detector", lambda model_dir: stub)
    src = _source(tmp_path)
    plan = plan_reframe(input_path=src, start=0.0, end=4.0, output_format="9:16", reframe={"layout": "inset"},
                        src_w=W, src_h=H, output_fps=30.0, work_dir=tmp_path)
    meta = plan.metadata
    assert plan.legacy is True and meta["fallback"] == "no_inset"
    assert meta["inset"]["probe"]["hits"] == 0 and "scene_cuts_s" not in meta["timings"]
    assert stub.call_count <= il.PROBES * 4              # quadrants of the probe frames only


def test_overlay_margin_override():
    from app.tools.captions import build_ass
    ost = {"enabled": True, "style": {"margin_v": 77},
           "items": [{"text": "Sell at high of day", "start": 0, "end": 2, "position": "top", "entrance": True}]}
    ass, _ = build_ass(width=1080, height=1920, captions=None, on_screen_text=ost, duration=4.0)
    overlay = next(line for line in ass.splitlines() if line.startswith("Style: Overlay"))
    assert overlay.split(",")[-2] == "77" and ",77,0,280)" in ass
