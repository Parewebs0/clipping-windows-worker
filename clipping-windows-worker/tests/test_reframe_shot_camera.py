"""P0 (2026-10-22): fixed window per shot. No jitter, clean cuts, eased re-frames."""

from __future__ import annotations

import math
import random
import shutil
import subprocess
from pathlib import Path

import pytest

from app.tools.reframe import (
    Camera,
    detect_scene_cuts,
    normalize_reframe,
    plan_holds,
    plan_reframe,
    shot_origins,
)

FPS = 30.0
TIMES = [i / FPS for i in range(int(10 * FPS))]
CROP_W, CROP_H, SRC_W, SRC_H = 606, 1080, 1920, 1080


def _steps(origins):
    return [math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(origins, origins[1:])]


def _wobble(x0: float, *, amp: float, seconds: float = 10.0, seed: int = 1):
    """A presenter swaying (+-amp px) plus detector noise, sampled at 8 fps."""
    rng = random.Random(seed)
    out, t = [], 0.0
    while t < seconds - 1e-6:
        out.append((round(t, 3), x0 + amp * math.sin(2 * math.pi * 0.4 * t) + rng.uniform(-12, 12),
                    540 + rng.uniform(-8, 8)))
        t += 0.125
    return out


def test_defaults_and_parsing():
    assert normalize_reframe({"mode": "face"})["camera"].mode == "shot"
    assert normalize_reframe({"camera": "follow"})["camera"].mode == "follow"
    cam = normalize_reframe({"camera": {"mode": "x", "safe_frac_x": 9, "ease_s": "bad"}})["camera"]
    assert cam.mode == "shot" and cam.safe_frac_x == 0.5 and cam.ease_s == 0.8
    assert normalize_reframe({"scene_cuts": [2, "x", 1.5]})["scene_cuts"] == [1.5, 2.0]


def test_small_moves_inside_the_safe_zone_never_move_the_window():
    # +-60 px sway (~10 cm at webcam distance) + detector noise: the rolling
    # median stays inside 15 % of the 606 px window.
    origins, info = shot_origins(_wobble(900, amp=60), TIMES, crop_w=CROP_W, crop_h=CROP_H, src_w=SRC_W,
                                 src_h=SRC_H, camera=Camera(), max_shift=30.0)
    assert len(set(origins)) == 1
    assert info["moves"] == 0 and info["snaps"] == []


def test_leaving_the_safe_zone_is_one_eased_move_capped_per_frame():
    pts = [(t, 700.0 if t < 4 else 900.0, 540.0) for t in [i * 0.125 for i in range(80)]]
    origins, info = shot_origins(pts, TIMES, crop_w=CROP_W, crop_h=CROP_H, src_w=SRC_W, src_h=SRC_H,
                                 camera=Camera(), max_shift=30.0)
    assert info["moves"] == 1 and info["snaps"] == []
    steps = _steps(origins)
    assert max(steps) <= 0.015 * CROP_W + 1  # gentle pan: ~9 px/frame on a 606 px window
    # Accelerates and decelerates: no single-frame jump, no constant crawl.
    moving = [s for s in steps if s > 0]
    assert moving[0] < max(moving) and moving[-1] < max(moving)
    assert len(moving) >= 0.8 * FPS * 0.8 - 2  # spread over at least ease_s
    assert origins[0][0] == round(700 - CROP_W / 2) and origins[-1][0] == round(900 - CROP_W / 2)


def test_scene_cut_is_a_clean_cut_and_a_false_cut_does_not_move():
    pts = [(t, 500.0 if t < 5 else 1400.0, 540.0) for t in [i * 0.125 for i in range(80)]]
    origins, info = shot_origins(pts, TIMES, crop_w=CROP_W, crop_h=CROP_H, src_w=SRC_W, src_h=SRC_H,
                                 camera=Camera(), max_shift=30.0, cuts=[2.5, 5.0])
    steps = _steps(origins)
    assert info["snaps"] == [5.0]  # the cut at 2.5 s keeps the same framing
    jumps = [i + 1 for i, s in enumerate(steps) if s > 0]
    assert jumps == [150]  # exactly one change, on the cut frame
    assert origins[150][0] == round(1400 - CROP_W / 2)


def test_shot_without_subject_is_a_fixed_centred_window():
    pts = [(t, 400.0, 540.0) for t in [i * 0.125 for i in range(40)]]  # face only in shot 1
    origins, _ = shot_origins(pts, TIMES, crop_w=CROP_W, crop_h=CROP_H, src_w=SRC_W, src_h=SRC_H,
                              camera=Camera(), max_shift=30.0, cuts=[5.0])
    assert origins[-1] == (round(SRC_W / 2 - CROP_W / 2), 0)
    assert len(set(origins[150:])) == 1


def test_plan_holds_uses_the_median_and_merges_close_holds():
    pts = [(t, 1000.0 + (300 if 1.0 <= t < 1.25 else 0), 540.0) for t in [i * 0.125 for i in range(40)]]
    holds = plan_holds(pts, start=0.0, end=5.0, safe_x=90, safe_y=108, camera=Camera())
    assert holds == [(0.0, 1000.0, 540.0)]  # a 0.25 s outlier never re-frames


def test_face_mode_render_plan_is_static_for_a_swaying_presenter(tmp_path):
    dets = [{"t": t, "x": x, "y": y, "w": 200, "h": 260, "kind": "face", "track_id": 1}
            for t, x, y in _wobble(1000, amp=50)]
    plan = plan_reframe(input_path=Path("unused.mp4"), start=0.0, end=10.0, output_format="9:16",
                        reframe={"mode": "face", "layout": "single", "detections": dets},
                        src_w=1920, src_h=1080, output_fps=30.0, work_dir=tmp_path)
    assert plan.legacy is False
    assert plan.metadata["camera"]["moves"] == 0
    lines = plan.command_files[0].read_text(encoding="ascii").splitlines()
    assert len(lines) == 1  # one position for the whole clip
    follow = plan_reframe(input_path=Path("unused.mp4"), start=0.0, end=10.0, output_format="9:16",
                          reframe={"mode": "face", "layout": "single", "detections": dets, "camera": "follow"},
                          src_w=1920, src_h=1080, output_fps=30.0, work_dir=tmp_path / "f")
    assert len(follow.command_files[0].read_text(encoding="ascii").splitlines()) > 40  # the old tracker re-aims all the time


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")
def test_detect_scene_cuts_finds_a_hard_cut(tmp_path):
    src = tmp_path / "cut.mp4"
    subprocess.run([
        "ffmpeg", "-y", "-v", "error",
        "-f", "lavfi", "-i", "testsrc=size=320x180:rate=30:duration=2",
        "-f", "lavfi", "-i", "color=c=red:size=320x180:rate=30:duration=2",
        "-filter_complex", "[0:v][1:v]concat=n=2:v=1[v]", "-map", "[v]", "-pix_fmt", "yuv420p", str(src),
    ], check=True, capture_output=True)
    cuts = detect_scene_cuts("ffmpeg", src, 0.0, 4.0)
    assert len(cuts) == 1 and abs(cuts[0] - 2.0) <= 0.05
