"""T1: face mode follows the main person, zooms wide shots, never letterboxes."""

from __future__ import annotations

from pathlib import Path

import pytest

from app.tools.reframe import (
    Det,
    Sample,
    centered_ratio,
    main_track,
    normalize_reframe,
    plan_reframe,
    zoom_for,
)


def _plan(tmp_path: Path, duration: float = 6.0, **reframe):
    return plan_reframe(
        input_path=Path("unused.mp4"), start=0.0, end=duration, output_format="9:16",
        reframe=reframe, src_w=1920, src_h=1080, output_fps=30.0, work_dir=tmp_path,
    )


def _track(tid: int, *, x: float, y: float, h: float, duration: float, step: float = 0.125) -> list[dict]:
    out, t = [], 0.0
    while t < duration - 1e-6:
        out.append({"t": round(t, 3), "x": x, "y": y, "w": h * 0.8, "h": h, "kind": "face", "track_id": tid})
        t += step
    return out


def _graph_text(plan) -> str:
    return " ".join(plan.vf or []) + (plan.complex_graph or "")


def test_zoom_defaults_on_only_in_face_mode():
    assert normalize_reframe({"mode": "face"})["zoom"]["enabled"] is True
    assert normalize_reframe({"mode": "center"})["zoom"]["enabled"] is False
    assert normalize_reframe({"mode": "face", "zoom": False})["zoom"]["enabled"] is False
    custom = normalize_reframe({"mode": "face", "zoom": {"max_zoom": 9, "target_face_frac": 0.2}})["zoom"]
    assert custom["max_zoom"] == 3.0 and custom["target_face_frac"] == 0.2


def test_zoom_for_uses_median_face_and_caps():
    cfg = normalize_reframe({"mode": "face"})["zoom"]
    # Small face in a wide shot: wants 0.14*1080/90 = 1.68, capped at 1.6.
    assert zoom_for([80, 90, 400], 1080, cfg) == pytest.approx(1.6)
    # Face already big: no zoom.
    assert zoom_for([300, 320, 310], 1080, cfg) == 1.0
    assert zoom_for([], 1080, cfg) == 1.0
    assert zoom_for([90], 1080, dict(cfg, enabled=False)) == 1.0


def test_main_track_is_the_biggest_screen_time():
    big = [Det(t=i, x=500, y=500, w=200, h=250, track_id=1) for i in range(10)]
    small = [Det(t=i, x=1500, y=500, w=60, h=80, track_id=2) for i in range(10)]
    samples = [Sample(t=float(i), faces=[big[i], small[i]]) for i in range(10)]
    assert main_track(samples) == 1
    assert main_track([Sample(t=0.0)]) is None


def test_centered_ratio_counts_faces_in_the_middle_half():
    followed = [(0.0, 300.0, 500.0, 100.0), (1.0, 900.0, 500.0, 100.0)]
    # Window 600 wide at x=0: first face centered, second outside.
    assert centered_ratio(followed, [(0, 0), (0, 0)], [0.0, 1.0], 600, 1080) == 0.5
    assert centered_ratio([], [(0, 0)], [0.0], 600, 1080) is None


def test_wide_shot_zooms_on_the_main_person_without_bars(tmp_path):
    # Main person small and off-center (wide shot), a passer-by even smaller.
    dets = _track(1, x=1400, y=600, h=90, duration=6.0) + _track(2, x=300, y=650, h=40, duration=6.0)
    plan = _plan(tmp_path, mode="face", layout="single", detections=dets)
    meta = plan.metadata
    assert plan.legacy is False
    assert meta["main_track"] == 1
    assert meta["zoom"] == pytest.approx(1.6)
    assert meta["crop"]["h"] < 1080 and meta["crop"]["w"] < 606
    assert meta["subject_centered"] >= 0.9
    text = _graph_text(plan)
    assert "pad=" not in text and "boxblur" not in text
    assert "scale=1080:1920" in text


def test_close_up_keeps_full_height(tmp_path):
    plan = _plan(tmp_path, mode="face", layout="single", detections=_track(1, x=1000, y=500, h=420, duration=4.0),
                 duration=4.0)
    assert plan.metadata["zoom"] == 1.0
    assert plan.metadata["crop"]["h"] == 1080


def test_main_person_is_kept_when_a_stranger_appears(tmp_path):
    dets = _track(1, x=1400, y=600, h=260, duration=6.0)
    dets += [d for d in _track(2, x=400, y=600, h=200, duration=6.0) if 2.0 <= d["t"] < 4.0]
    plan = _plan(tmp_path, mode="face", layout="single", detections=dets)
    assert plan.metadata["main_track"] == 1
    assert plan.metadata["subject_centered"] >= 0.9
    for line in plan.command_files[0].read_text(encoding="ascii").splitlines():
        x = int(line.split(" crop x ")[1].split(",")[0])
        assert x > 900  # never swings to the stranger on the left


def test_face_auto_never_uses_fit_with_blurred_background(tmp_path):
    # No faces at all: auto layout would pick `fit`; face mode keeps a full crop.
    plan = _plan(tmp_path, mode="face", layout="auto", detections=[], duration=3.0)
    assert all(span["layout"] != "fit" for span in plan.metadata.get("layouts") or [])
    assert "boxblur" not in _graph_text(plan)


def test_new_shot_snaps_instead_of_panning(tmp_path):
    # Shot 1: person on the left. Hard cut at 3 s: a new person on the right.
    dets = [d for d in _track(1, x=400, y=600, h=300, duration=6.0) if d["t"] < 3.0]
    dets += [d for d in _track(2, x=1500, y=600, h=300, duration=6.0) if d["t"] >= 3.0]
    plan = _plan(tmp_path, mode="face", layout="single", detections=dets)
    meta = plan.metadata
    assert meta["cuts"] == [3.0]
    lines = plan.command_files[0].read_text(encoding="ascii").splitlines()
    xs = {float(line.split()[0]): int(line.split(" crop x ")[1].split(",")[0]) for line in lines}
    after_cut = min(t for t in xs if t >= 3.0)
    assert abs(xs[after_cut] + 303 - 1500) < 60  # already on the new person
    assert meta["max_shift_observed_px"] <= meta["max_shift_px"] + 1e-6
    assert meta["subject_centered"] >= 0.95


def test_speaker_switch_between_faces_in_shot_still_pans(tmp_path):
    both = _track(1, x=480, y=540, h=260, duration=6.0) + _track(2, x=1440, y=540, h=260, duration=6.0)
    speakers = [{"start": 0.0, "end": 3.0, "speaker": "A"}, {"start": 3.0, "end": 6.0, "speaker": "B"}]
    for d in both:
        d["mouth"] = 1.0 if (d["track_id"] == 1) == (d["t"] < 3.0) else 0.0
    plan = _plan(tmp_path, mode="face", layout="single", detections=both, speakers=speakers)
    assert plan.metadata["cuts"] == []
