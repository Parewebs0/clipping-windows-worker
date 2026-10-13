"""Reframe: historical center crop, speaker follow, and layout graphs."""

from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from app.config import Settings
from app.jobs.render import RenderJob
from app.models.job import Job
from app.tools.ffmpeg import FFmpegTool
from app.tools.reframe import (
    Smoothing,
    classify_sample,
    coalesce_layouts,
    map_speakers,
    normalize_reframe,
    output_size,
    plan_reframe,
    smooth_origins,
    square_center_filters,
)
from app.tools.reframe import Sample, Det
from app.tools.whisperx import WhisperXTool


def _settings(tmp_path: Path) -> Settings:
    return Settings(
        api_base_url="http://localhost",
        api_token="test",
        worker_id="t",
        working_directory=str(tmp_path),
    )


def _job(job_id: str, payload: dict) -> Job:
    job = MagicMock(spec=Job)
    job.id = job_id
    job.payload = payload
    job.job_type = "render"
    return job


def _silent(tmp_path: Path, name: str, *, w: int, h: int, duration: float) -> Path:
    out = tmp_path / name
    subprocess.run(
        [
            "ffmpeg", "-y", "-f", "lavfi",
            "-i", f"color=c=0x404040:size={w}x{h}:rate=30:duration={duration}",
            "-pix_fmt", "yuv420p", str(out),
        ],
        check=True, capture_output=True,
    )
    return out


def _interview_source(tmp_path: Path, duration: float = 3.0) -> Path:
    """640x360. Red block on the left speaker, green block on the right, gray gap."""
    out = tmp_path / "interview.mp4"
    subprocess.run(
        [
            "ffmpeg", "-y",
            "-f", "lavfi", "-i", f"color=c=0x808080:size=640x360:rate=30:duration={duration}",
            "-f", "lavfi", "-i", f"color=c=red:size=260x360:rate=30:duration={duration}",
            "-f", "lavfi", "-i", f"color=c=green:size=260x360:rate=30:duration={duration}",
            "-filter_complex", "[0:v][1:v]overlay=30:0[a];[a][2:v]overlay=350:0",
            "-pix_fmt", "yuv420p", str(out),
        ],
        check=True, capture_output=True,
    )
    return out


def _mean_rgb(video: Path, t: float) -> tuple[float, float, float]:
    proc = subprocess.run(
        [
            "ffmpeg", "-v", "error", "-ss", f"{t:.3f}", "-i", str(video),
            "-frames:v", "1", "-vf", "scale=24:24:flags=area",
            "-f", "rawvideo", "-pix_fmt", "rgb24", "-",
        ],
        check=True, capture_output=True,
    )
    data = proc.stdout
    n = len(data) // 3
    assert n > 0
    return (
        sum(data[0::3]) / n,
        sum(data[1::3]) / n,
        sum(data[2::3]) / n,
    )


def _two_faces(duration: float, switch: float, step: float = 0.125) -> list[dict]:
    dets = []
    t = 0.0
    while t < duration - 1e-6:
        left = t < switch
        dets.append({
            "t": round(t, 3), "x": 480, "y": 540, "w": 200, "h": 260,
            "kind": "face", "track_id": 1, "mouth": 1.0 if left else 0.0,
        })
        dets.append({
            "t": round(t, 3), "x": 1440, "y": 540, "w": 200, "h": 260,
            "kind": "face", "track_id": 2, "mouth": 0.0 if left else 1.0,
        })
        t += step
    return dets


def _speakers(duration: float, switch: float) -> list[dict]:
    return [
        {"start": 0.0, "end": switch, "speaker": "A"},
        {"start": switch, "end": duration, "speaker": "B"},
    ]


def _plan(tmp_path: Path, **reframe):
    duration = float(reframe.pop("duration", 20.0))
    fmt = reframe.pop("fmt", "9:16")
    src = reframe.pop("src", (1920, 1080))
    return plan_reframe(
        input_path=Path("unused.mp4"),
        start=0.0,
        end=duration,
        output_format=fmt,
        reframe=reframe,
        src_w=src[0],
        src_h=src[1],
        output_fps=30.0,
        work_dir=tmp_path,
    )


LEGACY_9_16 = (
    "crop=ih*9/16:ih,"
    "scale=1080:1920:force_original_aspect_ratio=decrease,"
    "pad=1080:1920:(ow-iw)/2:(oh-ih)/2"
)
LEGACY_16_9 = (
    "crop=iw:iw*9/16,"
    "scale=1920:1080:force_original_aspect_ratio=decrease,"
    "pad=1920:1080:(ow-iw)/2:(oh-ih)/2"
)


def test_output_size_and_square_filters():
    assert output_size("9:16") == (1080, 1920)
    assert output_size("1:1") == (1080, 1080)
    assert output_size("16:9") == (1920, 1080)
    with pytest.raises(ValueError, match="Unsupported output_format"):
        output_size("4:5")
    assert square_center_filters()[0].startswith("crop=min(iw\\,ih)")
    assert "scale=1080:1080" in square_center_filters()[1]


def test_normalize_clamps_and_rejects_unknown_mode():
    cfg = normalize_reframe({"mode": "nope", "layout": "nope", "sample_fps": 30, "smoothing": 0.2})
    assert cfg["mode"] == "center"
    assert cfg["layout"] == "single"
    assert cfg["sample_fps"] == 10.0
    assert cfg["smoothing"].ema == pytest.approx(0.2)
    assert "detections" not in cfg
    kept = normalize_reframe({"mode": "face", "detections": []})
    assert kept["detections"] == []


def test_center_plan_is_legacy_and_does_not_write_a_command(tmp_path):
    plan = _plan(tmp_path, mode="center")
    assert plan.legacy is True
    assert plan.vf is None and plan.complex_graph is None
    assert plan.command_files == []
    assert plan.metadata["fallback"] is None
    assert plan.metadata["detector"] == "none"
    missing = _plan(tmp_path / "b", mode="center", layout="auto")
    assert missing.legacy is True


def test_legacy_ffmpeg_command_is_unchanged(tmp_path):
    tool = FFmpegTool(_settings(tmp_path))
    src = tmp_path / "in.mp4"
    src.write_bytes(b"in")

    def grab(fmt: str) -> list[str]:
        seen: dict[str, list[str]] = {}

        def fake(command, timeout=None, cwd=None):
            seen["cmd"] = list(command)
            Path(command[-1]).write_bytes(b"out")

        from app.tools import ffmpeg as ffmpeg_mod
        original = ffmpeg_mod.run_command
        ffmpeg_mod.run_command = fake
        try:
            tool.render_clip(
                input_path=src,
                output_path=tmp_path / f"{fmt.replace(':', '')}.mp4",
                start=0.0,
                end=1.0,
                output_format=fmt,
                output_fps=30.0,
            )
        finally:
            ffmpeg_mod.run_command = original
        return seen["cmd"]

    tail = [
        "-r", "30.000", "-c:v", "libx264", "-preset", "medium", "-crf", "23",
        "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "128k", "-movflags", "+faststart",
    ]
    cmd_9 = grab("9:16")
    assert cmd_9[:6] == ["ffmpeg", "-y", "-ss", "0.000", "-i", str(src)]
    assert cmd_9[6:8] == ["-t", "1.000"]
    assert cmd_9[8:10] == ["-vf", LEGACY_9_16]
    assert cmd_9[10:-1] == tail
    cmd_16 = grab("16:9")
    assert cmd_16[8:10] == ["-vf", LEGACY_16_9]
    cmd_1 = grab("1:1")
    assert "crop=min(iw\\,ih):min(iw\\,ih)" in cmd_1[9]


def test_two_speaker_interview_stays_in_frame_without_jerks(tmp_path):
    duration, switch = 20.0, 5.0
    plan = _plan(
        tmp_path,
        mode="face",
        layout="single",
        duration=duration,
        detections=_two_faces(duration, switch),
        speakers=_speakers(duration, switch),
    )
    meta = plan.metadata
    assert plan.legacy is False
    assert meta["speaker_map"] == {"A": 1, "B": 2}
    assert meta["speaker_in_frame"] >= 0.90
    assert meta["max_shift_observed_px"] <= meta["max_shift_px"] + 1e-6
    assert meta["max_shift_px"] == pytest.approx(0.05 * min(meta["crop"]["w"], meta["crop"]["h"]))
    text = plan.command_files[0].read_text(encoding="ascii")
    assert text.startswith("0.000 crop x ")
    # The window has to leave the first speaker. First and last x differ.
    first_x = int(text.split(" crop x ")[1].split(",")[0])
    last_x = int(text.strip().splitlines()[-1].split(" crop x ")[1].split(",")[0])
    assert last_x - first_x > 400


def test_tiny_shift_cap_can_miss_the_speaker_and_still_bounds_the_step(tmp_path):
    duration, switch = 8.0, 1.0
    plan = _plan(
        tmp_path,
        mode="face",
        layout="single",
        duration=duration,
        smoothing={"ema": 1.0, "deadzone_frac": 0.0, "max_shift_frac": 0.001},
        detections=_two_faces(duration, switch),
        speakers=_speakers(duration, switch),
    )
    meta = plan.metadata
    assert meta["speaker_in_frame"] < 0.90
    assert meta["max_shift_observed_px"] <= meta["max_shift_px"] + 1e-6


def test_deadzone_ignores_subthreshold_jitter():
    smoothing = Smoothing(ema=1.0, deadzone_frac=0.03, max_shift_frac=0.05)
    targets = [(i / 30.0, 300.0 + (1.0 if i % 2 else -1.0), 400.0) for i in range(30)]
    times = [i / 30.0 for i in range(30)]
    origins, _cap = smooth_origins(
        targets, times, crop_w=200, crop_h=200, src_w=1000, src_h=1000, smoothing=smoothing,
    )
    assert origins[1:] == [origins[0]] * (len(origins) - 1)
    jump = [(0.0, 100.0, 500.0), (1.0, 800.0, 500.0)]
    moved, cap = smooth_origins(
        jump, times, crop_w=200, crop_h=200, src_w=1000, src_h=1000, smoothing=smoothing,
    )
    steps = [
        ((b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2) ** 0.5
        for a, b in zip(moved, moved[1:])
    ]
    assert steps and max(steps) <= cap + 1e-6


def test_layout_classification_and_hold():
    face = Det(t=0, x=960, y=540, w=80, h=100, kind="face", track_id=1)
    other = Det(t=0, x=800, y=200, w=80, h=100, kind="face", track_id=2)
    screen = Det(t=0, x=500, y=300, w=700, h=500, kind="object", label="laptop")
    corner = Det(t=0, x=40, y=40, w=60, h=70, kind="face", track_id=1)
    assert classify_sample(Sample(0, [face, other], []), src_w=1920, src_h=1080, two_speakers=True) == "split"
    assert classify_sample(Sample(0, [face], [screen]), src_w=1920, src_h=1080, two_speakers=False) == "screenshare"
    assert classify_sample(Sample(0, [corner], []), src_w=1920, src_h=1080, two_speakers=False) == "screenshare"
    assert classify_sample(Sample(0, [], []), src_w=1920, src_h=1080, two_speakers=False) == "fit"
    assert classify_sample(Sample(0, [face], []), src_w=1920, src_h=1080, two_speakers=False) == "single"

    labels = [(i * 0.25, "split") for i in range(5)]
    labels += [(1.25 + i * 0.25, "fit") for i in range(12)]
    spans = coalesce_layouts(labels, hold_s=0.75, duration=4.0)
    assert spans[0]["layout"] == "split"
    assert spans[0]["end"] == pytest.approx(2.0)
    assert spans[-1]["layout"] == "fit"
    assert spans[-1]["start"] == pytest.approx(2.0)


def test_auto_layout_switches_after_hold(tmp_path):
    dets = []
    for i in range(8):
        t = round(i * 0.125, 3)
        dets.append({"t": t, "x": 480, "y": 540, "w": 180, "h": 220, "kind": "face", "track_id": 1, "mouth": 1})
        dets.append({"t": t, "x": 1440, "y": 540, "w": 180, "h": 220, "kind": "face", "track_id": 2, "mouth": 1})
    for i in range(24):
        t = round(1.0 + i * 0.125, 3)
        dets.append({"t": t, "x": 80, "y": 80, "w": 50, "h": 60, "kind": "face", "track_id": 3})
    plan = _plan(
        tmp_path, mode="auto", layout="auto", duration=4.0, hold_s=0.75,
        detections=dets,
        speakers=[{"start": 0, "end": 1, "speaker": "A"}, {"start": 1, "end": 4, "speaker": "B"}],
    )
    layouts = [span["layout"] for span in plan.metadata["layouts"]]
    assert layouts[0] == "split"
    assert "screenshare" in layouts
    assert plan.metadata["layouts"][0]["end"] >= 1.5
    assert plan.complex_graph
    assert "overlay=enable=" in plan.complex_graph
    assert "crop@ptop" in plan.complex_graph


def test_forced_layouts_build_the_expected_graphs(tmp_path):
    faces = _two_faces(1.0, 0.4)
    screen = [{
        "t": 0.0, "x": 960, "y": 540, "w": 1200, "h": 700,
        "kind": "object", "label": "laptop", "track_id": 9,
    }]
    fit = _plan(tmp_path / "fit", mode="face", layout="fit", duration=1.0, detections=[])
    assert "boxblur=20:2" in (fit.complex_graph or "")
    assert fit.command_files == []

    split = _plan(
        tmp_path / "split", mode="face", layout="split", duration=1.0,
        detections=faces, speakers=_speakers(1.0, 0.4),
    )
    assert "vstack" in (split.complex_graph or "")
    assert "crop@ptop" in (split.complex_graph or "")
    wide = _plan(
        tmp_path / "wide", mode="face", layout="split", duration=1.0, fmt="16:9",
        detections=faces, speakers=_speakers(1.0, 0.4),
    )
    assert "hstack" in (wide.complex_graph or "")

    share = _plan(
        tmp_path / "share", mode="auto", layout="screenshare", duration=1.0,
        detections=faces[:2] + screen,
    )
    assert "crop@face" in (share.complex_graph or "")
    assert "crop@screen" in (share.complex_graph or "")
    assert "[screen][face]vstack" in (share.complex_graph or "")


def test_object_mode_follows_the_object_when_there_is_no_face(tmp_path):
    plan = _plan(
        tmp_path, mode="object", layout="single", duration=1.0,
        detections=[{"t": 0.0, "x": 200, "y": 540, "w": 80, "h": 80, "kind": "object", "label": "person"}],
    )
    assert plan.legacy is False
    assert plan.vf is not None
    assert plan.vf[1].startswith("crop=")
    # 9:16 crop of 1920x1080 is ~606 px wide, so a subject at x=200 sits on the left edge.
    x0 = int(plan.vf[1].split(":")[2])
    assert x0 < 50


def test_detector_failure_falls_back_to_center(tmp_path, monkeypatch):
    monkeypatch.setattr("app.tools.reframe._collect_scored", lambda **kwargs: ([], "unavailable", "detector_unavailable", {}))
    plan = plan_reframe(
        input_path=Path("unused.mp4"), start=0, end=2, output_format="9:16",
        reframe={"mode": "face", "layout": "single"},
        src_w=1920, src_h=1080, output_fps=30, work_dir=tmp_path,
    )
    assert plan.legacy is True
    assert plan.metadata["fallback"] == "detector_unavailable"


def test_word_speaker_is_kept_only_when_present():
    tool = WhisperXTool(_settings(Path(".")))
    out = tool._normalize_result({
        "language": "es",
        "segments": [{
            "start": 0.0, "end": 1.0, "text": "hola", "speaker": "S0",
            "words": [
                {"word": "hola", "start": 0.0, "end": 0.4, "speaker": "S0"},
                {"word": "adios", "start": 0.4, "end": 1.0},
            ],
        }],
    })
    assert out["segments"][0]["speaker"] == "S0"
    assert out["words"][0]["speaker"] == "S0"
    assert "speaker" not in out["words"][1]


def test_empty_hf_token_is_none():
    assert Settings(api_base_url="http://localhost", api_token="t", worker_id="w", hf_token="  ").hf_token is None


def test_render_job_center_keeps_historical_filters(tmp_path, monkeypatch):
    src = _silent(tmp_path, "src.mp4", w=640, h=360, duration=0.4)
    captured: list[dict] = []

    def fake_render(self, **kwargs):
        captured.append(kwargs)
        Path(kwargs["output_path"]).write_bytes(b"x")
        return Path(kwargs["output_path"])

    monkeypatch.setattr(FFmpegTool, "render_clip", fake_render)
    monkeypatch.setattr(
        RenderJob, "_probe",
        lambda self, path: {"width": 1080, "height": 1920, "fps": 30, "has_audio": False, "duration": 0.4},
    )
    monkeypatch.setattr(RenderJob, "_frame_checks", lambda self, *args, **kwargs: {"samples": []})
    base = {"input_video": str(src), "format": "9:16", "start_time": 0, "end_time": 0.4}
    plain = RenderJob(settings=_settings(tmp_path), job=_job("plain", dict(base))).execute()
    center = RenderJob(settings=_settings(tmp_path), job=_job("center", {**base, "reframe": {"mode": "center"}})).execute()
    assert captured[0]["video_filters"] is None and captured[0]["video_complex"] is None
    assert captured[1]["video_filters"] is None and captured[1]["video_complex"] is None
    assert plain["applied"]["reframe"]["legacy"] is True
    assert center["applied"]["reframe"]["legacy"] is True
    assert captured[0]["filter_cwd"] is None

    def fake_plan(**kwargs):
        seen["speakers"] = kwargs["reframe"].get("speakers")
        from app.tools.reframe import ReframePlan
        return ReframePlan(True, None, None, {"mode": "face", "legacy": True, "timings": {}}, [])

    seen: dict = {}
    monkeypatch.setattr("app.jobs.render.plan_reframe", fake_plan)
    captioned = {
        **base,
        "reframe": {"mode": "face"},
        "captions": {"enabled": True, "segments": [
            {"start": 0.0, "end": 0.4, "text": "hola", "speaker": "S0"},
        ]},
    }
    RenderJob(settings=_settings(tmp_path), job=_job("cap", captioned)).execute()
    assert seen["speakers"] == [{"start": 0.0, "end": 0.4, "speaker": "S0"}]


def test_tracking_render_follows_the_active_color_and_layouts_encode(tmp_path):
    src = _interview_source(tmp_path, duration=3.0)
    duration, switch = 3.0, 0.4
    # Same geometry as the painted blocks: left face x=160, right face x=480.
    dets = []
    t = 0.0
    while t < duration - 1e-6:
        left = t < switch
        dets.append({
            "t": round(t, 3), "x": 160, "y": 180, "w": 120, "h": 160,
            "kind": "face", "track_id": 1, "mouth": 1.0 if left else 0.0,
        })
        dets.append({
            "t": round(t, 3), "x": 480, "y": 180, "w": 120, "h": 160,
            "kind": "face", "track_id": 2, "mouth": 0.0 if left else 1.0,
        })
        t += 0.125
    tracked = RenderJob(settings=_settings(tmp_path), job=_job("track", {
        "input_video": str(src), "format": "9:16", "start_time": 0, "end_time": duration,
        "reframe": {
            "mode": "face", "layout": "single",
            "detections": dets, "speakers": _speakers(duration, switch),
        },
    })).execute()
    assert tracked["probe"]["width"] == 1080 and tracked["probe"]["height"] == 1920
    meta = tracked["applied"]["reframe"]
    assert meta["legacy"] is False
    assert meta["speaker_map"] == {"A": 1, "B": 2}
    assert meta["max_shift_observed_px"] <= meta["max_shift_px"] + 1e-6
    assert meta["timings"]["render_s"] >= 0
    early = _mean_rgb(Path(tracked["file_path"]), 0.15)
    late = _mean_rgb(Path(tracked["file_path"]), 2.6)
    assert early[0] > early[1] + 40  # red
    assert late[1] > late[0] + 40  # green

    center = RenderJob(settings=_settings(tmp_path), job=_job("before", {
        "input_video": str(src), "format": "9:16", "start_time": 0, "end_time": 0.5,
    })).execute()
    explicit = RenderJob(settings=_settings(tmp_path), job=_job("center", {
        "input_video": str(src), "format": "9:16", "start_time": 0, "end_time": 0.5,
        "reframe": {"mode": "center", "layout": "single"},
    })).execute()
    a = _mean_rgb(Path(center["file_path"]), 0.2)
    b = _mean_rgb(Path(explicit["file_path"]), 0.2)
    assert all(abs(x - y) < 5 for x, y in zip(a, b))

    square = RenderJob(settings=_settings(tmp_path), job=_job("sq", {
        "input_video": str(src), "format": "1:1", "start_time": 0, "end_time": 0.4,
        "reframe": {"mode": "center"},
    })).execute()
    assert square["probe"]["width"] == 1080 and square["probe"]["height"] == 1080

    tool = FFmpegTool(_settings(tmp_path))
    for layout, extra in (
        ("split", dets),
        ("fit", []),
        ("screenshare", dets[:2] + [{
            "t": 0.0, "x": 320, "y": 180, "w": 400, "h": 220,
            "kind": "object", "label": "laptop", "track_id": 9,
        }]),
    ):
        plan = plan_reframe(
            input_path=src, start=0, end=0.4, output_format="9:16",
            reframe={"mode": "face", "layout": layout, "detections": extra, "speakers": _speakers(0.4, 0.2)},
            src_w=640, src_h=360, output_fps=30, work_dir=tmp_path / layout,
        )
        dest = tmp_path / f"{layout}.mp4"
        tool.render_clip(
            input_path=src, output_path=dest, start=0, end=0.4, output_format="9:16",
            video_filters=plan.vf, video_complex=plan.complex_graph,
            filter_cwd=str(tmp_path / layout) if plan.command_files else None,
        )
        probe = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=width,height", "-of", "csv=p=0", str(dest)],
            check=True, capture_output=True, text=True,
        )
        assert probe.stdout.strip() == "1080,1920"


def test_map_speakers_falls_back_to_left_to_right_without_mouth():
    samples = []
    for t in (0.0, 0.5):
        samples.append(Sample(t, faces=[
            Det(t=t, x=100, y=50, w=40, h=40, kind="face", track_id=2, mouth=0),
            Det(t=t, x=400, y=50, w=40, h=40, kind="face", track_id=7, mouth=0),
        ]))
    speakers = [{"start": 0, "end": 0.5, "speaker": "A"}, {"start": 0.5, "end": 1, "speaker": "B"}]
    assert map_speakers(samples, speakers) == {"A": 2, "B": 7}
