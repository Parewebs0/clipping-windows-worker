"""T1b: the campaign subject wins over the speaker; without it in shot, the speaker."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

from app.tools.reframe import Det, Sample, label_subject_tracks, normalize_reframe, plan_reframe


def _plan(tmp_path: Path, duration: float = 6.0, **reframe):
    return plan_reframe(
        input_path=Path("unused.mp4"), start=0.0, end=duration, output_format="9:16",
        reframe=reframe, src_w=1920, src_h=1080, output_fps=30.0, work_dir=tmp_path,
    )


def _track(tid, *, x, h=260, duration=6.0, ident=None, mouth=0.0, start=0.0, step=0.125):
    out, t = [], start
    while t < duration - 1e-6:
        d = {"t": round(t, 3), "x": x, "y": 540, "w": h * 0.8, "h": h, "kind": "face", "track_id": tid, "mouth": mouth}
        if ident is not None:
            d["ident"] = ident(t) if callable(ident) else ident
        out.append(d)
        t += step
    return out


SUBJECT = {"name": "Aishah", "reference_paths": ["/refs/a.jpg"]}


def _xs(plan):
    return [int(line.split(" crop x ")[1].split(",")[0]) for line in plan.command_files[0].read_text(encoding="ascii").splitlines()]


def test_subject_config_needs_reference_paths():
    assert "subject" not in normalize_reframe({"mode": "face", "subject": {"name": "x"}})
    cfg = normalize_reframe({"mode": "face", "subject": {"name": "Aishah", "reference_paths": ["a.jpg"], "threshold": 2}})
    assert cfg["subject"] == {"name": "Aishah", "reference_paths": ["a.jpg"], "threshold": 0.9}


def test_track_is_the_subject_when_enough_measured_faces_match():
    a = [Det(t=i, x=0, y=0, w=1, h=1, track_id=1, ident=0.5 if i % 2 else 0.1) for i in range(10)]
    b = [Det(t=i, x=0, y=0, w=1, h=1, track_id=2, ident=0.38 if i == 0 else 0.05) for i in range(10)]
    c = [Det(t=i, x=0, y=0, w=1, h=1, track_id=3) for i in range(10)]  # never measured
    samples = [Sample(t=float(i), faces=[a[i], b[i], c[i]]) for i in range(10)]
    assert label_subject_tracks(samples) == [1]
    assert all(f.subject == (f.track_id == 1) for s in samples for f in s.faces)


def test_subject_beats_the_speaker_and_the_bigger_face(tmp_path):
    # The speaker (track 2) is bigger and talks the whole time; the subject is smaller on the left.
    dets = _track(1, x=400, h=200, ident=0.55) + _track(2, x=1500, h=360, ident=0.05, mouth=1.0)
    speakers = [{"start": 0.0, "end": 6.0, "speaker": "A"}]
    plan = _plan(tmp_path, mode="face", layout="single", detections=dets, speakers=speakers, subject=SUBJECT)
    meta = plan.metadata
    assert meta["subject"]["tracks"] == [1]
    assert meta["subject"]["present"] == 1.0
    assert meta["subject"]["on_screen"] >= 0.95
    assert all(x < 400 for x in _xs(plan))  # the window holds the subject on the left


def test_without_the_subject_in_shot_the_speaker_is_followed(tmp_path):
    # Subject only in the first 2 s; then just a speaker on the right.
    dets = _track(1, x=400, ident=0.6, duration=2.0) + _track(2, x=1500, ident=0.02, mouth=1.0, start=2.0)
    speakers = [{"start": 2.0, "end": 6.0, "speaker": "B"}]
    plan = _plan(tmp_path, mode="face", layout="single", detections=dets, speakers=speakers, subject=SUBJECT)
    xs = _xs(plan)
    assert xs[0] < 400 and xs[-1] > 1100
    sub = plan.metadata["subject"]
    assert sub["present"] < 0.4 and sub["on_screen"] <= sub["present"]


def test_no_subject_config_keeps_t1(tmp_path):
    dets = _track(1, x=400, h=200, ident=0.55) + _track(2, x=1500, h=360)
    plan = _plan(tmp_path, mode="face", layout="single", detections=dets)
    assert "subject" not in plan.metadata
    assert plan.metadata["main_track"] == 2


def test_render_downloads_references_and_skips_failures(tmp_path):
    from app.jobs.render import RenderJob

    job = RenderJob.__new__(RenderJob)
    job.logger = MagicMock()
    local = tmp_path / "ref.jpg"
    local.write_bytes(b"x")
    calls = []

    def fake_download(url, hint):
        calls.append((url, hint))
        if "bad" in url:
            raise RuntimeError("404")
        return tmp_path / f"{hint[0]}.jpg"

    job._download_brand = fake_download
    out = job._subject_references([str(local), "/worker/campaigns/63/subject/a.jpg", "/worker/campaigns/63/subject/bad.jpg", ""])
    assert out == [str(local), str(tmp_path / "subject_1.jpg")]
    assert [c[1][0] for c in calls] == ["subject_1", "subject_2"]


def test_lookalike_above_the_absolute_threshold_is_not_the_subject(tmp_path):
    # Real 1f6c2e37 numbers: the subject scores ~0.8, a lookalike next to him 0.36-0.43.
    dets = _track(1, x=1400, h=220, ident=0.8) + _track(2, x=400, h=300, ident=0.41, mouth=1.0)
    speakers = [{"start": 0.0, "end": 6.0, "speaker": "A"}]
    plan = _plan(tmp_path, mode="face", layout="single", detections=dets, speakers=speakers, subject=SUBJECT)
    assert plan.metadata["subject"]["tracks"] == [1]
    assert all(x > 900 for x in _xs(plan))


def test_two_subject_tracks_in_shot_prefer_the_better_match():
    a = Det(t=0, x=0, y=0, w=1, h=1, track_id=1, ident=0.5)
    b = Det(t=0, x=0, y=0, w=2, h=2, track_id=2, ident=0.8)
    samples = [Sample(t=0.0, faces=[a, b])]
    assert label_subject_tracks(samples) == [1, 2]
    assert b.subject_score > a.subject_score
