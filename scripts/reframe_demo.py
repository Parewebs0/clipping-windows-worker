"""Regenerate the MOLI-50 before/after pair from one synthetic interview.

Same asset, same window. ``before.mp4`` is the historical center crop.
``after.mp4`` follows the active speaker. Faces are injected so the demo
does not need MediaPipe or YOLO; the render path is the real RenderJob.

    python scripts/reframe_demo.py
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INNER = ROOT / "clipping-windows-worker"
sys.path.insert(0, str(INNER))

from app.config import Settings  # noqa: E402
from app.jobs.render import RenderJob  # noqa: E402


DURATION = 12.0
SWITCH = 3.0
WIDTH, HEIGHT = 960, 540


def _run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True, capture_output=True)


def _source(path: Path) -> None:
    _run([
        "ffmpeg", "-y",
        "-f", "lavfi", "-i", f"color=c=0x808080:size={WIDTH}x{HEIGHT}:rate=30:duration={DURATION}",
        "-f", "lavfi", "-i", f"color=c=red:size=280x{HEIGHT}:rate=30:duration={DURATION}",
        "-f", "lavfi", "-i", f"color=c=0x20c020:size=280x{HEIGHT}:rate=30:duration={DURATION}",
        "-filter_complex", "[0:v][1:v]overlay=40:0[a];[a][2:v]overlay=640:0",
        "-pix_fmt", "yuv420p", str(path),
    ])


def _detections() -> list[dict]:
    dets = []
    t = 0.0
    while t < DURATION - 1e-6:
        left = t < SWITCH
        dets.append({
            "t": round(t, 3), "x": 180, "y": 270, "w": 160, "h": 220,
            "kind": "face", "track_id": 1, "mouth": 1.0 if left else 0.0,
        })
        dets.append({
            "t": round(t, 3), "x": 780, "y": 270, "w": 160, "h": 220,
            "kind": "face", "track_id": 2, "mouth": 0.0 if left else 1.0,
        })
        t += 0.125
    return dets


def _render(settings: Settings, src: Path, job_id: str, reframe: dict | None) -> dict:
    payload = {
        "input_video": str(src),
        "format": "9:16",
        "start_time": 0.0,
        "end_time": DURATION,
    }
    if reframe is not None:
        payload["reframe"] = reframe
    job = type("Job", (), {"id": job_id, "payload": payload, "job_type": "render"})()
    return RenderJob(settings=settings, job=job).execute()


def _mean_rgb(video: Path, t: float) -> list[float]:
    proc = subprocess.run(
        [
            "ffmpeg", "-v", "error", "-ss", f"{t:.3f}", "-i", str(video),
            "-frames:v", "1", "-vf", "scale=24:24:flags=area",
            "-f", "rawvideo", "-pix_fmt", "rgb24", "-",
        ],
        check=True, capture_output=True,
    )
    data = proc.stdout
    n = max(1, len(data) // 3)
    return [round(sum(data[i::3]) / n, 1) for i in range(3)]


def _optional(name: str) -> bool:
    try:
        __import__(name)
    except ImportError:
        return False
    return True


def main() -> None:
    out_dir = ROOT / "docs" / "reframe"
    out_dir.mkdir(parents=True, exist_ok=True)
    gpu = {"available": False, "name": None, "torch": None}
    try:
        import torch
        gpu["torch"] = torch.__version__
        gpu["available"] = bool(torch.cuda.is_available())
        if gpu["available"]:
            gpu["name"] = torch.cuda.get_device_name(0)
    except ImportError:
        pass

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        src = tmp_path / "interview.mp4"
        _source(src)
        settings = Settings(
            api_base_url="http://localhost",
            api_token="demo",
            worker_id="reframe-demo",
            working_directory=str(tmp_path / "work"),
        )
        before = _render(settings, src, "before", {"mode": "center", "layout": "single"})
        after = _render(settings, src, "after", {
            "mode": "face",
            "layout": "single",
            "detections": _detections(),
            "speakers": [
                {"start": 0.0, "end": SWITCH, "speaker": "A"},
                {"start": SWITCH, "end": DURATION, "speaker": "B"},
            ],
        })
        before_path = out_dir / "before.mp4"
        after_path = out_dir / "after.mp4"
        before_path.write_bytes(Path(before["file_path"]).read_bytes())
        after_path.write_bytes(Path(after["file_path"]).read_bytes())

    report = {
        "source": {"width": WIDTH, "height": HEIGHT, "duration_s": DURATION, "window": [0, DURATION]},
        "gpu": gpu,
        "mediapipe": _optional("mediapipe"),
        "ultralytics": _optional("ultralytics"),
        "before": {
            "reframe": before["applied"]["reframe"],
            "rgb_t1": _mean_rgb(before_path, 1.0),
            "rgb_t5": _mean_rgb(before_path, 5.0),
            "bytes": before_path.stat().st_size,
        },
        "after": {
            "reframe": after["applied"]["reframe"],
            "rgb_t1": _mean_rgb(after_path, 1.0),
            "rgb_t5": _mean_rgb(after_path, 5.0),
            "bytes": after_path.stat().st_size,
        },
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
