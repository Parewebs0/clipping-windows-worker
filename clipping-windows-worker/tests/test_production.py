"""MOLI-55: stock license, ducking, xfade and a real before/after render."""
from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from app.config import Settings
from app.jobs.render import RenderJob
from app.models.job import Job
from app.tools.production import (
    CC0,
    choose_pexels_url,
    compile_production,
    features_requested,
    pexels_license,
    pixabay_license,
    prepare_production,
    resolve_music,
)
from unittest.mock import MagicMock


def _settings(tmp_path: Path) -> Settings:
    return Settings(
        api_base_url="http://localhost",
        api_token="test",
        worker_id="t",
        working_directory=str(tmp_path),
        output_dir=str(tmp_path / "out"),
    )


def _job(job_id: str, payload: dict) -> Job:
    job = MagicMock(spec=Job)
    job.id = job_id
    job.payload = payload
    job.job_type = "render"
    return job


def _src(tmp_path: Path, *, color: str = "black", duration: int = 5) -> Path:
    out = tmp_path / "src.mp4"
    subprocess.run(
        ["ffmpeg", "-y", "-f", "lavfi", "-i", f"color=c={color}:s=640x360:r=30:d={duration}",
         "-f", "lavfi", "-i", f"sine=frequency=440:sample_rate=48000:d={duration}",
         "-shortest", "-pix_fmt", "yuv420p", str(out)],
        check=True, capture_output=True,
    )
    return out


def _green(tmp_path: Path) -> Path:
    out = tmp_path / "green.mp4"
    subprocess.run(
        ["ffmpeg", "-y", "-f", "lavfi", "-i", "color=c=green:s=320x180:r=30:d=3",
         "-pix_fmt", "yuv420p", str(out)],
        check=True, capture_output=True,
    )
    return out


def _uavg(video: Path, t: float) -> float:
    out = subprocess.run(
        ["ffmpeg", "-v", "info", "-ss", str(t), "-i", str(video), "-frames:v", "1",
         "-vf", "scale=32:32,signalstats,metadata=print:key=lavfi.signalstats.UAVG",
         "-f", "null", "-"],
        capture_output=True, text=True, check=True,
    )
    for line in (out.stderr + out.stdout).splitlines():
        if "UAVG=" in line:
            return float(line.split("UAVG=", 1)[1].split()[0])
    raise AssertionError(out.stderr)


def test_each_block_is_opt_in_and_original_only_blocks_music(tmp_path):
    assert features_requested({}) is False
    assert features_requested({"broll": {"enabled": False}, "music": {"enabled": True}}) is True
    music, applied = resolve_music(
        {"enabled": True, "track": "bed_soft", "ducking": True},
        library={"bed_soft": tmp_path / "missing.wav"},
        original_only=True,
        no_added_music=False,
    )
    assert music is None
    assert applied["reason"] == "audio.original_only"
    assert applied["applied"] is False


def test_stock_license_is_recorded_from_the_provider_payload():
    video = {
        "id": 2499611,
        "url": "https://www.pexels.com/video/2499611/",
        "user": {"name": "Pavel Danilyuk"},
        "video_files": [
            {"file_type": "video/mp4", "height": 2160, "link": "https://cdn.example/4k.mp4"},
            {"file_type": "video/mp4", "height": 720, "link": "https://cdn.example/720.mp4"},
        ],
    }
    assert choose_pexels_url(video) == "https://cdn.example/720.mp4"
    record = pexels_license(video, "city skyline")
    assert record["license"] == "Pexels License"
    assert record["license_url"] == "https://www.pexels.com/license/"
    assert record["author"] == "Pavel Danilyuk"
    assert record["query"] == "city skyline"
    pix = pixabay_license({"id": 9, "pageURL": "https://pixabay.com/videos/id-9/", "user": "ann"}, "desk")
    assert pix["license"] == "Pixabay Content License"
    assert pix["license_url"] == "https://pixabay.com/service/license-summary/"


def test_graph_uses_sidechaincompress_and_xfade_only_when_enabled():
    argv, _cwd = compile_production(
        binary="ffmpeg",
        source="in.mp4",
        output_path="out.mp4",
        start=1.5,
        duration=5,
        output_format="9:16",
        output_fps=30,
        captions_file=None,
        watermark=None,
        broll=[{
            "path": "b.mp4", "start": 1.0, "end": 3.0, "placement": "overlay",
            "opacity": 1.0, "license": {},
        }],
        music={"path": "m.wav", "volume": 0.18, "ducking": True, "license": {}},
        sfx=[{"path": "s.wav", "at_s": 0.2, "gain": 0.35, "license": {}}],
        transitions={"effect": "fade", "duration_s": 0.4},
        audio_filter="loudnorm=I=-14:TP=-1:LRA=11",
        has_speech=True,
    )
    graph = argv[argv.index("-filter_complex") + 1]
    assert "sidechaincompress=threshold=0.05:ratio=8:attack=20:release=250" in graph
    assert "xfade=transition=fade:duration=0.400" in graph
    assert "overlay=" in graph
    assert "adelay=" in graph
    assert argv[argv.index("-ss") + 1] == "1.500"

    quiet, _cwd = compile_production(
        binary="ffmpeg",
        source="in.mp4",
        output_path="out.mp4",
        start=0,
        duration=5,
        output_format="9:16",
        output_fps=30,
        captions_file=None,
        watermark=None,
        broll=[],
        music=None,
        sfx=[],
        transitions={"effect": "wipeleft", "duration_s": 0.3},
        audio_filter=None,
        has_speech=True,
    )
    quiet_graph = quiet[quiet.index("-filter_complex") + 1]
    assert "sidechaincompress" not in quiet_graph
    assert "xfade=transition=wipeleft" in quiet_graph


def test_before_and_after_same_window(tmp_path):
    src = _src(tmp_path)
    green = _green(tmp_path)
    window = {"input_video": str(src), "format": "9:16", "start_time": 0, "end_time": 4}
    before = RenderJob(settings=_settings(tmp_path), job=_job("before", dict(window))).execute()
    after_payload = {
        **window,
        "broll": {
            "enabled": True,
            "provider": "pexels",
            "placement": "overlay",
            "marks": [{
                "start": 1.0,
                "end": 3.0,
                "query": "green",
                "file": str(green),
                "placement": "overlay",
                "license": {**CC0, "provider": "local", "query": "green"},
            }],
        },
        "music": {"enabled": True, "track": "bed_soft", "volume": 0.2, "ducking": True},
        "transitions": {"enabled": True, "effect": "fade", "duration_s": 0.3},
        "sfx": {"enabled": True, "gain": 0.4, "events": [{"name": "whoosh", "at_s": 1.0}]},
    }
    after = RenderJob(settings=_settings(tmp_path), job=_job("after", after_payload)).execute()

    assert "broll" not in before["applied"]
    assert before["applied"]["audio"]["normalized"] is True
    assert after["applied"]["broll"]["applied"] is True
    assert after["applied"]["broll"]["assets"][0]["license"] == "CC0-1.0"
    assert after["applied"]["music"]["applied"] is True
    assert after["applied"]["music"]["ducking"] == "sidechaincompress"
    assert after["applied"]["music"]["asset"]["license"] == "CC0-1.0"
    assert after["applied"]["transitions"]["filter"] == "xfade"
    assert after["applied"]["sfx"]["applied"] is True
    assert after["applied"]["sfx"]["assets"][0]["license"] == "CC0-1.0"
    assert Path(before["file_path"]).stat().st_size > 0
    assert Path(after["file_path"]).stat().st_size > 0
    # The green plate sits on the frame in the after clip and not in the before clip.
    assert abs(_uavg(Path(after["file_path"]), 2.0) - _uavg(Path(before["file_path"]), 2.0)) > 15

    kept = RenderJob(settings=_settings(tmp_path), job=_job("kept", {
        **window,
        "audio": {"original_only": True},
        "music": {"enabled": True, "track": "bed_soft"},
    })).execute()
    assert kept["applied"]["audio"]["reason"] == "audio.original_only"
    assert kept["applied"]["music"]["applied"] is False
    assert kept["applied"]["music"]["reason"] == "audio.original_only"


def test_prepare_records_a_missing_stock_key_without_rendering(tmp_path):
    prepared = prepare_production(
        {"broll": {"enabled": True, "marks": [{"start": 0.5, "end": 2.0, "query": "skyline"}]}},
        duration=4,
        original_only=False,
        work_dir=tmp_path,
        binary="ffmpeg",
        pexels_key=None,
        pixabay_key=None,
    )
    assert prepared["use_graph"] is False
    assert prepared["applied"]["broll"]["reason"] == "missing_api_key"
    assert prepared["applied"]["broll"]["assets"][0]["error"] == "missing_api_key"
