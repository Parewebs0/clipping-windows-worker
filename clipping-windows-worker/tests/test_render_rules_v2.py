"""Issue #4: render_spec v2 — captions from transcript, brand dictionary, logo, on-screen text."""

from __future__ import annotations

import http.server
import subprocess
import threading
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from app.config import Settings
from app.jobs.render import RenderJob
from app.models.job import Job
from app.tools.captions import apply_brand_dictionary, build_ass, caption_events


def _settings(tmp_path: Path) -> Settings:
    return Settings(api_base_url="http://localhost", api_token="test", worker_id="t",
                    working_directory=str(tmp_path), output_dir=str(tmp_path / "out"))


def _job(job_id: str, payload: dict) -> Job:
    j = MagicMock(spec=Job)
    j.id = job_id
    j.payload = payload
    j.job_type = "render"
    return j


def _src(tmp_path: Path, audio: bool = True, duration: int = 6) -> Path:
    out = tmp_path / ("src_a.mp4" if audio else "src_na.mp4")
    cmd = ["ffmpeg", "-y", "-f", "lavfi", "-i", f"color=c=black:size=640x360:rate=30:duration={duration}"]
    if audio:
        cmd += ["-f", "lavfi", "-i", f"sine=frequency=440:duration={duration}", "-shortest"]
    cmd += ["-pix_fmt", "yuv420p", str(out)]
    subprocess.run(cmd, check=True, capture_output=True)
    return out


def _logo(tmp_path: Path) -> Path:
    p = tmp_path / "srv" / "logo.png"
    p.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["ffmpeg", "-y", "-f", "lavfi", "-i", "color=c=red:size=200x100", "-frames:v", "1", str(p)],
                   check=True, capture_output=True)
    return p


def _frame_mean(video: Path, t: float, crop: str) -> float:
    """Mean luma/colour of a crop at time t (signalstats YAVG)."""
    out = subprocess.run(
        ["ffmpeg", "-v", "info", "-ss", f"{t}", "-i", str(video), "-frames:v", "1",
         "-vf", f"crop={crop},signalstats,metadata=print:key=lavfi.signalstats.YAVG", "-f", "null", "-"],
        capture_output=True, text=True, check=True)
    for line in (out.stderr + out.stdout).splitlines():
        if "YAVG=" in line:
            return float(line.split("YAVG=")[1])
    raise AssertionError(out.stderr)


def test_brand_dictionary():
    d = ["BOXABL", "Casita", "foldable home"]
    assert apply_brand_dictionary("this boxabl casita, a Foldable Home!", d) == "this BOXABL Casita, a foldable home!"
    assert apply_brand_dictionary("box abl", ["BOXABL"]) == "BOXABL"
    assert apply_brand_dictionary("hello", []) == "hello"


def test_caption_events_word_timing_and_split():
    segs = [{"start": 0.0, "end": 2.0, "text": "a b c d e f", "words": [
        {"start": 0.1 * i, "end": 0.1 * i + 0.1, "word": w} for i, w in enumerate("a b c d e boxabl".split())]}]
    evs = caption_events(segs, max_words=4, dictionary=["BOXABL"])
    assert [e[2] for e in evs] == ["a b c d", "e BOXABL"]
    assert evs[0][0] == pytest.approx(0.0) and evs[1][1] == pytest.approx(0.6)
    # without words: evenly spread, clipped to duration
    evs = caption_events([{"start": 0, "end": 4, "text": "one two three four five six seven eight"}], duration=3)
    assert len(evs) == 2 and evs[-1][1] <= 3


def test_build_ass_contains_styles_and_events():
    content, summary = build_ass(width=1080, height=1920,
                                 captions={"enabled": True, "segments": [{"start": 0, "end": 1, "text": "hi there"}]},
                                 on_screen_text={"enabled": True, "items": [{"text": "Link in bio", "start": 0, "end": 2}]},
                                 duration=5)
    assert "PlayResX: 1080" in content and "Style: Caption" in content and "Style: Overlay" in content
    assert "Link in bio" in content and "hi there" in content
    assert summary["captions"]["events"] == 1 and summary["on_screen_text"]["applied"]


def test_render_full_spec_with_logo_url_and_captions(tmp_path):
    src = _src(tmp_path, audio=True)
    logo = _logo(tmp_path)
    handler = lambda *a, **k: http.server.SimpleHTTPRequestHandler(*a, directory=str(logo.parent), **k)  # noqa: E731
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        url = f"http://127.0.0.1:{srv.server_address[1]}/logo.png"
        job = _job("r-v2", {
            "input_video": str(src), "format": "9:16", "start_time": 1.0, "end_time": 5.0,
            "captions": {"enabled": True, "brand_dictionary": ["BOXABL"],
                         "segments": [{"start": 0.0, "end": 3.5, "text": "the boxabl casita is here now"}]},
            "on_screen_text": {"enabled": True, "items": [{"text": "Tag @boxabl", "start": 0, "end": 2, "position": "top"}]},
            "watermark": {"enabled": True, "url": url, "position": "top_right", "width": 200},
        })
        r = RenderJob(settings=_settings(tmp_path), job=job).execute()
    finally:
        srv.shutdown()
    out = Path(r["file_path"])
    assert out.exists()
    assert r["render_spec_version"] == 2
    a = r["applied"]
    assert a["captions"]["applied"] and a["captions"]["events"] >= 1 and "BOXABL" in a["captions"]["text"]
    assert a["on_screen_text"]["applied"] and a["on_screen_text"]["texts"][0]["text"] == "Tag @boxabl"
    assert a["watermark"]["applied"] and a["watermark"]["position"] == "top_right"
    p = r["probe"]
    assert (p["width"], p["height"]) == (1080, 1920) and p["has_audio"] and p["duration"] == pytest.approx(4.0, abs=0.15)
    # frame sample: the red logo area (top right) is brighter than the black background,
    # and the caption band (lower third) has white text pixels.
    logo_y = _frame_mean(out, 1.0, "200:100:840:40")
    bg_y = _frame_mean(out, 1.0, "200:100:40:900")
    assert logo_y > bg_y + 20
    cap_y = _frame_mean(out, 1.0, "1080:200:0:1380")
    assert cap_y > bg_y + 1
    fc = a["frame_checks"]
    assert fc["logo_visible"] is True and fc["captions_visible"] is True
    assert {s["what"] for s in fc["samples"]} >= {"logo", "captions", "background"}


def test_render_without_audio_and_watermark_does_not_crash(tmp_path):
    src = _src(tmp_path, audio=False, duration=3)
    logo = _logo(tmp_path)
    job = _job("r-na", {"input_video": str(src), "format": "9:16", "start_time": 0, "end_time": 2,
                        "watermark": {"enabled": True, "file": str(logo), "start": 0, "end": 1}})
    r = RenderJob(settings=_settings(tmp_path), job=job).execute()
    assert r["probe"]["has_audio"] is False and r["applied"]["watermark"]["applied"]


def test_logo_request_sends_bearer_only_on_our_worker_route():
    from app.jobs.render import logo_request

    url, headers = logo_request("/worker/campaigns/3/logo", api_base_url="https://api.example", api_token="sek")
    assert url == "https://api.example/worker/campaigns/3/logo"
    assert headers == {"Authorization": "Bearer sek"}
    _, external = logo_request("https://cdn.example/logo.png", api_base_url="https://api.example", api_token="sek")
    assert external == {}
    _, other = logo_request("https://evil.example/worker/campaigns/3/logo", api_base_url="https://api.example", api_token="sek")
    assert other == {}


def test_watermark_enabled_without_source_fails(tmp_path):
    src = _src(tmp_path, audio=False, duration=2)
    job = _job("r-bad", {"input_video": str(src), "format": "9:16", "start_time": 0, "end_time": 1,
                         "watermark": {"enabled": True}})
    with pytest.raises(ValueError):
        RenderJob(settings=_settings(tmp_path), job=job).execute()
