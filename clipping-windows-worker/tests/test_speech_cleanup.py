"""MOLI-51: fillers, long pauses, and bad takes. Off leaves the clip alone."""

from __future__ import annotations

import math
import struct
import subprocess
import wave
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from app.config import Settings
from app.jobs.render import RenderJob
from app.models.job import Job
from app.tools.ffmpeg import cleanup_concat_graph
from app.tools.speech_cleanup import (
    DEFAULT_CLEANUP,
    FADE_S,
    PAUSE_MARGIN_S,
    cleanup_options,
    keep_ranges,
    plan_cleanup,
    remap_segments,
    remap_time,
)


def _words(pairs: list[tuple[float, float, str]], *, end: float | None = None) -> list[dict]:
    words = [{"start": start, "end": stop, "word": text} for start, stop, text in pairs]
    return [{"start": words[0]["start"], "end": end or words[-1]["end"], "text": " ".join(w["word"] for w in words), "words": words}]


def test_cleanup_options_default_off():
    assert cleanup_options({}) == DEFAULT_CLEANUP
    assert cleanup_options(None)["fillers"] is False
    assert cleanup_options(None)["max_pause_ms"] is None
    assert cleanup_options({"cleanup": {"fillers": True, "max_pause_ms": "800", "bad_takes": 1,
                                        "filler_list": ["eh"]}}) == {
        "fillers": True, "max_pause_ms": 800, "bad_takes": True, "filler_list": ["eh"],
    }
    assert cleanup_options({"cleanup": {"max_pause_ms": 0, "fillers": "nope"}})["fillers"] is False
    assert cleanup_options({"cleanup": {"max_pause_ms": 0}})["max_pause_ms"] is None


def test_disabled_plan_does_not_cut():
    segments = _words([(0.0, 0.4, "hola"), (0.5, 0.7, "eh"), (1.2, 1.6, "mundo")])
    plan = plan_cleanup(segments, duration=2.0, options=DEFAULT_CLEANUP, language="es")
    assert plan["enabled"] is False
    assert plan["applied"] is False
    assert plan["removed_seconds"] == 0.0
    assert plan["cuts"] == []
    assert keep_ranges(plan) is None


def test_fillers_and_long_pauses_keep_a_margin_and_drop_the_filler():
    segments = _words([
        (0.0, 0.40, "hola"),
        (0.50, 0.70, "eh"),
        (0.80, 1.00, "o"),
        (1.00, 1.20, "sea"),
        (3.00, 3.40, "mundo"),
    ])
    plan = plan_cleanup(segments, duration=4.0, options={
        "fillers": True, "max_pause_ms": 500, "bad_takes": False, "filler_list": ["eh", "o sea"],
    }, language="es")
    assert plan["applied"] is True
    kinds = {cut["kind"] for cut in plan["cuts"]}
    assert kinds == {"filler", "pause"}
    filler_text = " ".join(cut.get("text", "") for cut in plan["cuts"] if cut["kind"] == "filler")
    assert "eh" in filler_text and "o sea" in filler_text
    # 80 ms pad against "hola" (ends 0.40) and against "mundo" (starts 3.00).
    # Filler slices split the pause, so the pad is the outer edges.
    pauses = [cut for cut in plan["cuts"] if cut["kind"] == "pause"]
    # Pad sits on both sides of the kept words. The tail after "mundo" is its own pause.
    assert any(abs(cut["start"] - (0.40 + PAUSE_MARGIN_S)) < 0.002 for cut in pauses)
    assert any(abs(cut["end"] - (3.00 - PAUSE_MARGIN_S)) < 0.002 for cut in pauses)
    assert plan["removed_seconds"] == pytest.approx(plan["by_kind"]["fillers"] + plan["by_kind"]["pauses"], abs=0.002)
    assert plan["removed_seconds"] > 1.0
    keeps = keep_ranges(plan)
    assert keeps is not None
    remapped = remap_segments(segments, keeps)
    words = [word["word"] for seg in remapped for word in seg["words"]]
    assert words == ["hola", "mundo"]
    # "mundo" source 3.00 sits 80 ms into the second keep.
    assert remapped[0]["words"][1]["start"] == pytest.approx(
        remap_time(3.00, keeps), abs=0.002,
    )
    assert remapped[0]["words"][1]["start"] < 3.00


def test_bad_take_keeps_the_last_repetition():
    def phrase(start: float) -> list[tuple[float, float, str]]:
        tokens = ["vamos", "a", "hablar", "de", "casas."]
        return [(start + i * 0.3, start + i * 0.3 + 0.25, token) for i, token in enumerate(tokens)]

    first = phrase(0.2)
    second = phrase(2.4)
    segments = [
        {"start": 0.2, "end": 1.7, "text": "vamos a hablar de casas.", "words": [
            {"start": a, "end": b, "word": w} for a, b, w in first]},
        {"start": 2.4, "end": 3.9, "text": "vamos a hablar de casas.", "words": [
            {"start": a, "end": b, "word": w} for a, b, w in second]},
    ]
    plan = plan_cleanup(segments, duration=4.2, options={
        "fillers": False, "max_pause_ms": None, "bad_takes": True,
    }, language="es")
    assert plan["applied"] is True
    assert plan["by_kind"]["bad_takes"] > 1.0
    assert plan["by_kind"]["fillers"] == 0.0
    takes = [cut for cut in plan["cuts"] if cut["kind"] == "bad_take"]
    assert len(takes) == 1
    assert takes[0]["start"] == pytest.approx(0.0, abs=0.002)
    assert takes[0]["end"] == pytest.approx(2.4 - PAUSE_MARGIN_S, abs=0.002)
    keeps = keep_ranges(plan)
    assert keeps is not None
    remapped = remap_segments(segments, keeps)
    text = " ".join(seg["text"] for seg in remapped)
    assert text.count("vamos") == 1
    assert remapped[0]["words"][0]["start"] == pytest.approx(PAUSE_MARGIN_S, abs=0.02)


def test_near_paraphrase_is_not_a_bad_take():
    def seg(start: float, tokens: list[str]) -> dict:
        words = [{"start": start + i * 0.3, "end": start + i * 0.3 + 0.25, "word": token} for i, token in enumerate(tokens)]
        return {"start": start, "end": words[-1]["end"], "text": " ".join(tokens), "words": words}

    segments = [
        seg(0.0, ["esto", "es", "una", "casa", "pequeña."]),
        seg(2.0, ["el", "precio", "baja", "este", "mes."]),
    ]
    plan = plan_cleanup(segments, duration=4.0, options={"bad_takes": True}, language="es")
    assert plan["applied"] is False
    assert plan["reason"] == "nothing_to_cut"


def test_este_matches_the_token_not_a_longer_word():
    segments = _words([(0.0, 0.3, "estelar"), (0.4, 0.6, "este"), (0.7, 1.1, "producto")])
    plan = plan_cleanup(segments, duration=1.5, options={
        "fillers": True, "filler_list": ["este"], "bad_takes": False, "max_pause_ms": None,
    }, language="es")
    assert plan["applied"] is True
    assert [cut.get("text") for cut in plan["cuts"]] == ["este"]
    keeps = keep_ranges(plan)
    words = [word["word"] for seg in remap_segments(segments, keeps) for word in seg["words"]]
    assert words == ["estelar", "producto"]


def test_cleanup_filter_uses_trim_concat_and_a_short_fade():
    graph = cleanup_concat_graph([(0.0, 0.88), (3.12, 4.0)], fade_s=FADE_S, video=True)
    assert "trim=start=0.000:end=0.880" in graph
    assert "atrim=start=3.120:end=4.000" in graph
    assert "concat=n=2:v=1:a=1" in graph
    assert "afade=t=in:st=0:d=0.015" in graph
    assert "afade=t=out" in graph
    single = cleanup_concat_graph([(0.0, 1.25)], fade_s=FADE_S, video=True)
    assert "concat=" not in single
    assert single.endswith("[acat]") or "[acat]" in single
    assert "afade=t=in:st=0:d=0.015" in single
    assert FADE_S == pytest.approx(0.015)
    assert 0.010 <= FADE_S <= 0.020
    assert 0.060 <= PAUSE_MARGIN_S <= 0.100


def _write_wav(path: Path, samples: list[float], rate: int = 48000) -> None:
    with wave.open(str(path), "w") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        frames = b"".join(
            struct.pack("<h", max(-32767, min(32767, int(sample * 32767))))
            for sample in samples
        )
        handle.writeframes(frames)


def _read_wav(path: Path) -> list[float]:
    with wave.open(str(path), "r") as handle:
        channels = handle.getnchannels()
        count = handle.getnframes()
        raw = handle.readframes(count)
    interleaved = [sample / 32767.0 for sample, in struct.iter_unpack("<h", raw)]
    return interleaved[::channels] if channels > 1 else interleaved


def _max_step(samples: list[float]) -> float:
    return max(abs(b - a) for a, b in zip(samples, samples[1:]))


def test_audio_fades_remove_the_click_at_the_join(tmp_path: Path):
    rate = 48000
    seconds = 4.0
    samples = [math.sin(2 * math.pi * 440 * n / rate) for n in range(int(rate * seconds))]
    source = tmp_path / "tone.wav"
    _write_wav(source, samples, rate)
    keeps = [(0.0, 0.88), (3.12, 4.0)]
    faded = tmp_path / "faded.wav"
    hard = tmp_path / "hard.wav"
    fade_graph = cleanup_concat_graph(keeps, fade_s=FADE_S, video=False)
    hard_graph = cleanup_concat_graph(keeps, fade_s=0.0, video=False)
    for graph, dest in ((fade_graph, faded), (hard_graph, hard)):
        subprocess.run(
            ["ffmpeg", "-y", "-i", str(source), "-filter_complex", graph, "-map", "[acat]", str(dest)],
            check=True, capture_output=True,
        )
    fade_step = _max_step(_read_wav(faded))
    hard_step = _max_step(_read_wav(hard))
    # A raw join of this 440 Hz tone jumps by ~1.9. The 15 ms fade stays near zero.
    assert hard_step > 1.0
    assert fade_step < 0.20
    assert fade_step < hard_step * 0.25


def _job(job_id: str, payload: dict) -> Job:
    job = MagicMock(spec=Job)
    job.id = job_id
    job.payload = payload
    job.job_type = "render"
    return job


def test_render_cleanup_shortens_the_clip_and_keeps_subtitles_in_sync(tmp_path: Path):
    source = tmp_path / "src.mp4"
    subprocess.run(
        ["ffmpeg", "-y", "-f", "lavfi", "-i", "color=c=black:size=640x360:rate=30:duration=6",
         "-f", "lavfi", "-i", "sine=frequency=440:duration=6", "-shortest", "-pix_fmt", "yuv420p", str(source)],
        check=True, capture_output=True,
    )
    segments = [{
        "start": 0.0, "end": 5.5, "text": "hola eh mundo",
        "words": [
            {"start": 0.2, "end": 0.6, "word": "hola"},
            {"start": 0.8, "end": 1.0, "word": "eh"},
            {"start": 3.2, "end": 3.6, "word": "mundo"},
        ],
    }]
    settings = Settings(
        api_base_url="http://localhost", api_token="test", worker_id="t",
        working_directory=str(tmp_path), output_dir=str(tmp_path / "out"),
    )
    base = {
        "input_video": str(source), "format": "9:16", "start_time": 0.0, "end_time": 5.0,
        "language": "es",
        "captions": {"enabled": True, "segments": segments},
    }
    off = RenderJob(settings=settings, job=_job("clean-off", dict(base))).execute()
    on_payload = dict(base)
    on_payload["cleanup"] = {"fillers": True, "max_pause_ms": 400, "bad_takes": False, "filler_list": ["eh"]}
    on = RenderJob(settings=settings, job=_job("clean-on", on_payload)).execute()

    assert off["applied"]["cleanup"]["enabled"] is False
    assert off["applied"]["cleanup"]["removed_seconds"] == 0.0
    assert off["duration_seconds"] == pytest.approx(5.0, abs=0.15)
    assert off["probe"]["duration"] == pytest.approx(5.0, abs=0.15)

    removed = on["applied"]["cleanup"]["removed_seconds"]
    assert on["applied"]["cleanup"]["applied"] is True
    assert removed > 1.0
    assert on["probe"]["duration"] == pytest.approx(5.0 - removed, abs=0.2)
    assert on["duration_seconds"] == pytest.approx(on["probe"]["duration"], abs=0.05)
    assert "eh" not in on["applied"]["captions"]["text"]
    assert "mundo" in on["applied"]["captions"]["text"]

    ass = (tmp_path / "jobs" / "clean-on" / "output" / "overlay.ass").read_text(encoding="utf-8")
    mundo_line = next(line for line in ass.splitlines() if "mundo" in line)
    # Dialogue: 0,0:00:00.76,0:00:01.16,Caption,...
    start_ts = mundo_line.split(",")[1]
    assert start_ts.startswith("0:00:00.")
    centis = int(start_ts.split(".")[-1])
    assert centis < 200  # well before the original 3.2 s, on the shortened timeline
