"""MOLI-49: animated caption presets, emoji cues, censor, hook entrance.

`classic` must stay the static burn-in the rest of the suite already renders.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from app.config import Settings
from app.jobs.render import RenderJob
from app.models.job import Job
from app.tools.captions import build_ass, caption_events, resolve_caption_style
from app.tools.emoji_png import codepoint_name, resolve_emoji_png


def _words(pairs):
    return [{"start": a, "end": b, "word": w} for a, b, w in pairs]


def _captions(preset, **style):
    body = {"preset": preset, **style}
    return {
        "enabled": True,
        "segments": [{
            "start": 0.0,
            "end": 2.0,
            "text": "hola mundo cruel",
            "words": _words([(0.10, 0.45, "hola"), (0.50, 0.95, "mundo"), (1.20, 1.70, "cruel")]),
        }],
        "style": body,
    }


def test_classic_preset_matches_the_static_burn_in():
    content, summary = build_ass(
        width=1080, height=1920,
        captions={"enabled": True, "segments": [{"start": 0, "end": 1, "text": "hi there"}]},
        on_screen_text={"enabled": True, "items": [{"text": "Link in bio", "start": 0, "end": 2}]},
        duration=5,
    )
    assert "Style: Caption,Arial,73,&H00FFFFFF,&H00FFFFFF,&H00000000,&H64000000,-1,0,0,0,100,100,0,0,1,6,2,2,60,60,422,1" in content
    assert "Dialogue: 0,0:00:00.00,0:00:01.00,Caption,,0,0,0,,hi there" in content
    assert "Dialogue: 1,0:00:00.00,0:00:02.00,Overlay,,0,0,0,,{\\an8}Link in bio" in content
    assert "\\k" not in content and "\\t(" not in content and "\\fscx" not in content
    assert summary["captions"]["preset"] == "classic"
    assert summary["captions"]["events"] == 1
    assert summary["captions"]["emoji_cues"] == []
    assert resolve_caption_style({})["name"] == "classic"
    assert resolve_caption_style({"preset": "nope"})["name"] == "classic"


def test_three_presets_are_selectable_and_distinct():
    names = []
    bodies = []
    for preset in ("karaoke", "pop", "bold"):
        content, summary = build_ass(
            width=1080, height=1920, captions=_captions(preset), on_screen_text=None, duration=3,
        )
        names.append(summary["captions"]["preset"])
        bodies.append(content)
        assert summary["captions"]["font_file"]
    assert names == ["karaoke", "pop", "bold"]
    assert len(set(bodies)) == 3
    karaoke, pop, bold = bodies
    assert "\\k" in karaoke and "Montserrat" in karaoke
    assert "\\fscx132" in pop and "\\t(0,90," in pop
    assert "Anton" in bold
    assert "HOLA" in bold and "MUNDO" in bold and "CRUEL" in bold and "hola" not in bold
    assert "\\k" in bold


def test_active_word_stays_within_80ms_of_whisperx():
    _, karaoke = build_ass(
        width=1080, height=1920, captions=_captions("karaoke"), on_screen_text=None, duration=3,
    )
    _, pop = build_ass(
        width=1080, height=1920, captions=_captions("pop"), on_screen_text=None, duration=3,
    )
    for summary in (karaoke, pop):
        sync = summary["captions"]["word_sync"]
        assert [row["word"] for row in sync] == ["hola", "mundo", "cruel"]
        for row in sync:
            assert abs(row["ass"] - row["whisperx"]) <= 0.08


def test_censor_masks_the_word_and_reports_a_beep_span():
    content, summary = build_ass(
        width=720, height=1280,
        captions=_captions("classic", censor={"words": ["mundo"], "beep": True}),
        on_screen_text=None,
        duration=3,
    )
    assert "mundo" not in content.lower()
    assert "***" in content
    assert summary["captions"]["censor_beep"] is True
    assert summary["captions"]["censor_spans"] == [{"start": 0.5, "end": 0.95}]
    assert "***" in summary["captions"]["text"]


def test_second_speaker_is_coloured_and_a_single_speaker_is_not():
    solo = {
        "enabled": True,
        "segments": [{
            "start": 0, "end": 1, "speaker": "SPEAKER_00", "text": "hola",
            "words": [{"start": 0.1, "end": 0.6, "word": "hola", "speaker": "SPEAKER_00"}],
        }],
    }
    content, _ = build_ass(width=720, height=1280, captions=solo, on_screen_text=None, duration=2)
    assert "\\c&H" not in content
    duo = {
        "enabled": True,
        "segments": [
            {"start": 0, "end": 0.6, "speaker": "SPEAKER_00", "text": "hola",
             "words": [{"start": 0.1, "end": 0.5, "word": "hola", "speaker": "SPEAKER_00"}]},
            {"start": 0.7, "end": 1.4, "speaker": "SPEAKER_01", "text": "mundo",
             "words": [{"start": 0.7, "end": 1.3, "word": "mundo", "speaker": "SPEAKER_01"}]},
        ],
    }
    coloured, summary = build_ass(width=720, height=1280, captions=duo, on_screen_text=None, duration=2)
    assert "mundo" in coloured and "\\c&H00FFE599&" in coloured
    assert summary["captions"]["preset"] == "classic"


def test_emoji_cue_lands_on_the_keyword_and_hook_entrance_is_opt_in():
    captions = _captions("karaoke", emoji=True)
    captions["keywords"] = ["mundo"]
    captions["emojis"] = [{"emoji": "🔥", "keyword": "mundo"}]
    content, summary = build_ass(
        width=1080, height=1920,
        captions=captions,
        on_screen_text={"enabled": True, "items": [
            {"text": "WAIT", "start": 0, "end": 2, "position": "top", "entrance": True},
        ]},
        duration=3,
    )
    cues = summary["captions"]["emoji_cues"]
    assert cues[0]["emoji"] == "🔥"
    assert abs(cues[0]["start"] - 0.50) <= 0.08
    assert "\\move(" in content and "WAIT" in content
    still = build_ass(
        width=1080, height=1920,
        captions=_captions("classic"),
        on_screen_text={"enabled": True, "items": [{"text": "WAIT", "start": 0, "end": 2}]},
        duration=3,
    )[0]
    assert "\\move(" not in still


def test_bundled_twemoji_png_resolves_offline(tmp_path):
    assert codepoint_name("🔥") == "1f525"
    png = resolve_emoji_png("🔥", tmp_path)
    assert png is not None and png.name == "1f525.png"
    assert png.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    assert resolve_emoji_png("", tmp_path) is None


def test_caption_events_contract_is_unchanged():
    segs = [{"start": 0.0, "end": 2.0, "text": "a b c d e f", "words": [
        {"start": 0.1 * i, "end": 0.1 * i + 0.1, "word": w} for i, w in enumerate("a b c d e boxabl".split())]}]
    evs = caption_events(segs, max_words=4, dictionary=["BOXABL"])
    assert [e[2] for e in evs] == ["a b c d", "e BOXABL"]


def _settings(tmp_path: Path) -> Settings:
    return Settings(api_base_url="http://localhost", api_token="test", worker_id="t",
                    working_directory=str(tmp_path), output_dir=str(tmp_path / "out"))


def _job(job_id: str, payload: dict) -> Job:
    j = MagicMock(spec=Job)
    j.id = job_id
    j.payload = payload
    j.job_type = "render"
    return j


def test_karaoke_render_keeps_caption_frame_checks(tmp_path):
    src = tmp_path / "src.mp4"
    subprocess.run(
        ["ffmpeg", "-y", "-f", "lavfi", "-i", "color=c=black:size=640x360:rate=30:duration=4",
         "-f", "lavfi", "-i", "sine=frequency=440:duration=4", "-shortest", "-pix_fmt", "yuv420p", str(src)],
        check=True, capture_output=True,
    )
    payload = {
        "input_video": str(src), "format": "9:16", "start_time": 0.0, "end_time": 4.0,
        "captions": {
            "enabled": True,
            "keywords": ["mundo"],
            "emojis": [{"emoji": "🔥", "keyword": "mundo"}],
            "style": {
                "preset": "karaoke",
                "highlight_color": "&H0000FFFF",
                "emoji": True,
                "censor": {"words": ["cruel"], "beep": True},
            },
            "segments": [{
                "start": 0.0, "end": 3.2, "text": "hola mundo cruel",
                "words": _words([(0.2, 0.8, "hola"), (1.0, 1.7, "mundo"), (2.0, 2.8, "cruel")]),
            }],
        },
        "on_screen_text": {"enabled": True, "items": [
            {"text": "HOOK", "start": 0.0, "end": 1.5, "position": "top", "entrance": True},
        ]},
    }
    result = RenderJob(settings=_settings(tmp_path), job=_job("r-karaoke", payload)).execute()
    applied = result["applied"]
    assert applied["captions"]["preset"] == "karaoke"
    assert "***" in applied["captions"]["text"]
    assert applied["captions"]["emoji_applied"] == 1
    assert applied["captions"]["font_file"] == "Montserrat-Bold.ttf"
    assert applied["on_screen_text"]["texts"][0]["entrance"] is True
    checks = applied["frame_checks"]
    assert checks["captions_visible"] is True
    assert {s["what"] for s in checks["samples"]} >= {"captions", "background"}
    for row in applied["captions"]["word_sync"]:
        assert abs(row["ass"] - row["whisperx"]) <= 0.08
