"""T2: `shorts` karaoke preset stays in the 0.70-0.78 band, clear of the TikTok/Reels UI."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

from app.tools.captions import (
    SAFE_BAND,
    build_ass,
    caption_search_window,
    line_chars,
    resolve_caption_style,
    safe_zone,
)

W, H = 1080, 1920
FONTS = Path(__file__).resolve().parents[1] / "app" / "assets" / "fonts"


def _captions(words, **style):
    segs = [{
        "start": words[0][0], "end": words[-1][1], "text": " ".join(w for _, _, w in words),
        "words": [{"start": a, "end": b, "word": w} for a, b, w in words],
    }]
    return {"enabled": True, "segments": segs, "style": {"preset": "shorts", **style}}


def _seq(text: str, step: float = 0.4):
    return [(round(i * step, 2), round(i * step + step - 0.05, 2), w) for i, w in enumerate(text.split())]


def _caption_events(content: str) -> list[str]:
    return [line for line in content.splitlines() if line.startswith("Dialogue: 0,")]


def test_shorts_style_is_pop_white_thick_yellow_active():
    visual = resolve_caption_style({"preset": "shorts"})
    assert visual["name"] == "shorts" and visual["mode"] == "pop"
    assert visual["primary"] == "&H00FFFFFF" and visual["highlight"] == "&H0000FFFF"
    assert visual["outline"] >= 8
    content, _ = build_ass(width=W, height=H, captions=_captions(_seq("uno dos")), on_screen_text=None, duration=2.0)
    style = next(line for line in content.splitlines() if line.startswith("Style: Caption"))
    assert "Montserrat" in style and ",9,2,2," in style  # outline 9, shadow 2, bottom
    assert "WrapStyle: 2" in content
    # Active word yellow, the others white.
    first = _caption_events(content)[0]
    assert "\\c&H0000FFFF&" in first and "\\c&H00FFFFFF&" in first


def test_shorts_shows_two_or_three_words_per_line():
    text = "this is a quick test of the new karaoke captions for shorts"
    content, summary = build_ass(width=W, height=H, captions=_captions(_seq(text)), on_screen_text=None, duration=6.0)
    lines = summary["captions"]["text"]
    assert summary["captions"]["preset"] == "shorts"
    events = _caption_events(content)
    shown = [re.sub(r"\{[^}]*\}", "", e.split(",,", 1)[1].split(",", 3)[-1]) for e in events]
    for words in shown:
        assert 1 <= len(words.split()) <= 3
    assert lines


def test_long_words_split_the_line_and_never_wrap():
    max_chars = line_chars(W, round(H * 0.046), round(W * 0.14), pop_scale=112)
    text = "extraordinarily unbelievable internationalization"
    content, _ = build_ass(width=W, height=H, captions=_captions(_seq(text)), on_screen_text=None, duration=3.0)
    for event in _caption_events(content):
        body = re.sub(r"\{[^}]*\}", "", event.split(",,", 1)[1].split(",", 3)[-1])
        if len(body) > max_chars:
            assert "\\fs" in event  # shrunk, not wrapped
    assert "\\N" not in content


def test_safe_zone_summary_in_band_and_clear_of_ui():
    _, summary = build_ass(
        width=W, height=H, captions=_captions(_seq("water industry scam right here")),
        on_screen_text=None, duration=3.0,
    )
    zone = summary["captions"]["safe_zone"]
    assert zone["in_band"] is True
    assert SAFE_BAND[0] <= zone["top"] < zone["bottom"] <= SAFE_BAND[1]
    assert zone["left"] >= 0.14 - 1e-6 and zone["right"] <= 0.86 + 1e-6
    assert zone["ui_overlap"] == []


def test_safe_zone_flags_ui_overlap():
    low = safe_zone(width=W, height=H, size=80, margin_v=100, margin_lr=60, outline=6, align=2)
    assert "bottom_bar" in low["ui_overlap"] and low["in_band"] is False
    wide = safe_zone(width=W, height=H, size=80, margin_v=430, margin_lr=20, outline=6, align=2)
    assert wide["ui_overlap"] == ["right_rail"]


def test_search_window_follows_the_shorts_band():
    win = caption_search_window(W, H, {"preset": "shorts"})
    bottom = win["y"] + win["h"]
    assert H * 0.76 <= bottom <= H * 0.80
    classic = caption_search_window(W, H, {"preset": "classic"})
    assert classic["y"] + classic["h"] != bottom  # band comes from the preset


@pytest.mark.parametrize("text", ["water industry scam", "Okay, okay,", "unbelievable internationalization"])
def test_libass_pixels_stay_in_the_band(tmp_path, text):
    """Real render: the burned text sits inside 0.70-0.78 and between the side rails."""
    content, _ = build_ass(width=W, height=H, captions=_captions(_seq(text, 0.5)), on_screen_text=None, duration=2.0)
    ass = tmp_path / "c.ass"
    ass.write_text(content, encoding="utf-8")
    proc = subprocess.run(
        [
            "ffmpeg", "-v", "info", "-f", "lavfi", "-i", f"color=c=black:size={W}x{H}:rate=10:duration=2",
            "-vf", f"subtitles={ass}:fontsdir={FONTS},bbox=min_val=40", "-f", "null", "-",
        ],
        capture_output=True, text=True, check=True, cwd=tmp_path,
    )
    boxes = [tuple(map(int, m)) for m in re.findall(r"x1:(\d+) x2:(\d+) y1:(\d+) y2:(\d+)", proc.stderr)]
    boxes = [b for b in boxes if b[1] > b[0]]
    assert boxes, "no text rendered"
    x1 = min(b[0] for b in boxes); x2 = max(b[1] for b in boxes)
    y1 = min(b[2] for b in boxes); y2 = max(b[3] for b in boxes)
    assert y1 >= H * SAFE_BAND[0] and y2 <= H * SAFE_BAND[1]
    assert x1 >= W * 0.12 and x2 <= W * 0.86


def test_lines_break_at_sentence_end():
    from app.tools.captions import _group_lines

    items = [{"text": w, "start": i * 0.3, "end": i * 0.3 + 0.25} for i, w in enumerate("I swear. I never said that, okay? Okay, okay, so".split())]
    lines = [" ".join(w["text"] for w in ln["words"]) for ln in _group_lines(items, max_words=3, duration=None)]
    assert lines == ["I swear.", "I never said", "that, okay?", "Okay, okay,", "so"]
    assert not any(re.search(r"[.?!]\s", ln) for ln in lines)  # no sentence end inside a line
    assert all(len(ln.split()) <= 3 for ln in lines)


def test_single_comma_word_does_not_force_a_one_word_line():
    from app.tools.captions import _ends_clause

    assert _ends_clause("Okay,", 1) is False
    assert _ends_clause("okay,", 2) is True
    assert _ends_clause("swear.", 1) is True
    assert _ends_clause('said."', 1) is True
    assert _ends_clause("3.5", 1) is False
