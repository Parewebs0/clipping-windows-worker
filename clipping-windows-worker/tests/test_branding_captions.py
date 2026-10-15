"""MOLI-54 — branding en el worker: ASS con fuentes/colores, caption_events con position."""

from app.tools.captions import _ass_colour, apply_brand_dictionary, build_ass


def _ass(cap=None, ost=None, branding=None, duration=10.0):
    content, summary = build_ass(width=1080, height=1920, captions=cap,
                                 on_screen_text=ost, duration=duration, branding=branding)
    return content, summary


def test_default_style_unchanged_without_branding():
    cap = {"enabled": True, "segments": [{"start": 0.5, "end": 2.0, "text": "hola mundo"}]}
    content, summary = _ass(cap)
    assert "Style: Caption,Arial," in content
    assert "&H00FFFFFF" in content
    assert summary["captions"]["applied"] is True


def test_branding_sets_font_and_colors():
    cap = {"enabled": True,
           "segments": [{"start": 0.0, "end": 1.0, "text": "hello"}],
           "style": {"position": "bottom", "max_words": 4}}
    branding = {"font_name": "BrandFont", "caption_color": "&H002315EF",
                "outline_color": "&H00000000"}
    content, summary = _ass(cap, branding=branding)
    assert "Style: Caption,BrandFont," in content
    assert "&H002315EF" in content
    assert summary["captions"]["font"] == "BrandFont"


def test_malformed_color_falls_back():
    assert _ass_colour("&H00ZZZZ", "&H00FFFFFF") == "&H00FFFFFF"
    assert _ass_colour("&H002315EF", "&H00FFFFFF") == "&H002315EF"
    assert _ass_colour(None, "&H00000000") == "&H00000000"


def test_style_position_comes_from_branding_style_block():
    """El VPS aplasta branding.position en captions.style; el worker obedece style."""
    cap = {"enabled": True,
           "segments": [{"start": 0.0, "end": 1.0, "text": "hello"}],
           "style": {"position": "top", "max_words": 4}}
    content, _ = _ass(cap)
    # alignment 8 = top
    line = [l for l in content.splitlines() if l.startswith("Style: Caption")][0]
    parts = line.split(",")
    # Alignment is the 19th field per Format
    assert parts[18].strip() == "8"


def test_overlay_event_uses_top_alignment_even_with_branding_font():
    ost = {"enabled": True,
           "items": [{"text": "TOP TEXT", "start": 0.0, "end": 2.0, "position": "top"}]}
    branding = {"font_name": "BrandFont"}
    content, summary = _ass(None, ost=ost, branding=branding)
    assert "BrandFont" in content
    assert "\\an8" in content
    assert summary["on_screen_text"]["applied"] is True


def test_brand_dictionary_applied_with_branding():
    cap = {"enabled": True,
           "segments": [{"start": 0.0, "end": 1.0, "text": "visit boxabl now"}],
           "brand_dictionary": ["BOXABL"]}
    branding = {"font_name": "Brand"}
    _, summary = _ass(cap, branding=branding)
    assert "BOXABL" in summary["captions"]["text"]


def test_apply_brand_dictionary_multiword():
    assert apply_brand_dictionary("the boxabl casita tour", ["BOXABL CASITA"]) == "the BOXABL CASITA tour"