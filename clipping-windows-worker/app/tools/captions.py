"""Subtítulos y texto en pantalla como ASS (contrato render_spec).

El VPS manda los segmentos de la transcripción ya recortados al clip (tiempos
relativos) y el diccionario de marca. Aquí:
  * se aplica el diccionario de marca (ortografía exacta: "boxabl" → "BOXABL");
  * se trocean en líneas cortas estilo shorts (por palabra si hay `words`);
  * se añaden los textos en pantalla / CTA como eventos con otro estilo.

`captions.style.preset` elige el look. `classic` (defecto) es el burn-in
estático de siempre. `karaoke`, `pop` y `bold` resaltan la palabra activa
con los tiempos de WhisperX (cuantizados a centésimas, error ≤ 80 ms).

`shorts` (T2) parte de `pop`: blanco con borde grueso, 2-3 palabras en una
sola línea y la palabra activa en amarillo. El texto vive en la franja
0.70-0.78 de la altura y deja libres los márgenes laterales, fuera de la
interfaz de TikTok/Reels (`UI_ZONES`). El resumen trae `safe_zone` con la
caja estimada del texto y las zonas de interfaz que pisa.

Los emojis no van dentro del ASS: libass no pinta emoji a color. El resumen
trae `emoji_cues` para que el render los superponga como PNG (Twemoji).
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Iterable

_WORD = re.compile(r"\S+")

PRESETS = ("classic", "karaoke", "pop", "bold", "shorts")

# TikTok/Reels interface on a 9:16 frame, as fractions (x0, y0, x1, y1):
# status/tabs on top, caption + buttons at the bottom, action rail on the right.
UI_ZONES = {
    "top_bar": (0.0, 0.0, 1.0, 0.08),
    "bottom_bar": (0.0, 0.80, 1.0, 1.0),
    "right_rail": (0.86, 0.35, 1.0, 0.80),
}
# Caption band the `shorts` preset keeps to (top, bottom).
SAFE_BAND = (0.70, 0.78)
# Montserrat Bold: average advance per character, in em (mixed case).
_CHAR_EM = 0.62

# Family name → file shipped under app/assets/fonts (OFL).
_FONT_FILES = {
    "montserrat": "Montserrat-Bold.ttf",
    "anton": "Anton-Regular.ttf",
}

# ASS colours are &H00BBGGRR (no trailing &). First speaker stays white so a
# single diarized voice matches the classic burn-in.
_SPEAKER_COLOURS = (
    "&H00FFFFFF",
    "&H00FFE599",
    "&H0080FF99",
    "&H0099A0FF",
    "&H00FF99E5",
)

_PRESET_VISUAL = {
    "classic": {
        "font": "Arial",
        "uppercase": False,
        "outline": 6,
        "shadow": 2,
        "bold": -1,
        "mode": "static",
        "primary": "&H00FFFFFF",
        "secondary": "&H00FFFFFF",
        "highlight": "&H0000FFFF",
    },
    "karaoke": {
        "font": "Montserrat",
        "uppercase": False,
        "outline": 5,
        "shadow": 2,
        "bold": -1,
        "mode": "karaoke",
        "primary": "&H0000FFFF",
        "secondary": "&H00FFFFFF",
        "highlight": "&H0000FFFF",
    },
    "pop": {
        "font": "Montserrat",
        "uppercase": False,
        "outline": 5,
        "shadow": 2,
        "bold": -1,
        "mode": "pop",
        "primary": "&H00FFFFFF",
        "secondary": "&H00FFFFFF",
        "highlight": "&H0000FFFF",
    },
    "bold": {
        "font": "Anton",
        "uppercase": True,
        "outline": 8,
        "shadow": 3,
        "bold": 0,
        "mode": "karaoke",
        "primary": "&H0000FFFF",
        "secondary": "&H00FFFFFF",
        "highlight": "&H0000FFFF",
    },
    # T2: pop look tuned for TikTok/Reels. Layout keys are defaults that
    # explicit style keys (size, margin_v, max_words) still override.
    "shorts": {
        "font": "Montserrat",
        "uppercase": False,
        "outline": 9,
        "shadow": 2,
        "bold": -1,
        "mode": "pop",
        "primary": "&H00FFFFFF",
        "secondary": "&H00FFFFFF",
        "highlight": "&H0000FFFF",
        "pop_scale": 112,
        "max_words": 3,
        "size_frac": 0.046,
        "margin_v_frac": 0.225,
        "side_frac": 0.14,
        "single_line": True,
    },
}


def _norm(s: str) -> str:
    return re.sub(r"[^0-9a-z]+", "", s.lower())


def apply_brand_dictionary(text: str, dictionary: Iterable[str]) -> str:
    """Corrige la ortografía de las marcas (1–3 palabras) conservando la puntuación."""
    entries = [d.strip() for d in dictionary or [] if d and d.strip()]
    if not entries or not text:
        return text
    table = {_norm(e): e for e in entries if _norm(e)}
    tokens = text.split(" ")
    out: list[str] = []
    i = 0
    while i < len(tokens):
        for n in (3, 2, 1):
            if i + n > len(tokens):
                continue
            window = tokens[i:i + n]
            key = _norm("".join(window))
            if key and key in table:
                lead = re.match(r"^\W*", window[0]).group(0)
                trail = re.search(r"\W*$", window[-1]).group(0)
                out.append(f"{lead}{table[key]}{trail}")
                i += n
                break
        else:
            out.append(tokens[i])
            i += 1
    return " ".join(out)


def _ts(t: float) -> str:
    t = max(0.0, float(t))
    h = int(t // 3600)
    m = int((t % 3600) // 60)
    s = t - h * 3600 - m * 60
    return f"{h:d}:{m:02d}:{s:05.2f}"


def _quant(t: float) -> float:
    """Same centisecond grid that `_ts` writes into the ASS dialogue."""
    text = _ts(t)
    h, m, rest = text.split(":")
    return int(h) * 3600 + int(m) * 60 + float(rest)


def _esc(text: str) -> str:
    return text.replace("\\", "\\\\").replace("{", "(").replace("}", ")").replace("\n", "\\N")


def ass_bgr(value: str | None, default: str = "&H0000FFFF") -> str:
    """Accept #RRGGBB or &HBBGGRR / &HAABBGGRR and return &H00BBGGRR."""
    if not value:
        return default
    raw = str(value).strip()
    if raw.startswith("#") and len(raw) == 7:
        r, g, b = raw[1:3], raw[3:5], raw[5:7]
        return f"&H00{b}{g}{r}".upper()
    if raw.lower().startswith("&h"):
        body = raw[2:].rstrip("&")
        if len(body) == 8 and all(c in "0123456789abcdefABCDEF" for c in body):
            return "&H" + body.upper()
        if len(body) == 6 and all(c in "0123456789abcdefABCDEF" for c in body):
            return "&H00" + body.upper()
    return default


def font_file_for(font_name: str | None) -> str | None:
    if not font_name:
        return None
    return _FONT_FILES.get(str(font_name).strip().lower())


def _emoji_enabled(value: Any) -> bool:
    if isinstance(value, dict):
        return bool(value.get("enabled", True))
    return bool(value)


def _censor_config(style: dict[str, Any]) -> tuple[list[str], bool]:
    raw = style.get("censor")
    if not raw:
        return [], False
    if isinstance(raw, str):
        text = raw.strip()
        return ([text] if text else []), False
    if isinstance(raw, list):
        return [str(w).strip() for w in raw if str(w).strip()], False
    if isinstance(raw, dict):
        words = [str(w).strip() for w in (raw.get("words") or []) if str(w).strip()]
        return words, bool(raw.get("beep"))
    return [], False


def caption_safe_band(style: dict[str, Any] | None) -> tuple[float, float] | None:
    """Height band the captions must keep to, or None when the style has none.

    Only the `shorts` preset is laid out for the TikTok/Reels UI (the VPS
    verifier checks its 0.70-0.78 band); other presets are free to move.
    """
    return SAFE_BAND if resolve_caption_style(style)["name"] == "shorts" else None


def resolve_caption_style(style: dict[str, Any] | None) -> dict[str, Any]:
    """Preset defaults, then explicit style keys. Unknown preset → classic."""
    src = style or {}
    name = str(src.get("preset") or "classic").strip().lower()
    if name not in _PRESET_VISUAL:
        name = "classic"
    out = dict(_PRESET_VISUAL[name])
    out["name"] = name
    if src.get("font"):
        out["font"] = str(src["font"]).strip() or out["font"]
    if "uppercase" in src and src["uppercase"] is not None:
        out["uppercase"] = bool(src["uppercase"])
    if src.get("highlight_color"):
        out["highlight"] = ass_bgr(str(src["highlight_color"]), out["highlight"])
        if out["mode"] == "karaoke":
            out["primary"] = out["highlight"]
    return out


def caption_search_window(width: int, height: int, style: dict[str, Any] | None) -> dict[str, Any]:
    """Franja donde libass pinta el estilo, para muestrear el frame ya quemado.

    Misma ancla que ``build_ass``: ``bottom`` apoya el texto en MarginV desde
    abajo (crece hacia arriba), ``top`` desde arriba, y ``middle``/``center``
    (alineación 5) ignoran MarginV y quedan centrados. La región cubre unas
    dos líneas; el detector se desliza dentro con una ventana de una línea.
    """
    src = style if isinstance(style, dict) else {}
    visual = resolve_caption_style(src)
    size_default = round(height * float(visual.get("size_frac", 0.038)))
    margin_default = round(height * float(visual.get("margin_v_frac", 0.22)))
    try:
        size = int(src.get("size") or size_default)
    except (TypeError, ValueError):
        size = size_default
    size = max(8, min(int(size), max(8, height)))
    if src.get("margin_v") is None:
        margin_v = margin_default
    else:
        try:
            margin_v = int(src.get("margin_v"))
        except (TypeError, ValueError):
            margin_v = margin_default
    outline = int(visual.get("outline") or 6)
    shadow = int(visual.get("shadow") or 0)
    position = str(src.get("position") or "bottom").strip().lower()
    if position not in {"bottom", "top", "middle", "center"}:
        position = "bottom"
    line_h = max(16, min(height, int(size + outline * 3 + shadow + 8)))
    region_h = max(line_h, min(height, int(line_h * 2.6)))
    pad = outline * 2
    if position in {"middle", "center"}:
        y = max(0, (height - region_h) // 2)
    elif position == "top":
        y = max(0, min(height - region_h, margin_v - pad))
    else:
        text_bottom = height - margin_v + pad
        y = max(0, min(height - region_h, text_bottom - region_h))
    return {
        "position": position,
        "x": 0,
        "y": int(y),
        "w": int(width),
        "h": int(region_h),
        "line_h": int(line_h),
        # Horizontal slice. Short centered words fill ~40% of the width;
        # a full-width average dilutes them under the visibility floor.
        "win_frac": 0.40,
    }


def caption_events(segments: list[dict[str, Any]], *, max_words: int = 4, duration: float | None = None,
                   dictionary: Iterable[str] = ()) -> list[tuple[float, float, str]]:
    """(start, end, text) por línea corta. Usa `words` si vienen; si no, reparte el segmento."""
    events: list[tuple[float, float, str]] = []
    for seg in segments or []:
        s0, s1 = float(seg.get("start", 0.0)), float(seg.get("end", 0.0))
        words = [w for w in (seg.get("words") or []) if str(w.get("word") or w.get("text") or "").strip()]
        if words and all("start" in w and "end" in w for w in words):
            items = [(float(w["start"]), float(w["end"]), str(w.get("word") or w.get("text")).strip()) for w in words]
        else:
            toks = _WORD.findall(str(seg.get("text") or ""))
            if not toks or s1 <= s0:
                continue
            step = (s1 - s0) / len(toks)
            items = [(s0 + k * step, s0 + (k + 1) * step, t) for k, t in enumerate(toks)]
        for k in range(0, len(items), max_words):
            chunk = items[k:k + max_words]
            a, b = chunk[0][0], chunk[-1][1]
            if duration is not None:
                if a >= duration:
                    continue
                b = min(b, duration)
            a = max(0.0, a)
            if b - a < 0.05:
                continue
            text = apply_brand_dictionary(" ".join(c[2] for c in chunk), dictionary)
            events.append((a, b, text))
    return events


def _speaker_ids(segments: list[dict[str, Any]] | None) -> list[str]:
    found: set[str] = set()
    for seg in segments or []:
        if seg.get("speaker"):
            found.add(str(seg["speaker"]))
        for word in seg.get("words") or []:
            if isinstance(word, dict) and word.get("speaker"):
                found.add(str(word["speaker"]))
    return sorted(found)


def _raw_words(segments: list[dict[str, Any]] | None, *, duration: float | None) -> list[dict[str, Any]]:
    """Word timings. WhisperX `words` win; otherwise the segment is split evenly."""
    items: list[dict[str, Any]] = []
    for seg in segments or []:
        s0, s1 = float(seg.get("start", 0.0)), float(seg.get("end", 0.0))
        seg_speaker = str(seg["speaker"]) if seg.get("speaker") else None
        words = [w for w in (seg.get("words") or []) if isinstance(w, dict) and str(w.get("word") or w.get("text") or "").strip()]
        timed = bool(words) and all("start" in w and "end" in w for w in words)
        if timed:
            raw = []
            for w in words:
                raw.append((
                    float(w["start"]),
                    float(w["end"]),
                    str(w.get("word") or w.get("text")).strip(),
                    str(w["speaker"]) if w.get("speaker") else seg_speaker,
                ))
        else:
            toks = _WORD.findall(str(seg.get("text") or ""))
            if not toks or s1 <= s0:
                continue
            step = (s1 - s0) / len(toks)
            raw = [(s0 + k * step, s0 + (k + 1) * step, t, seg_speaker) for k, t in enumerate(toks)]
        for start, end, text, speaker in raw:
            if duration is not None and start >= duration:
                continue
            if duration is not None:
                end = min(end, duration)
            start = max(0.0, start)
            if end - start < 0.02 and (duration is None or end <= duration):
                # keep very short words; the line filter drops empty lines later
                end = max(end, start)
            if not text:
                continue
            items.append({"start": start, "end": max(end, start), "text": text, "speaker": speaker})
    return items


def _brand_words(items: list[dict[str, Any]], dictionary: Iterable[str]) -> list[dict[str, Any]]:
    entries = [d.strip() for d in dictionary or [] if d and str(d).strip()]
    if not entries:
        return [dict(it) for it in items]
    table = {_norm(e): e for e in entries if _norm(e)}
    out: list[dict[str, Any]] = []
    i = 0
    while i < len(items):
        matched = False
        for n in (3, 2, 1):
            if i + n > len(items):
                continue
            window = items[i:i + n]
            key = _norm("".join(w["text"] for w in window))
            if not key or key not in table:
                continue
            lead = re.match(r"^\W*", window[0]["text"]).group(0)
            trail = re.search(r"\W*$", window[-1]["text"]).group(0)
            merged = dict(window[0])
            merged["text"] = f"{lead}{table[key]}{trail}"
            merged["end"] = window[-1]["end"]
            out.append(merged)
            i += n
            matched = True
            break
        if not matched:
            out.append(dict(items[i]))
            i += 1
    return out


def _censor_words(items: list[dict[str, Any]], phrases: list[str]) -> tuple[list[dict[str, Any]], list[dict[str, float]]]:
    """Replace listed words with `***`. A phrase collapses into one token."""
    keys = [_norm(p) for p in phrases if _norm(p)]
    if not keys:
        return [dict(it) for it in items], []
    keyset = set(keys)
    spans: list[dict[str, float]] = []
    out: list[dict[str, Any]] = []
    i = 0
    while i < len(items):
        matched_n = 0
        for n in (3, 2, 1):
            if i + n > len(items):
                continue
            window = items[i:i + n]
            if _norm("".join(w["text"] for w in window)) in keyset:
                matched_n = n
                break
        if not matched_n:
            out.append(dict(items[i]))
            i += 1
            continue
        window = items[i:i + matched_n]
        token = dict(window[0])
        token["text"] = "***"
        token["end"] = window[-1]["end"]
        token["censored"] = True
        out.append(token)
        spans.append({"start": round(float(window[0]["start"]), 3), "end": round(float(window[-1]["end"]), 3)})
        i += matched_n
    return out, spans


_SENTENCE_END = re.compile(r"[.!?\u2026;:]+[\"'\u201d\u2019)\]]*$")
_COMMA_END = re.compile(r",[\"'\u201d\u2019)\]]*$")


def _ends_clause(text: str, words_in_line: int) -> bool:
    """Line break after this word: sentence punctuation, or a comma once the line has 2 words."""
    if _SENTENCE_END.search(text):
        return True
    return words_in_line >= 2 and bool(_COMMA_END.search(text))


def _group_lines(items: list[dict[str, Any]], *, max_words: int, duration: float | None,
                 max_chars: int | None = None) -> list[dict[str, Any]]:
    lines: list[dict[str, Any]] = []
    buf: list[dict[str, Any]] = []

    def flush() -> None:
        if not buf:
            return
        start, end = buf[0]["start"], buf[-1]["end"]
        if duration is not None:
            end = min(end, duration)
        if end - start >= 0.05:
            lines.append({
                "start": start,
                "end": end,
                "speaker": buf[0].get("speaker"),
                "words": [dict(w) for w in buf],
            })
        buf.clear()

    for item in items:
        speaker = item.get("speaker")
        if buf and _ends_clause(buf[-1]["text"], len(buf)):
            flush()  # T2b: never carry a sentence end into the next words ("swear. I")
        chars = sum(len(w["text"]) + 1 for w in buf) + len(item["text"])
        if buf and (
            len(buf) >= max_words
            or (max_chars is not None and chars > max_chars)
            or (speaker and buf[0].get("speaker") and speaker != buf[0].get("speaker"))
        ):
            flush()
        buf.append(item)
    flush()
    return lines


def _display(text: str, *, uppercase: bool) -> str:
    shown = text.upper() if uppercase else text
    return shown


def _speaker_colour(speaker: str | None, ids: list[str]) -> str:
    if not speaker or speaker not in ids:
        return "&H00FFFFFF"
    return _SPEAKER_COLOURS[ids.index(speaker) % len(_SPEAKER_COLOURS)]


def _keyword_hit(text: str, keywords: set[str]) -> bool:
    key = _norm(text)
    return bool(key) and key in keywords


def _k_plan(words: list[dict[str, Any]]) -> list[tuple[dict[str, Any], int, float]]:
    """Word, karaoke duration in centiseconds, and the activation time on the ASS grid."""
    if not words:
        return []
    event = int(round(_quant(words[0]["start"]) * 100))
    bounds: list[int] = []
    for i, word in enumerate(words):
        if i + 1 < len(words):
            bounds.append(int(round(_quant(words[i + 1]["start"]) * 100)))
        else:
            bounds.append(max(int(round(_quant(word["end"]) * 100)), int(round(_quant(word["start"]) * 100)) + 1))
    cursor = event
    plan: list[tuple[dict[str, Any], int, float]] = []
    for word, bound in zip(words, bounds):
        bound = max(cursor + 1, bound)
        plan.append((word, bound - cursor, cursor / 100.0))
        cursor = bound
    return plan


def _karaoke_body(line: dict[str, Any], visual: dict[str, Any], *, uppercase: bool, keywords: set[str],
                  speaker_colour: str) -> tuple[str, list[dict[str, Any]], float]:
    highlight = visual["highlight"]
    base = speaker_colour or visual["secondary"]
    plan = _k_plan(line["words"])
    parts = [f"{{\\1c{highlight}&\\2c{base}&}}"]
    sync: list[dict[str, Any]] = []
    for word, dur, ass_t in plan:
        shown = _display(word["text"], uppercase=uppercase)
        sync.append({
            "word": shown,
            "whisperx": round(float(word["start"]), 3),
            "ass": round(ass_t, 3),
        })
        token = _esc(shown)
        if _keyword_hit(word["text"], keywords):
            token = "{\\b1}" + token + "{\\b0}"
        parts.append(f"{{\\k{dur}}}{token} ")
    total_cs = sum(dur for _, dur, _ in plan)
    line_end = max(float(line["end"]), plan[0][2] + total_cs / 100.0)
    return "".join(parts).rstrip(), sync, line_end


def _pop_body(line: dict[str, Any], active: int, visual: dict[str, Any], *, uppercase: bool,
              keywords: set[str], speaker_colour: str) -> str:
    base = speaker_colour or "&H00FFFFFF"
    highlight = visual["highlight"]
    peak = int(visual.get("pop_scale", 132))
    settle = 100 + (peak - 100) // 4
    fs = line.get("fs")
    reset = f"{{\\r\\fs{fs}}}" if fs else "{\\r}"
    chunks: list[str] = []
    for i, word in enumerate(line["words"]):
        shown = _esc(_display(word["text"], uppercase=uppercase))
        if i == active:
            chunks.append(
                f"{{\\c{highlight}&\\fscx100\\fscy100\\t(0,90,\\fscx{peak}\\fscy{peak})"
                f"\\t(90,180,\\fscx{settle}\\fscy{settle})}}{shown}{reset}"
            )
        else:
            weight = "\\b1" if _keyword_hit(word["text"], keywords) else ""
            chunks.append(f"{{\\c{base}&{weight}}}{shown}{reset}")
    body = " ".join(chunks)
    return f"{{\\fs{fs}}}{body}" if fs else body


def _styled_dialogues(lines: list[dict[str, Any]], visual: dict[str, Any], *, uppercase: bool,
                      keywords: set[str], speaker_ids: list[str]) -> tuple[list[str], list[dict[str, Any]], list[str]]:
    dialogues: list[str] = []
    sync: list[dict[str, Any]] = []
    texts: list[str] = []
    for line in lines:
        shown_words = [_display(w["text"], uppercase=uppercase) for w in line["words"]]
        texts.append(" ".join(shown_words))
        colour = _speaker_colour(line.get("speaker"), speaker_ids)
        if visual["mode"] == "karaoke":
            body, line_sync, end_t = _karaoke_body(
                line, visual, uppercase=uppercase, keywords=keywords, speaker_colour=colour,
            )
            sync.extend(line_sync)
            dialogues.append(
                f"Dialogue: 0,{_ts(line['start'])},{_ts(end_t)},Caption,,0,0,0,,{body}"
            )
            continue
        if visual["mode"] == "pop":
            for i, word in enumerate(line["words"]):
                nxt = line["words"][i + 1]["start"] if i + 1 < len(line["words"]) else word["end"]
                end = max(float(nxt), float(word["start"]) + 0.05)
                body = _pop_body(
                    line, i, visual, uppercase=uppercase, keywords=keywords, speaker_colour=colour,
                )
                dialogues.append(
                    f"Dialogue: 0,{_ts(word['start'])},{_ts(end)},Caption,,0,0,0,,{body}"
                )
                sync.append({
                    "word": _display(word["text"], uppercase=uppercase),
                    "whisperx": round(float(word["start"]), 3),
                    "ass": round(_quant(word["start"]), 3),
                })
            continue
        # static (classic with censor, case, or a second speaker)
        text = _esc(" ".join(shown_words))
        if colour != "&H00FFFFFF":
            text = f"{{\\c{colour}&}}{text}"
        dialogues.append(
            f"Dialogue: 0,{_ts(line['start'])},{_ts(line['end'])},Caption,,0,0,0,,{text}"
        )
        for word in line["words"]:
            sync.append({
                "word": _display(word["text"], uppercase=uppercase),
                "whisperx": round(float(word["start"]), 3),
                "ass": round(_quant(word["start"]), 3),
            })
    return dialogues, sync, texts


def _place_emojis(cues: Any, lines: list[dict[str, Any]], duration: float) -> list[dict[str, Any]]:
    placed: list[dict[str, Any]] = []
    used: set[tuple[float, str]] = set()
    for raw in list(cues or [])[:4]:
        if isinstance(raw, str):
            raw = {"emoji": raw}
        if not isinstance(raw, dict):
            continue
        emoji = str(raw.get("emoji") or raw.get("char") or "").strip()
        if not emoji or len(emoji) > 16:
            continue
        start = raw.get("start")
        end = raw.get("end")
        keyword = str(raw.get("keyword") or "").strip()
        if start is None and keyword:
            needle = _norm(keyword)
            for line in lines:
                for word in line["words"]:
                    marker = (round(float(word["start"]), 3), _norm(word["text"]))
                    if marker in used or _norm(word["text"]) != needle:
                        continue
                    start = float(word["start"])
                    end = min(duration, max(float(word["end"]), start + 1.2))
                    used.add(marker)
                    break
                if start is not None:
                    break
        if start is None:
            continue
        try:
            start_f = max(0.0, float(start))
        except (TypeError, ValueError):
            continue
        if start_f >= duration:
            continue
        try:
            end_f = float(end) if end is not None else min(duration, start_f + 1.2)
        except (TypeError, ValueError):
            end_f = min(duration, start_f + 1.2)
        end_f = min(duration, max(end_f, start_f + 0.4))
        cue = {"emoji": emoji, "start": round(start_f, 3), "end": round(end_f, 3)}
        if keyword:
            cue["keyword"] = keyword
        placed.append(cue)
        if len(placed) >= 2:
            break
    return placed


def _entrance_tag(width: int, height: int, position: str, top_margin: int | None = None) -> str:
    anchor = _POS_ALIGN.get(position, 8)
    x = width // 2
    if anchor == 8 and top_margin is not None:
        y_to = int(top_margin)
    elif anchor == 8:
        y_to = int(height * 0.10)
    elif anchor == 2:
        y_to = int(height * 0.78)
    else:
        y_to = int(height * 0.46)
    y_from = max(0, y_to - int(height * 0.055))
    return (
        f"{{\\an{anchor}\\move({x},{y_from},{x},{y_to},0,280)"
        f"\\fscx85\\fscy85\\t(0,280,\\fscx100\\fscy100)\\fad(120,180)}}"
    )


_POS_ALIGN = {"bottom": 2, "center": 5, "middle": 5, "top": 8}


def _ass_colour(value: Any, default: str) -> str:
    """Validate an ASS &HAABBGGRR colour; fall back to `default` when malformed."""
    s = str(value or "").strip()
    if re.fullmatch(r"&H[0-9A-Fa-f]{6,8}(&)?", s):
        return s.rstrip("&")
    return default


def _header(*, width: int, height: int, font: str, size: int, primary: str, secondary: str,
            outline: int, shadow: int, bold: int, align: int, margin_v: int, ost_size: int,
            outline_colour: str = "&H00000000", overlay_primary: str = "&H00FFFFFF",
            margin_lr: int = 60, wrap_style: int = 0, ost_margin_v: int | None = None) -> str:
    return (
        "[Script Info]\nScriptType: v4.00+\n"
        f"PlayResX: {width}\nPlayResY: {height}\nWrapStyle: {wrap_style}\nScaledBorderAndShadow: yes\n\n"
        "[V4+ Styles]\n"
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, "
        "Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, "
        "MarginR, MarginV, Encoding\n"
        f"Style: Caption,{font},{size},{primary},{secondary},{outline_colour},&H64000000,{bold},0,0,0,100,100,0,0,1,{outline},{shadow},"
        f"{align},{margin_lr},{margin_lr},{margin_v},1\n"
        f"Style: Overlay,{font},{ost_size},{overlay_primary},{overlay_primary},{outline_colour},&HA0000000,-1,0,0,0,100,100,0,0,3,4,0,"
        f"8,60,60,{round(height * 0.12) if ost_margin_v is None else int(ost_margin_v)},1\n\n"
        "[Events]\nFormat: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
    )


def _overlay_lines(ost: dict[str, Any], *, duration: float, width: int, height: int) -> tuple[list[str], list[dict[str, Any]]]:
    lines: list[str] = []
    texts: list[dict[str, Any]] = []
    if not ost.get("enabled"):
        return lines, texts
    for it in ost.get("items") or []:
        t = str(it.get("text") or "").strip()
        if not t:
            continue
        a = float(it.get("start", 0.0))
        b = float(it.get("end") if it.get("end") is not None else duration)
        b = min(b, duration)
        if b <= a:
            continue
        position = it.get("position", "top")
        align = _POS_ALIGN.get(position, 8)
        if it.get("entrance"):
            prefix = _entrance_tag(width, height, position, (ost.get("style") or {}).get("margin_v"))
        else:
            prefix = f"{{\\an{align}}}"
        lines.append(f"Dialogue: 1,{_ts(a)},{_ts(b)},Overlay,,0,0,0,,{prefix}{_esc(t)}")
        texts.append({"text": t, "start": a, "end": b, "entrance": bool(it.get("entrance"))})
    return lines, texts


def _prepare_lines(cap: dict[str, Any], *, max_words: int, duration: float,
                   censor_words: list[str], max_chars: int | None = None) -> tuple[list[dict[str, Any]], list[dict[str, float]]]:
    raw = _raw_words(cap.get("segments"), duration=duration)
    branded = _brand_words(raw, cap.get("brand_dictionary") or [])
    censored, spans = _censor_words(branded, censor_words)
    lines = _group_lines(censored, max_words=max_words, duration=duration, max_chars=max_chars)
    return lines, spans


def line_chars(width: int, size: int, margin_lr: int, *, pop_scale: int = 100) -> int:
    """Characters that fit on one line between the side margins (active word popped)."""
    usable = max(1, width - 2 * margin_lr)
    per_char = max(1.0, size * _CHAR_EM * pop_scale / 100.0)
    return max(4, int(usable / per_char))


def _fit_long_lines(lines: list[dict[str, Any]], *, max_chars: int, size: int) -> None:
    """A single word longer than the line gets a smaller \\fs instead of wrapping."""
    for line in lines:
        chars = len(" ".join(w["text"] for w in line["words"]))
        if chars > max_chars:
            line["fs"] = max(8, int(size * max_chars / chars))


def safe_zone(*, width: int, height: int, size: int, margin_v: int, margin_lr: int,
              outline: int, align: int, max_line_frac: float = 1.0) -> dict[str, Any]:
    """Estimated caption box (fractions of the frame) and the UI zones it touches.

    One line, bottom-anchored: the box spans the usable width between margins
    (worst case) and one line of text plus outline above MarginV.
    """
    line_h = size * 1.2 + 2 * outline
    if align == 2:
        bottom = (height - margin_v + outline) / height
        top = bottom - line_h / height
    elif align == 8:
        top = (margin_v - outline) / height
        bottom = top + line_h / height
    else:
        top = (height - line_h) / 2 / height
        bottom = top + line_h / height
    half = (width - 2 * margin_lr) * max_line_frac / 2 / width
    left, right = 0.5 - half, 0.5 + half
    overlap = [
        name for name, (x0, y0, x1, y1) in UI_ZONES.items()
        if left < x1 and right > x0 and top < y1 and bottom > y0
    ]
    return {
        "top": round(top, 4),
        "bottom": round(bottom, 4),
        "left": round(left, 4),
        "right": round(right, 4),
        "band": list(SAFE_BAND),
        "in_band": top >= SAFE_BAND[0] - 1e-6 and bottom <= SAFE_BAND[1] + 1e-6,
        "ui_overlap": overlap,
    }


def build_ass(*, width: int, height: int, captions: dict[str, Any] | None, on_screen_text: dict[str, Any] | None,
              duration: float, branding: dict[str, Any] | None = None) -> tuple[str, dict[str, Any]]:
    """Devuelve (contenido .ass, resumen de lo aplicado).

    `branding` (MOLI-54) sustituye fuente y colores cuando llega:
    font_name (familia, debe coincidir con la fuente en fontsdir),
    caption_color / outline_color en formato ASS &HAABBGGRR.
    Cualquier campo ausente mantiene el preset (`classic` = Arial estático).
    """
    cap = captions or {}
    ost = on_screen_text or {}
    br = branding or {}
    style = cap.get("style") or {}
    visual = resolve_caption_style(style)
    censor_words, censor_beep = _censor_config(style)
    speaker_ids = _speaker_ids(cap.get("segments"))
    uppercase_override = "uppercase" in style and style.get("uppercase") is not None
    plain_classic = (
        visual["name"] == "classic"
        and visual["font"] == "Arial"
        and visual["outline"] == 6
        and visual["shadow"] == 2
        and not visual["uppercase"]
        and not uppercase_override
        and not censor_words
        and len(speaker_ids) <= 1
    )
    # Brand font wins over the preset family so libass loads the downloaded file.
    font = br.get("font_name") or visual["font"]
    size = int(style.get("size") or round(height * float(visual.get("size_frac", 0.038))))
    cap_align = _POS_ALIGN.get(style.get("position", "bottom"), 2)
    cap_margin_v = int(style.get("margin_v") or round(height * float(visual.get("margin_v_frac", 0.22))))
    single_line = bool(visual.get("single_line"))
    margin_lr = round(width * float(visual["side_frac"])) if "side_frac" in visual else 60
    primary = _ass_colour(br.get("caption_color"), visual["primary"])
    secondary = primary if br.get("caption_color") else visual["secondary"]
    outline_c = _ass_colour(br.get("outline_color"), "&H00000000")
    overlay_primary = primary if br.get("caption_color") else "&H00FFFFFF"
    ost_size = int((ost.get("style") or {}).get("size", round(height * 0.034)))
    header = _header(
        width=width, height=height, font=font, size=size,
        primary=primary, secondary=secondary,
        outline=int(visual["outline"]), shadow=int(visual["shadow"]), bold=int(visual["bold"]),
        align=cap_align, margin_v=cap_margin_v, ost_size=ost_size,
        outline_colour=outline_c, overlay_primary=overlay_primary,
        margin_lr=margin_lr, wrap_style=2 if single_line else 0,
        ost_margin_v=(ost.get("style") or {}).get("margin_v"),
    )
    summary: dict[str, Any] = {
        "captions": {"applied": False, "events": 0, "preset": visual["name"]},
        "on_screen_text": {"applied": False, "texts": []},
    }
    dialogues: list[str] = []
    max_words = int(style.get("max_words") or visual.get("max_words", 4))
    max_chars = None
    if single_line:
        max_chars = line_chars(width, size, margin_lr, pop_scale=int(visual.get("pop_scale", 100)))
    emoji_on = _emoji_enabled(style.get("emoji"))
    need_lines = (not plain_classic) or emoji_on
    lines: list[dict[str, Any]] = []
    spans: list[dict[str, float]] = []
    shown_texts: list[str] = []
    if cap.get("enabled") and cap.get("segments") and need_lines:
        lines, spans = _prepare_lines(
            cap, max_words=max_words, duration=duration, censor_words=censor_words, max_chars=max_chars,
        )
        if max_chars is not None:
            _fit_long_lines(lines, max_chars=max_chars, size=size)

    if cap.get("enabled") and cap.get("segments") and plain_classic:
        evs = caption_events(
            cap["segments"], max_words=max_words, duration=duration,
            dictionary=cap.get("brand_dictionary") or [],
        )
        shown_texts = [t for _, _, t in evs]
        for a, b, t in evs:
            dialogues.append(f"Dialogue: 0,{_ts(a)},{_ts(b)},Caption,,0,0,0,,{_esc(t)}")
        summary["captions"] = {
            "applied": bool(evs),
            "events": len(evs),
            "preset": "classic",
            "brand_dictionary": list(cap.get("brand_dictionary") or []),
            "text": " ".join(t for _, _, t in evs)[:4000],
        }
    elif cap.get("enabled") and cap.get("segments"):
        keywords = {_norm(k) for k in (cap.get("keywords") or []) if _norm(str(k))}
        cap_dialogues, sync, texts = _styled_dialogues(
            lines, visual, uppercase=bool(visual["uppercase"]), keywords=keywords, speaker_ids=speaker_ids,
        )
        dialogues.extend(cap_dialogues)
        shown_texts = texts
        summary["captions"] = {
            "applied": bool(cap_dialogues),
            "events": len(cap_dialogues),
            "preset": visual["name"],
            "brand_dictionary": list(cap.get("brand_dictionary") or []),
            "text": " ".join(texts)[:4000],
            "word_sync": sync,
        }

    cap_summary = summary["captions"]
    longest = max((len(t) for t in shown_texts), default=0)
    if max_chars is not None:
        longest = min(longest, max_chars)  # longer lines were shrunk with \fs
    pop = int(visual.get("pop_scale", 100)) if visual["mode"] == "pop" else 100
    usable = max(1, width - 2 * margin_lr)
    line_frac = min(1.0, longest * size * _CHAR_EM * pop / 100.0 / usable) if longest else 1.0
    cap_summary["safe_zone"] = safe_zone(
        width=width, height=height, size=size, margin_v=cap_margin_v, margin_lr=margin_lr,
        outline=int(visual["outline"]), align=cap_align, max_line_frac=line_frac,
    )
    if br.get("font_name"):
        cap_summary["font"] = str(br["font_name"])
    if censor_words:
        cap_summary["censor_spans"] = spans
        cap_summary["censor_beep"] = bool(censor_beep and spans)
    else:
        cap_summary["censor_beep"] = False
    # Preset files (Montserrat/Anton). A brand font is downloaded separately.
    font_file = None if br.get("font_name") else font_file_for(font)
    if font_file:
        cap_summary["font_file"] = font_file
    if emoji_on:
        cap_summary["emoji_cues"] = _place_emojis(cap.get("emojis"), lines, duration)
    else:
        cap_summary["emoji_cues"] = []

    overlay, texts = _overlay_lines(ost, duration=duration, width=width, height=height)
    dialogues.extend(overlay)
    summary["on_screen_text"] = {"applied": bool(texts), "texts": texts}
    return header + "\n".join(dialogues) + "\n", summary


def write_ass(path: Path, **kw: Any) -> dict[str, Any]:
    content, summary = build_ass(**kw)
    path.write_text(content, encoding="utf-8")
    return summary
