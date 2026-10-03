"""Subtítulos y texto en pantalla como ASS (contrato render_spec v2, issue #4).

El VPS manda los segmentos de la transcripción ya recortados al clip (tiempos
relativos) y el diccionario de marca. Aquí:
  * se aplica el diccionario de marca (ortografía exacta: "boxabl" → "BOXABL");
  * se trocean en líneas cortas estilo shorts (por palabra si hay `words`);
  * se añaden los textos en pantalla / CTA como eventos con otro estilo.
Todo sale en un único .ass que el filtro `ass` de ffmpeg (libass) quema; así
no dependemos de rutas de fuentes para `drawtext` en Windows.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Iterable

_WORD = re.compile(r"\S+")


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


def _esc(text: str) -> str:
    return text.replace("\\", "\\\\").replace("{", "(").replace("}", ")").replace("\n", "\\N")


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


_POS_ALIGN = {"bottom": 2, "center": 5, "middle": 5, "top": 8}


def build_ass(*, width: int, height: int, captions: dict[str, Any] | None, on_screen_text: dict[str, Any] | None,
              duration: float) -> tuple[str, dict[str, Any]]:
    """Devuelve (contenido .ass, resumen de lo aplicado)."""
    cap = captions or {}
    ost = on_screen_text or {}
    style = cap.get("style") or {}
    font = style.get("font", "Arial")
    size = int(style.get("size", round(height * 0.038)))
    cap_align = _POS_ALIGN.get(style.get("position", "bottom"), 2)
    cap_margin_v = int(style.get("margin_v", round(height * 0.22)))
    ost_size = int((ost.get("style") or {}).get("size", round(height * 0.034)))
    header = (
        "[Script Info]\nScriptType: v4.00+\n"
        f"PlayResX: {width}\nPlayResY: {height}\nWrapStyle: 0\nScaledBorderAndShadow: yes\n\n"
        "[V4+ Styles]\n"
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, "
        "Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, "
        "MarginR, MarginV, Encoding\n"
        f"Style: Caption,{font},{size},&H00FFFFFF,&H00FFFFFF,&H00000000,&H64000000,-1,0,0,0,100,100,0,0,1,6,2,"
        f"{cap_align},60,60,{cap_margin_v},1\n"
        f"Style: Overlay,{font},{ost_size},&H00FFFFFF,&H00FFFFFF,&H00000000,&HA0000000,-1,0,0,0,100,100,0,0,3,4,0,"
        f"8,60,60,{round(height * 0.12)},1\n\n"
        "[Events]\nFormat: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
    )
    lines: list[str] = []
    summary: dict[str, Any] = {"captions": {"applied": False, "events": 0},
                               "on_screen_text": {"applied": False, "texts": []}}
    if cap.get("enabled") and cap.get("segments"):
        evs = caption_events(cap["segments"], max_words=int(style.get("max_words", 4)), duration=duration,
                             dictionary=cap.get("brand_dictionary") or [])
        for a, b, t in evs:
            lines.append(f"Dialogue: 0,{_ts(a)},{_ts(b)},Caption,,0,0,0,,{_esc(t)}")
        summary["captions"] = {"applied": bool(evs), "events": len(evs),
                               "brand_dictionary": list(cap.get("brand_dictionary") or []),
                               "text": " ".join(t for _, _, t in evs)[:4000]}
    if ost.get("enabled"):
        texts = []
        for it in ost.get("items") or []:
            t = str(it.get("text") or "").strip()
            if not t:
                continue
            a = float(it.get("start", 0.0))
            b = float(it.get("end") if it.get("end") is not None else duration)
            b = min(b, duration)
            if b <= a:
                continue
            align = _POS_ALIGN.get(it.get("position", "top"), 8)
            lines.append(f"Dialogue: 1,{_ts(a)},{_ts(b)},Overlay,,0,0,0,,{{\\an{align}}}{_esc(t)}")
            texts.append({"text": t, "start": a, "end": b})
        summary["on_screen_text"] = {"applied": bool(texts), "texts": texts}
    return header + "\n".join(lines) + "\n", summary


def write_ass(path: Path, **kw: Any) -> dict[str, Any]:
    content, summary = build_ass(**kw)
    path.write_text(content, encoding="utf-8")
    return summary
