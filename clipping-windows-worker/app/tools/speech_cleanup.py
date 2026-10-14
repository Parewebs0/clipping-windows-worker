"""Speech cleanup (MOLI-51): fillers, long pauses, and bad takes.

Word times are WhisperX times relative to the clip. The default config
removes nothing, so the renderer keeps the single-trim path.

Cuts are planned in milliseconds, then turned into keep ranges. The
renderer joins those ranges with trim/atrim, concat, and a 15 ms audio
fade (inside the 10–20 ms window) so the joins do not click. Pauses keep
an 80 ms pad on each speech edge (inside the 60–100 ms window).
"""

from __future__ import annotations

import re
from typing import Any, Iterable, Sequence

# 80 ms sits inside the 60–100 ms pad that must stay against each word.
PAUSE_MARGIN_S = 0.080
# 15 ms sits inside the 10–20 ms audio fade.
FADE_S = 0.015
# Word bigram Jaccard. High enough that a one-word paraphrase survives.
BAD_TAKE_JACCARD = 0.60
BAD_TAKE_MIN_TOKENS = 4
MIN_CUT_S = 0.040
MIN_KEEP_S = 0.020
MIN_OUTPUT_S = 0.400

_SENTENCE_END = re.compile(r"[.!?…]+[\"'»”’)]*$")
_TOKEN = re.compile(r"[0-9a-záéíóúüñ]+", re.IGNORECASE)

# Illustrative list from the issue, plus the closest hesitation variants.
# "este" is included because the spec names it; campaigns can replace the list.
FILLERS_BY_LANGUAGE: dict[str, list[str]] = {
    "es": [
        "o sea", "osea",
        "eh", "ehh", "ehm", "em", "um", "uh", "uhm", "ah", "ahm", "mmm", "mm", "hmm",
        "este",
    ],
    "en": [
        "you know", "i mean",
        "um", "uh", "uhm", "er", "ah", "eh", "hmm", "mm", "mmm",
    ],
}

DEFAULT_CLEANUP: dict[str, Any] = {
    "fillers": False,
    "max_pause_ms": None,
    "bad_takes": False,
    "filler_list": None,
}

_KIND_PRIORITY = {"bad_take": 3, "filler": 2, "pause": 1}


def _as_bool(value: Any, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and not isinstance(value, bool) and value in (0, 1):
        return bool(int(value))
    if isinstance(value, str):
        token = value.strip().lower()
        if token in {"1", "true", "yes", "on"}:
            return True
        if token in {"0", "false", "no", "off", ""}:
            return False
    return default


def _as_pause_ms(value: Any) -> int | None:
    if value is None or value is False:
        return None
    if isinstance(value, str) and not value.strip():
        return None
    try:
        number = int(float(value))
    except (TypeError, ValueError):
        return None
    if number <= 0:
        return None
    return number


def _as_filler_list(value: Any) -> list[str] | dict[str, list[str]] | None:
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip()
        return [text] if text else None
    if isinstance(value, dict):
        out: dict[str, list[str]] = {}
        for key, items in value.items():
            phrases = _phrase_strings(items)
            if phrases:
                out[str(key).lower().split("-")[0]] = phrases
        return out or None
    if isinstance(value, (list, tuple)):
        phrases = _phrase_strings(value)
        return phrases or None
    return None


def _phrase_strings(items: Any) -> list[str]:
    if isinstance(items, str):
        text = items.strip()
        return [text] if text else []
    out: list[str] = []
    for item in items or []:
        text = str(item).strip()
        if text:
            out.append(text)
    return out


def normalize_cleanup(raw: dict[str, Any] | None) -> dict[str, Any]:
    """Return the four-key cleanup block. Unknown values fall back to off."""
    data = raw or {}
    return {
        "fillers": _as_bool(data.get("fillers"), False),
        "max_pause_ms": _as_pause_ms(data.get("max_pause_ms")),
        "bad_takes": _as_bool(data.get("bad_takes"), False),
        "filler_list": _as_filler_list(data.get("filler_list")),
    }


def cleanup_options(payload: dict[str, Any] | None) -> dict[str, Any]:
    """Read `cleanup` from a render payload. Missing block = current behaviour."""
    if not isinstance(payload, dict):
        return dict(DEFAULT_CLEANUP)
    raw = payload.get("cleanup")
    if not isinstance(raw, dict):
        return dict(DEFAULT_CLEANUP)
    return normalize_cleanup(raw)


def _language_code(language: str | None) -> str:
    return (language or "").strip().lower().split("-")[0].split("_")[0]


def resolve_filler_phrases(options: dict[str, Any], language: str | None) -> list[str]:
    custom = options.get("filler_list")
    lang = _language_code(language)
    if isinstance(custom, dict):
        if lang and isinstance(custom.get(lang), list):
            return list(custom[lang])
        for fallback in ("es", "en"):
            if isinstance(custom.get(fallback), list):
                return list(custom[fallback])
        merged: list[str] = []
        for items in custom.values():
            if isinstance(items, list):
                merged.extend(items)
        return merged
    if isinstance(custom, list) and custom:
        return list(custom)
    if lang in FILLERS_BY_LANGUAGE:
        return list(FILLERS_BY_LANGUAGE[lang])
    return list(FILLERS_BY_LANGUAGE["es"]) + list(FILLERS_BY_LANGUAGE["en"])


def _norm_token(text: str) -> str:
    return "".join(_TOKEN.findall(str(text).lower()))


def _parse_phrases(items: Iterable[str]) -> list[list[str]]:
    parsed: list[list[str]] = []
    seen: set[tuple[str, ...]] = set()
    for item in items:
        tokens = [_norm_token(part) for part in str(item).split()]
        tokens = [token for token in tokens if token]
        key = tuple(tokens)
        if not key or key in seen:
            continue
        seen.add(key)
        parsed.append(tokens)
    parsed.sort(key=len, reverse=True)
    return parsed


def _ms(value: float) -> int:
    return int(round(float(value) * 1000))


def _sec(value_ms: int) -> float:
    return round(value_ms / 1000.0, 3)


def _timed_words(segments: Sequence[dict[str, Any]] | None) -> list[dict[str, Any]]:
    words: list[dict[str, Any]] = []
    for seg in segments or []:
        if not isinstance(seg, dict):
            continue
        for raw in seg.get("words") or []:
            if not isinstance(raw, dict):
                continue
            text = str(raw.get("word") or raw.get("text") or "").strip()
            if not text:
                continue
            try:
                start = float(raw["start"])
                end = float(raw["end"])
            except (KeyError, TypeError, ValueError):
                continue
            if end <= start:
                continue
            words.append({"start": start, "end": end, "word": text})
    words.sort(key=lambda word: (word["start"], word["end"]))
    return words


def _phrases(segments: Sequence[dict[str, Any]] | None) -> list[list[dict[str, Any]]]:
    phrases: list[list[dict[str, Any]]] = []
    buffer: list[dict[str, Any]] = []

    def flush() -> None:
        if buffer:
            phrases.append(list(buffer))
            buffer.clear()

    def seg_start(seg: dict[str, Any]) -> float:
        try:
            return float(seg.get("start") or 0.0)
        except (TypeError, ValueError):
            return 0.0

    ordered = sorted((seg for seg in segments or [] if isinstance(seg, dict)), key=seg_start)
    for seg in ordered:
        if not isinstance(seg, dict):
            continue
        words = _timed_words([seg])
        if not words:
            continue
        for word in words:
            buffer.append(word)
            if _SENTENCE_END.search(word["word"].strip()):
                flush()
        flush()
    flush()
    return phrases


def _strip_fillers(tokens: list[str], phrases: Sequence[Sequence[str]]) -> list[str]:
    out: list[str] = []
    index = 0
    while index < len(tokens):
        matched = 0
        for phrase in phrases:
            size = len(phrase)
            if size and tokens[index:index + size] == list(phrase):
                matched = size
                break
        if matched:
            index += matched
        else:
            out.append(tokens[index])
            index += 1
    return out


def _jaccard(left: Sequence[str], right: Sequence[str]) -> float:
    if len(left) < BAD_TAKE_MIN_TOKENS or len(right) < BAD_TAKE_MIN_TOKENS:
        return 0.0
    grams_left = set(zip(left, left[1:]))
    grams_right = set(zip(right, right[1:]))
    if not grams_left or not grams_right:
        return 0.0
    union = grams_left | grams_right
    return len(grams_left & grams_right) / len(union)


def _phrase_text(words: Sequence[dict[str, Any]]) -> str:
    return " ".join(word["word"] for word in words).strip()


def _empty_report(*, enabled: bool, duration: float, reason: str | None = None) -> dict[str, Any]:
    return {
        "enabled": enabled,
        "applied": False,
        "removed_seconds": 0.0,
        "by_kind": {"fillers": 0.0, "pauses": 0.0, "bad_takes": 0.0},
        "cuts": [],
        "keeps": [],
        "output_duration": round(max(0.0, duration), 3),
        "reason": reason,
    }


def _covered(start_ms: int, end_ms: int, intervals: Sequence[tuple[int, int]]) -> bool:
    mid = (start_ms + end_ms) // 2
    return any(a <= mid < b for a, b in intervals)


def _attribute(intervals: Sequence[tuple[int, int, str, str]], duration_ms: int) -> list[tuple[int, int, str, str]]:
    """Split overlapping cuts so each millisecond belongs to one kind."""
    points = {0, duration_ms}
    for start, end, _kind, _text in intervals:
        points.add(start)
        points.add(end)
    ordered = [point for point in sorted(points) if 0 <= point <= duration_ms]
    slices: list[tuple[int, int, str, str]] = []
    for left, right in zip(ordered, ordered[1:]):
        if right <= left:
            continue
        mid = (left + right) // 2
        covering = [item for item in intervals if item[0] <= mid < item[1]]
        if not covering:
            continue
        kind, text = max(
            ((item[2], item[3]) for item in covering),
            key=lambda item: _KIND_PRIORITY.get(item[0], 0),
        )
        slices.append((left, right, kind, text))
    merged: list[tuple[int, int, str, str]] = []
    for start, end, kind, text in slices:
        if merged and merged[-1][2] == kind and merged[-1][1] == start and (kind == "pause" or merged[-1][3] == text):
            prev = merged[-1]
            merged[-1] = (prev[0], end, kind, prev[3] or text)
        else:
            merged.append((start, end, kind, text))
    return merged


def plan_cleanup(
    segments: Sequence[dict[str, Any]] | None,
    *,
    duration: float,
    options: dict[str, Any] | None,
    language: str | None = None,
) -> dict[str, Any]:
    """Plan keep ranges inside the clip. `applied` is false when nothing changes."""
    cfg = normalize_cleanup(options)
    duration = max(0.0, float(duration))
    enabled = bool(cfg["fillers"] or cfg["bad_takes"] or cfg["max_pause_ms"])
    if not enabled or duration <= 0:
        return _empty_report(enabled=enabled, duration=duration, reason=None if not enabled else "empty_clip")

    words = _timed_words(segments)
    if not words:
        return _empty_report(enabled=True, duration=duration, reason="no_word_timings")

    duration_ms = _ms(duration)
    filler_phrases = _parse_phrases(resolve_filler_phrases(cfg, language))
    logical: list[tuple[int, int, str, str]] = []

    bad_intervals: list[tuple[int, int]] = []
    if cfg["bad_takes"]:
        phrases = _phrases(segments)
        stripped = [
            _strip_fillers([_norm_token(word["word"]) for word in phrase], filler_phrases)
            for phrase in phrases
        ]
        drop = [False] * len(phrases)
        for index in range(len(phrases) - 1):
            if _jaccard(stripped[index], stripped[index + 1]) >= BAD_TAKE_JACCARD:
                drop[index] = True
        margin_ms = _ms(PAUSE_MARGIN_S)
        for index, phrase in enumerate(phrases):
            if not drop[index]:
                continue
            # A run of retakes is removed up to the take we keep, leaving the
            # same 80 ms pad used for pauses. Leading silence before the run
            # goes with it when nothing worth keeping comes first.
            if index == 0 or all(drop[:index]):
                start_ms = 0
            else:
                start_ms = max(0, _ms(phrase[0]["start"]))
            if index + 1 < len(phrases):
                next_start = max(0, _ms(phrases[index + 1][0]["start"]))
                if drop[index + 1]:
                    end_ms = min(duration_ms, next_start)
                else:
                    end_ms = min(duration_ms, max(_ms(phrase[-1]["end"]), next_start - margin_ms))
            else:
                end_ms = duration_ms
            if end_ms - start_ms < _ms(MIN_CUT_S):
                continue
            text = _phrase_text(phrase)[:180]
            logical.append((start_ms, end_ms, "bad_take", text))
            bad_intervals.append((start_ms, end_ms))

    if cfg["fillers"] and filler_phrases:
        index = 0
        while index < len(words):
            start_ms = _ms(words[index]["start"])
            end_ms = _ms(words[index]["end"])
            if _covered(start_ms, end_ms, bad_intervals):
                index += 1
                continue
            matched = 0
            tokens = [_norm_token(word["word"]) for word in words[index:]]
            for phrase in filler_phrases:
                size = len(phrase)
                if size and tokens[:size] == phrase:
                    matched = size
                    break
            if not matched:
                index += 1
                continue
            group = words[index:index + matched]
            cut_start = max(0, _ms(group[0]["start"]))
            cut_end = min(duration_ms, _ms(group[-1]["end"]))
            if cut_end - cut_start >= _ms(MIN_CUT_S):
                logical.append((cut_start, cut_end, "filler", _phrase_text(group)[:80]))
            index += matched

    removed_words = [(item[0], item[1]) for item in logical]
    if cfg["max_pause_ms"]:
        surviving = [
            word for word in words
            if not _covered(_ms(word["start"]), _ms(word["end"]), removed_words)
        ]
        margin_ms = _ms(PAUSE_MARGIN_S)
        max_pause = int(cfg["max_pause_ms"])
        edges: list[tuple[int, int]] = []
        if surviving:
            edges.append((0, max(0, _ms(surviving[0]["start"]))))
            for prev, nxt in zip(surviving, surviving[1:]):
                edges.append((max(0, _ms(prev["end"])), min(duration_ms, _ms(nxt["start"]))))
            edges.append((min(duration_ms, _ms(surviving[-1]["end"])), duration_ms))
        else:
            edges.append((0, duration_ms))
        for left, right in edges:
            if right - left <= max_pause:
                continue
            left_pad = 0 if left <= 0 else margin_ms
            right_pad = 0 if right >= duration_ms else margin_ms
            cut_start = left + left_pad
            cut_end = right - right_pad
            if cut_end - cut_start < _ms(MIN_CUT_S):
                continue
            logical.append((cut_start, cut_end, "pause", ""))

    attributed = _attribute(logical, duration_ms)
    removed_ms = sum(end - start for start, end, _kind, _text in attributed)
    output_ms = duration_ms - removed_ms
    if removed_ms < _ms(MIN_CUT_S):
        report = _empty_report(enabled=True, duration=duration, reason="nothing_to_cut")
        return report
    if output_ms < _ms(MIN_OUTPUT_S):
        report = _empty_report(enabled=True, duration=duration, reason="would_remove_too_much")
        return report

    removed: list[tuple[int, int]] = []
    for start, end, _kind, _text in attributed:
        if removed and removed[-1][1] >= start:
            removed[-1] = (removed[-1][0], max(removed[-1][1], end))
        else:
            removed.append((start, end))

    keeps: list[tuple[float, float]] = []
    cursor = 0
    for start, end in removed:
        if start - cursor >= _ms(MIN_KEEP_S):
            keeps.append((_sec(cursor), _sec(start)))
        cursor = max(cursor, end)
    if duration_ms - cursor >= _ms(MIN_KEEP_S):
        keeps.append((_sec(cursor), _sec(duration_ms)))
    if not keeps:
        return _empty_report(enabled=True, duration=duration, reason="would_remove_too_much")

    kind_ms = {"fillers": 0, "pauses": 0, "bad_takes": 0}
    cuts: list[dict[str, Any]] = []
    for start, end, kind, text in attributed:
        key = {"filler": "fillers", "pause": "pauses", "bad_take": "bad_takes"}[kind]
        kind_ms[key] += end - start
        cut = {"kind": kind, "start": _sec(start), "end": _sec(end), "seconds": _sec(end - start)}
        if text:
            cut["text"] = text
        cuts.append(cut)

    output_duration = round(sum(end - start for start, end in keeps), 3)
    return {
        "enabled": True,
        "applied": True,
        "removed_seconds": _sec(removed_ms),
        "by_kind": {key: _sec(value) for key, value in kind_ms.items()},
        "cuts": cuts,
        "keeps": [{"start": start, "end": end} for start, end in keeps],
        "output_duration": output_duration,
        "reason": None,
    }


def keep_ranges(plan: dict[str, Any]) -> list[tuple[float, float]] | None:
    """Clip-relative ranges for ffmpeg. None keeps the single-trim render."""
    if not plan.get("applied"):
        return None
    ranges = [(float(item["start"]), float(item["end"])) for item in plan.get("keeps") or []]
    ranges = [(start, end) for start, end in ranges if end - start >= MIN_CUT_S]
    return ranges or None


def remap_time(time_s: float, keeps: Sequence[tuple[float, float]]) -> float:
    """Map a clip-relative source time onto the concatenated timeline."""
    acc = 0.0
    for start, end in keeps:
        if time_s < start:
            return round(acc, 3)
        if time_s <= end:
            return round(acc + (time_s - start), 3)
        acc += end - start
    return round(acc, 3)


def _overlap(start: float, end: float, keeps: Sequence[tuple[float, float]]) -> float:
    total = 0.0
    for keep_start, keep_end in keeps:
        lo = max(start, keep_start)
        hi = min(end, keep_end)
        if hi > lo:
            total += hi - lo
    return total


def remap_segments(segments: Sequence[dict[str, Any]] | None, keeps: Sequence[tuple[float, float]]) -> list[dict[str, Any]]:
    """Drop words that were cut and shift the rest onto the new timeline."""
    out: list[dict[str, Any]] = []
    for seg in segments or []:
        if not isinstance(seg, dict):
            continue
        raw_words = [word for word in (seg.get("words") or []) if isinstance(word, dict)]
        timed = _timed_words([{"words": raw_words}]) if raw_words else []
        if timed:
            new_words: list[dict[str, Any]] = []
            for word in timed:
                span = float(word["end"]) - float(word["start"])
                if _overlap(word["start"], word["end"], keeps) < max(0.02, span * 0.5):
                    continue
                start = remap_time(word["start"], keeps)
                end = remap_time(word["end"], keeps)
                if end - start < 0.02:
                    continue
                mapped = {"start": start, "end": end, "word": word["word"]}
                new_words.append(mapped)
            if not new_words:
                continue
            copied = dict(seg)
            copied["start"] = new_words[0]["start"]
            copied["end"] = new_words[-1]["end"]
            copied["text"] = " ".join(word["word"] for word in new_words)
            copied["words"] = new_words
            out.append(copied)
            continue
        try:
            start = float(seg.get("start", 0.0))
            end = float(seg.get("end", 0.0))
        except (TypeError, ValueError):
            continue
        if end <= start or _overlap(start, end, keeps) < 0.05:
            continue
        copied = dict(seg)
        copied["start"] = remap_time(start, keeps)
        copied["end"] = remap_time(end, keeps)
        if copied["end"] - copied["start"] < 0.05:
            continue
        out.append(copied)
    return out


def remap_items(
    items: Sequence[dict[str, Any]] | None,
    keeps: Sequence[tuple[float, float]],
    *,
    source_duration: float,
) -> list[dict[str, Any]]:
    """Shift on-screen text (and any other start/end item) onto the new timeline."""
    out: list[dict[str, Any]] = []
    for item in items or []:
        if not isinstance(item, dict):
            continue
        try:
            start = float(item.get("start", 0.0))
        except (TypeError, ValueError):
            start = 0.0
        if item.get("end") is None:
            end = source_duration
        else:
            try:
                end = float(item["end"])
            except (TypeError, ValueError):
                end = source_duration
        mapped_start = remap_time(start, keeps)
        mapped_end = remap_time(end, keeps)
        if mapped_end - mapped_start < 0.05:
            continue
        copied = dict(item)
        copied["start"] = mapped_start
        copied["end"] = mapped_end
        out.append(copied)
    return out


def remap_watermark(watermark: dict[str, Any], keeps: Sequence[tuple[float, float]], *, source_duration: float) -> dict[str, Any]:
    """Shift a finite logo window. A full-clip window (both edges null) stays full-clip."""
    if not watermark or (watermark.get("start") is None and watermark.get("end") is None):
        return watermark
    start = 0.0 if watermark.get("start") is None else float(watermark["start"])
    end = source_duration if watermark.get("end") is None else float(watermark["end"])
    copied = dict(watermark)
    copied["start"] = remap_time(start, keeps)
    copied["end"] = remap_time(end, keeps)
    return copied
