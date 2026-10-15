"""Color emoji as PNG. libass does not paint color emoji, so stickers are overlays.

Bundled samples live in app/assets/emoji (Twemoji, CC-BY 4.0). Missing
codepoints are downloaded once into the job cache from the same Twemoji set.
"""

from __future__ import annotations

from pathlib import Path

_ASSETS = Path(__file__).resolve().parent.parent / "assets" / "emoji"
_TWEMOJI = "https://cdn.jsdelivr.net/gh/jdecked/twemoji@15.1.0/assets/72x72/{name}.png"
_PNG = b"\x89PNG\r\n\x1a\n"


def codepoint_name(emoji: str) -> str:
    """Twemoji filename: lowercase hex, FE0F stripped, ZWJ kept as hyphen."""
    parts: list[str] = []
    for ch in str(emoji or ""):
        cp = ord(ch)
        if cp in (0xFE0F, 0xFE0E):
            continue
        parts.append(f"{cp:x}")
    return "-".join(parts)


def resolve_emoji_png(emoji: str, cache_dir: Path) -> Path | None:
    """Return a local PNG for `emoji`, or None if it cannot be resolved."""
    name = codepoint_name(emoji)
    if not name:
        return None
    bundled = _ASSETS / f"{name}.png"
    if bundled.is_file() and bundled.read_bytes()[:8] == _PNG:
        return bundled
    dest = Path(cache_dir) / f"{name}.png"
    if dest.is_file() and dest.stat().st_size > 32 and dest.read_bytes()[:8] == _PNG:
        return dest
    try:
        import httpx

        response = httpx.get(_TWEMOJI.format(name=name), timeout=20, follow_redirects=True)
    except Exception:
        return None
    if response.status_code != 200 or response.content[:8] != _PNG:
        return None
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(response.content)
    return dest
