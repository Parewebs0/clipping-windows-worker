"""Job de renderizado de clips con FFmpeg."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from app.jobs.base import BaseJob

RENDER_SPEC_VERSION = 3
# MOLI-48: same keys as VPS `app/services/rules/render_spec.py`.
# The worker does not clip-decide; these travel on the payload for one contract.
DEFAULT_DECIDER = {
    "chunk_chars": 4000,
    "candidates_per_chunk": 3,
    "top_n": 8,
    "overlap": 0.10,
}


def decider_options(payload: dict[str, Any] | None) -> dict[str, Any]:
    """Read `decider.*` from the render payload. Missing keys keep the VPS defaults."""
    raw = (payload or {}).get("decider") if isinstance(payload, dict) else None
    raw = raw if isinstance(raw, dict) else {}
    out = dict(DEFAULT_DECIDER)
    for key, default in DEFAULT_DECIDER.items():
        if key not in raw:
            continue
        try:
            out[key] = type(default)(raw[key])
        except (TypeError, ValueError):
            continue
    return out


def logo_request(url: str, *, api_base_url: str, api_token: str) -> tuple[str, dict[str, str]]:
    """Join a relative worker path and attach the token only for our /worker/ routes.

    External http(s) URLs stay unauthenticated (#55).
    """
    base = (api_base_url or "").rstrip("/")
    full = f"{base}{url}" if url.startswith("/") and base else url
    headers: dict[str, str] = {}
    if base and api_token and full.startswith(base + "/worker/"):
        headers["Authorization"] = f"Bearer {api_token}"
    return full, headers
from app.tools.captions import write_ass
from app.tools.ffmpeg import FFmpegTool
from app.tools.ffprobe import FFprobeTool

BRAND_MEDIA_SUFFIXES = (".png", ".jpg", ".jpeg", ".webp", ".mp4", ".webm", ".mov")
# Worker routes for intro/outro have no suffix (`/branding/intro`). The
# VPS sets Content-Type from the stored file (video/mp4, image/png, …).
_CONTENT_TYPE_SUFFIX = {
    "image/png": ".png",
    "image/jpeg": ".jpg",
    "image/jpg": ".jpg",
    "image/webp": ".webp",
    "video/mp4": ".mp4",
    "video/webm": ".webm",
    "video/quicktime": ".mov",
    "font/ttf": ".ttf",
    "font/otf": ".otf",
    "application/x-font-ttf": ".ttf",
    "application/x-font-otf": ".otf",
    "application/font-sfnt": ".ttf",
    "application/vnd.ms-opentype": ".otf",
}


def suffix_for_content_type(content_type: str) -> str:
    """File suffix for a Content-Type, or "" when the type is not a brand asset."""
    media = (content_type or "").split(";", 1)[0].strip().lower()
    return _CONTENT_TYPE_SUFFIX.get(media, "")


class RenderJob(BaseJob):
    """Recorta, reencuadra, añade subtítulos y watermark, y codifica."""

    type = "render"

    def execute(self) -> dict[str, Any]:
        payload = self.job.payload
        # Aceptar `input_video` (canónico Worker) o `video` (contrato VPS / OpenClaw).
        # Priorizar `input_video` si llega.
        input_video = payload.get("input_video") or payload.get("video")
        # Aceptamos tanto `start_time`/`end_time` (canónico) como `start`/`end`
        # (legacy) por compatibilidad con jobs antiguos.
        start = float(payload.get("start_time", payload.get("start", 0.0)))
        end = float(payload.get("end_time", payload.get("end", 0.0)))
        # Aceptamos tanto `format` (canónico VPS) como `output_format` (legacy)
        output_format = payload.get("format") or payload.get("output_format", "9:16")

        if not input_video:
            raise ValueError("payload.input_video or payload.video is required")
        if end <= start:
            raise ValueError(f"end ({end}) must be greater than start ({start})")

        source = self._resolve_input(input_video)
        if not source.exists():
            raise FileNotFoundError(f"Input video not found: {source}")

        # --- Contrato render_spec v3 (v2 + loudnorm) -----------------------
        # captions{enabled, file | segments, brand_dictionary, style}
        # watermark{enabled, file | url, position, width, start, end}
        # on_screen_text{enabled, items[{text, start, end, position}]}
        # audio{original_only, normalize, target_lufs, true_peak_db}
        # Tiempos relativos al clip. `captions.file` / `watermark.file`
        # (contrato antiguo) siguen funcionando. `audio.original_only`
        # deja la pista sin loudnorm.
        duration = end - start
        width, height = (1080, 1920) if output_format == "9:16" else (1920, 1080)
        captions = payload.get("captions") or {}
        on_screen_text = payload.get("on_screen_text") or {}
        branding = payload.get("branding") if isinstance(payload.get("branding"), dict) else {}
        applied: dict[str, Any] = {
            "captions": {"applied": False, "events": 0},
            "on_screen_text": {"applied": False, "texts": []},
            "watermark": {"applied": False},
            "branding": {"applied": False},
        }
        captions_file = None
        wants_generated = (captions.get("enabled") and captions.get("segments")) or on_screen_text.get("enabled")
        if captions.get("enabled") and captions.get("file") and not captions.get("segments"):
            captions_file = self._resolve_input(captions["file"])
            applied["captions"] = {"applied": True, "events": None, "file": str(captions_file)}
            if on_screen_text.get("enabled"):
                self.logger.warning("on_screen_text ignored: captions.file given (send segments instead)")
        elif wants_generated:
            ass_path = self.directory.output / "overlay.ass"
            summary = write_ass(ass_path, width=width, height=height, captions=captions,
                                on_screen_text=on_screen_text, duration=duration, branding=branding)
            applied.update(summary)
            if summary["captions"]["applied"] or summary["on_screen_text"]["applied"]:
                captions_file = ass_path

        # Brand assets (MOLI-54): font for libass + intro/outro cards.
        fontsdir: Path | None = None
        brand_head: Path | None = None
        brand_tail: Path | None = None
        brand_meta = {"applied": False}
        if branding:
            font_url = branding.get("font_url") or ""
            if font_url:
                try:
                    font_path = self._download_brand(font_url, ("font", ".ttf"))
                    fontsdir = font_path.parent
                    # libass needs the exact family; keep the file for the fontsdir
                    # and record it for applied metadata
                    font_ref = font_path.name
                    brand_meta["font"] = font_ref
                    brand_meta["applied"] = True
                except Exception as e:  # noqa: BLE001
                    self.logger.warning("brand font download failed; default font kept", error=str(e))
            for side, key in (("intro", "intro"), ("outro", "outro")):
                piece = branding.get(key) or {}
                url = piece.get("url") or ""
                if not url:
                    continue
                try:
                    path = self._download_brand(url, (side,))
                    if side == "intro":
                        brand_head = path
                    else:
                        brand_tail = path
                    brand_meta.setdefault("cards", []).append(side)
                    brand_meta["applied"] = True
                except Exception as e:  # noqa: BLE001 - a broken asset never blocks the render
                    self.logger.warning("brand %s download failed; skipped", side, error=str(e))
            applied["branding"] = brand_meta

        # Watermark / logo
        watermark = dict(payload.get("watermark") or {})
        if watermark.get("enabled"):
            if watermark.get("file"):
                watermark["file"] = str(self._resolve_input(watermark["file"]))
            elif watermark.get("url"):
                watermark["file"] = str(self._download_logo(str(watermark["url"])))
            else:
                raise ValueError("watermark.enabled=true requires watermark.file or watermark.url")
            applied["watermark"] = {
                "applied": True,
                "position": watermark.get("position", "top_right"),
                "width": int(watermark.get("width", 180)),
                "start": watermark.get("start"),
                "end": watermark.get("end"),
                "source": watermark.get("url") or Path(watermark["file"]).name,
            }

        # Framerate de salida. Por defecto 30 fps para evitar que fuentes
        # nativos a 23.976/24/25/29.97 fallen la regla QA de ``min_fps``.
        # El VPS puede sobrescribirlo vía ``payload.fps`` (o ``output_fps``).
        output_fps = float(
            payload.get("fps")
            or payload.get("output_fps")
            or self.settings.render_output_fps
            or 30.0
        )

        # Nombre del clip: usar `candidate_id` si el VPS lo manda (para que
        # el QA luego pueda moverlo a `<clip_storage_root>/<campaign_id>/pending_upload/<candidate_id>.mp4`
        # sin renombrar). Fallback a `clip.mp4` por compatibilidad legacy.
        candidate_id = payload.get("candidate_id")
        clip_filename = f"{candidate_id}.mp4" if candidate_id else "clip.mp4"
        output_path = self.directory.output / clip_filename
        ffmpeg = FFmpegTool(self.settings)
        audio_payload = payload.get("audio") if isinstance(payload.get("audio"), dict) else {}
        original_only = bool(audio_payload.get("original_only"))
        has_audio = FFprobeTool(self.settings).get_audio_stream(source) is not None
        audio_meta, audio_filter = ffmpeg.prepare_loudnorm(
            source,
            start=start,
            duration=duration,
            original_only=original_only,
            has_audio=has_audio,
        )
        applied["audio"] = audio_meta

        self.logger.info(
            "rendering clip",
            input=str(source),
            start=start,
            end=end,
            format=output_format,
            output_fps=output_fps,
            decider=decider_options(payload),
        )
        ffmpeg.render_clip(
            input_path=source,
            output_path=output_path,
            start=start,
            end=end,
            output_format=output_format,
            captions_file=captions_file,
            watermark=watermark,
            output_fps=output_fps,
            audio_filter=audio_filter,
            fontsdir=fontsdir,
        )

        # Brand cards (MOLI-54): prepend/append intro/outro only when present.
        # Concat is a full re-encode of head/body/tail; body keeps its own
        # resolution and cards are normalised to it. This inflates duration —
        # the QA duration window must allow it (render payload `qa_duration`).
        if brand_head is not None or brand_tail is not None:
            cap_seconds = float(branding.get("max_card_seconds") or 10.0)
            intro_secs = float((branding.get("intro") or {}).get("max_seconds") or 1.5)
            outro_secs = float((branding.get("outro") or {}).get("max_seconds") or 1.5)
            head_cap = max(0.1, min(cap_seconds, intro_secs))
            tail_cap = max(0.1, min(cap_seconds, outro_secs))
            if brand_head is None:
                head_cap = 0.0
            if brand_tail is None:
                tail_cap = 0.0
            joined = output_path.with_name(f"_branded_{output_path.name}")
            try:
                ffmpeg.concat_clips(head=brand_head if head_cap else None,
                                    body=output_path,
                                    tail=brand_tail if tail_cap else None,
                                    output_path=joined,
                                    max_side_seconds=max(head_cap, tail_cap))
                joined.replace(output_path)
                duration_seconds = max(0.0, (end - start) + head_cap + tail_cap)
            except Exception as e:  # noqa: BLE001 - a failed concat must not drop the clip
                self.logger.warning(
                    "brand card concat failed; keeping clip without cards", error=str(e),
                )
                brand_meta["cards_error"] = str(e)[:500]
                if joined.exists():
                    joined.unlink()
                duration_seconds = max(0.0, end - start)
        else:
            duration_seconds = max(0.0, end - start)

        self.logger.info("clip rendered", output=str(output_path))
        file_size = output_path.stat().st_size
        probe = self._probe(output_path)
        applied["frame_checks"] = self._frame_checks(output_path, applied, payload, probe, watermark)
        # Duración real del clip generado (puede diferir ligeramente de
        # end-start por el redondeo a keyframe, o crecer con intro/outro).
        # La usamos en el QA y la persistimos en VPS como `duration_seconds`.
        return {
            # Canónico (VPS consume esto)
            "file_path": str(output_path),
            "file_size": file_size,
            "duration_seconds": duration_seconds,
            "format": output_format,
            # Aliases para retro-compatibilidad con jobs antiguos
            "output_path": str(output_path),
            "filename": output_path.name,
            "size": file_size,
            "output_format": output_format,
            # v3: applied.audio (loudnorm) + probe. v2 sigue siendo válido en el VPS.
            "render_spec_version": RENDER_SPEC_VERSION,
            "applied": applied,
            "probe": probe,
        }

    def _download_logo(self, url: str) -> Path:
        """Descarga el logo a la carpeta del job (PNG/JPG/WebP)."""
        import httpx

        full, headers = logo_request(
            url, api_base_url=self.settings.api_base_url, api_token=self.settings.api_token,
        )
        suffix = Path(urlparse(full).path).suffix.lower()
        if suffix not in (".png", ".jpg", ".jpeg", ".webp"):
            suffix = ".png"
        dest = self.directory.input / f"logo{suffix}"
        dest.parent.mkdir(parents=True, exist_ok=True)
        with httpx.Client(follow_redirects=True, timeout=60) as client:
            r = client.get(full, headers=headers)
            r.raise_for_status()
            ctype = r.headers.get("content-type", "")
            if not ctype.startswith("image/") or "svg" in ctype:
                raise ValueError(f"logo URL is not a raster image ({ctype or 'unknown'}): {url}")
            dest.write_bytes(r.content)
        return dest

    def _download_brand(self, url: str, name_hint: tuple[str, ...]) -> Path:
        """Descarga un asset de marca (fuente TTF/OTF o tarjeta intro/outro).

        Intro/outro se sirven en `/worker/campaigns/{id}/branding/{side}` sin
        extensión. Sin sufijo el fichero caía en `.bin` y un MP4 se concatenaba
        como imagen fija. La extensión sale de la URL si es un tipo conocido y,
        si no, del Content-Type (`video/mp4`, `image/png`, …).
        """
        import httpx

        full, headers = logo_request(
            url, api_base_url=self.settings.api_base_url, api_token=self.settings.api_token,
        )
        url_suffix = Path(urlparse(full).path).suffix.lower()
        if url_suffix not in (".ttf", ".otf", ".png", ".jpg", ".jpeg", ".webp", ".mp4", ".webm", ".mov"):
            url_suffix = ""
        with httpx.Client(follow_redirects=True, timeout=120) as client:
            r = client.get(full, headers=headers)
            r.raise_for_status()
            data = r.content
            content_type = r.headers.get("content-type", "")
        if not data:
            raise ValueError(f"brand asset is empty: {url}")
        base = name_hint[0] if name_hint else "brand"
        ext = url_suffix or suffix_for_content_type(content_type)
        if not ext and len(name_hint) > 1:
            ext = name_hint[1]
        if not ext:
            ext = ".bin"
        dest = self.directory.input / f"{base}{ext}"
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)
        return dest

    def _frame_checks(self, video: Path, applied: dict, payload: dict, probe: dict, watermark: dict) -> dict[str, Any]:
        """Sample the rendered clip. Missing samples omit the bool so the VPS keeps metadata-only."""
        samples: list[dict] = []
        out: dict[str, Any] = {"samples": samples}
        try:
            duration = float(probe.get("duration") or 0)
        except (TypeError, ValueError):
            duration = 0.0
        if duration <= 0:
            return out
        vw, vh = int(probe.get("width") or 0), int(probe.get("height") or 0)
        if vw < 16 or vh < 16:
            return out
        bg_crop = f"200:100:40:{max(0, vh // 2)}"
        bg = self._yavg(video, min(1.0, duration / 2), bg_crop)
        if bg is None:
            return out
        samples.append({"what": "background", "t": round(min(1.0, duration / 2), 3), "yavg": round(bg, 2)})
        wm = applied.get("watermark") or {}
        if wm.get("applied"):
            t = _sample_time(wm.get("start"), wm.get("end"), duration)
            crop = self._logo_crop(video_w=vw, video_h=vh, logo=Path(str(watermark.get("file") or "")), watermark=watermark)
            y = self._yavg(video, t, crop) if crop else None
            if y is not None:
                out["logo_visible"] = y > bg + 20
                samples.append({"what": "logo", "t": round(t, 3), "yavg": round(y, 2), "crop": crop})
        cap = applied.get("captions") or {}
        if cap.get("applied"):
            segs = (payload.get("captions") or {}).get("segments") or []
            if segs:
                t = _sample_time(segs[0].get("start"), segs[0].get("end"), duration)
            else:
                t = min(1.0, max(0.05, duration / 2))
            band_h = min(200, vh)
            y0 = max(0, vh - 540) if vh > 540 else max(0, vh - band_h)
            crop = f"{vw}:{band_h}:0:{y0}"
            y = self._yavg(video, t, crop)
            if y is not None:
                out["captions_visible"] = y > bg + 1
                samples.append({"what": "captions", "t": round(t, 3), "yavg": round(y, 2), "crop": crop})
        return out

    def _logo_crop(self, *, video_w: int, video_h: int, logo: Path, watermark: dict) -> str | None:
        lw = int(watermark.get("width") or 180)
        iw, ih = self._image_size(logo)
        lh = max(1, int(round(ih * lw / iw))) if iw else lw
        mx = int(watermark.get("margin_x") or 40)
        my = int(watermark.get("margin_y") or 40)
        pos = watermark.get("position") or "top_right"
        if pos == "top_left":
            x, y = mx, my
        elif pos == "bottom_left":
            x, y = mx, video_h - lh - my
        elif pos == "bottom_right":
            x, y = video_w - lw - mx, video_h - lh - my
        elif pos == "center":
            x, y = (video_w - lw) // 2, (video_h - lh) // 2
        else:
            x, y = video_w - lw - mx, my
        if x < 0 or y < 0 or x + lw > video_w or y + lh > video_h:
            return None
        return f"{lw}:{lh}:{x}:{y}"

    def _image_size(self, path: Path) -> tuple[int, int]:
        if not path.is_file():
            return (0, 0)
        try:
            data = FFprobeTool(self.settings).probe(path)
        except Exception:  # noqa: BLE001
            return (0, 0)
        for st in data.get("streams", []):
            if st.get("width") and st.get("height"):
                return int(st["width"]), int(st["height"])
        return (0, 0)

    def _yavg(self, video: Path, t: float, crop: str) -> float | None:
        import subprocess

        ffmpeg = self.settings.ffmpeg_path or "ffmpeg"
        try:
            proc = subprocess.run(
                [ffmpeg, "-v", "info", "-ss", f"{t:.3f}", "-i", str(video), "-frames:v", "1",
                 "-vf", f"crop={crop},signalstats,metadata=print:key=lavfi.signalstats.YAVG",
                 "-f", "null", "-"],
                capture_output=True, text=True, timeout=60, check=False,
            )
        except Exception:  # noqa: BLE001
            return None
        for line in ((proc.stderr or "") + (proc.stdout or "")).splitlines():
            if "YAVG=" in line:
                try:
                    return float(line.split("YAVG=", 1)[1].split()[0])
                except ValueError:
                    return None
        return None

    def _probe(self, path: Path) -> dict[str, Any]:
        try:
            data = FFprobeTool(self.settings).probe(path)
        except Exception as e:  # noqa: BLE001
            self.logger.warning("ffprobe failed", error=str(e))
            return {}
        out: dict[str, Any] = {"has_audio": False}
        for st in data.get("streams", []):
            if st.get("codec_type") == "video" and "width" not in out:
                num, _, den = str(st.get("avg_frame_rate") or "0/1").partition("/")
                try:
                    fps = float(num) / float(den or 1)
                except (ValueError, ZeroDivisionError):
                    fps = 0.0
                out.update({"width": st.get("width"), "height": st.get("height"), "fps": round(fps, 3),
                            "codec": st.get("codec_name")})
            elif st.get("codec_type") == "audio":
                out["has_audio"] = True
        try:
            out["duration"] = round(float(data.get("format", {}).get("duration")), 3)
        except (TypeError, ValueError):
            pass
        return out

    def _resolve_input(self, path: str) -> Path:
        """Resuelve una ruta de entrada, absoluta o relativa al job."""
        p = Path(path)
        if p.is_absolute():
            return p
        candidates = [
            self.directory.input / p,
            self.directory.root / p,
            Path(self.settings.working_directory) / p,
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate
        return candidates[0]


def _sample_time(start, end, duration: float) -> float:
    try:
        s = 0.0 if start is None else float(start)
        e = duration if end is None else float(end)
    except (TypeError, ValueError):
        s, e = 0.0, duration
    if e < s:
        s, e = 0.0, duration
    mid = (s + e) / 2.0
    hi = max(0.05, duration - 0.05)
    return min(max(mid, 0.05), hi)
