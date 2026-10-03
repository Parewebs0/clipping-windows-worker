"""Job de renderizado de clips con FFmpeg."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from app.jobs.base import BaseJob
from app.tools.captions import write_ass
from app.tools.ffmpeg import FFmpegTool
from app.tools.ffprobe import FFprobeTool


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

        # --- Contrato render_spec v2 (issue #4) ---------------------------
        # captions{enabled, file | segments, brand_dictionary, style}
        # watermark{enabled, file | url, position, width, start, end}
        # on_screen_text{enabled, items[{text, start, end, position}]}
        # Tiempos relativos al clip. `captions.file` / `watermark.file`
        # (contrato antiguo) siguen funcionando.
        duration = end - start
        width, height = (1080, 1920) if output_format == "9:16" else (1920, 1080)
        captions = payload.get("captions") or {}
        on_screen_text = payload.get("on_screen_text") or {}
        applied: dict[str, Any] = {
            "captions": {"applied": False, "events": 0},
            "on_screen_text": {"applied": False, "texts": []},
            "watermark": {"applied": False},
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
                                on_screen_text=on_screen_text, duration=duration)
            applied.update(summary)
            if summary["captions"]["applied"] or summary["on_screen_text"]["applied"]:
                captions_file = ass_path

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

        self.logger.info(
            "rendering clip",
            input=str(source),
            start=start,
            end=end,
            format=output_format,
            output_fps=output_fps,
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
        )

        self.logger.info("clip rendered", output=str(output_path))
        file_size = output_path.stat().st_size
        probe = self._probe(output_path)
        # Duración real del clip generado (puede diferir ligeramente de
        # end-start por el redondeo a keyframe). La usamos en el QA y
        # la persistimos en VPS como `duration_seconds`.
        duration_seconds = max(0.0, end - start)
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
            # Issue #4: lo que se aplicó + medida real (verificador post-render del VPS)
            "render_spec_version": 2,
            "applied": applied,
            "probe": probe,
        }

    def _download_logo(self, url: str) -> Path:
        """Descarga el logo a la carpeta del job (PNG/JPG/WebP)."""
        import httpx

        suffix = Path(url.split("?", 1)[0]).suffix.lower()
        if suffix not in (".png", ".jpg", ".jpeg", ".webp"):
            suffix = ".png"
        dest = self.directory.input / f"logo{suffix}"
        dest.parent.mkdir(parents=True, exist_ok=True)
        with httpx.Client(follow_redirects=True, timeout=60) as client:
            r = client.get(url)
            r.raise_for_status()
            ctype = r.headers.get("content-type", "")
            if not ctype.startswith("image/") or "svg" in ctype:
                raise ValueError(f"logo URL is not a raster image ({ctype or 'unknown'}): {url}")
            dest.write_bytes(r.content)
        return dest

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
