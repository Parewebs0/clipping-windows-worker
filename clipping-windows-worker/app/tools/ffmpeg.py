"""Wrapper de FFmpeg para procesamiento y renderizado de vídeo."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from app.config import Settings
from app.utils.logging import get_logger
from app.utils.subprocess import run_command

logger = get_logger("ffmpeg")

# Posiciones de watermark soportadas.
WATERMARK_POSITIONS = {
    "top_left": "x=0:y=0",
    "top_right": "x=W-w:y=0",
    "bottom_left": "x=0:y=H-h",
    "bottom_right": "x=W-w:y=H-h",
    "center": "x=(W-w)/2:y=(H-h)/2",
}


class FFmpegTool:
    """Encapsula la detección y uso de ffmpeg."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._binary = settings.ffmpeg_path or "ffmpeg"

    @property
    def available(self) -> bool:
        try:
            run_command([self._binary, "-version"], timeout=10)
            return True
        except Exception:  # noqa: BLE001 - cualquier error = no disponible
            return False

    def extract_audio(
        self,
        input_path: str | Path,
        output_path: str | Path,
        *,
        sample_rate: int = 16000,
        channels: int = 1,
    ) -> Path:
        """Extrae el audio de un vídeo a WAV mono (ideal para WhisperX)."""
        output = Path(output_path)
        run_command(
            [
                self._binary,
                "-y",
                "-i",
                str(input_path),
                "-vn",
                "-acodec",
                "pcm_s16le",
                "-ar",
                str(sample_rate),
                "-ac",
                str(channels),
                str(output),
            ],
            timeout=600,
        )
        return output

    def render_clip(
        self,
        *,
        input_path: str | Path,
        output_path: str | Path,
        start: float,
        end: float,
        output_format: str = "9:16",
        captions_file: str | Path | None = None,
        watermark: dict[str, Any] | None = None,
        output_fps: float = 30.0,
    ) -> Path:
        """Renderiza un clip: trim + crop/scale + subtítulos + watermark + encode.

        ``output_format`` soporta "9:16" (1080x1920) y "16:9" (1920x1080).

        ``output_fps`` fuerza el framerate de salida (default 30.0) para que el
        QA no falle por fuentes nativos a 23.976/24/25/29.97. El ajuste se
        aplica como ``-r`` **después** del filtergraph (encoder/muxer), no
        como filtro ``fps=`` (eso duplicaría trabajo con ``scale``/``crop``).
        Para clips cortos de 30-60s este cambio sólo reescribe timestamps;
        si en el futuro hace falta interpolación real de frames, mover a
        ``fps=30:round=near`` dentro del filtergraph.
        """
        output = Path(output_path)
        duration = max(0.0, end - start)

        # --- Filtros de vídeo (crop/scale) ---
        vf_parts: list[str] = []
        if output_format == "9:16":
            vf_parts.append("crop=ih*9/16:ih")
            vf_parts.append("scale=1080:1920:force_original_aspect_ratio=decrease")
            vf_parts.append("pad=1080:1920:(ow-iw)/2:(oh-ih)/2")
        elif output_format == "16:9":
            vf_parts.append("crop=iw:iw*9/16")
            vf_parts.append("scale=1920:1080:force_original_aspect_ratio=decrease")
            vf_parts.append("pad=1920:1080:(ow-iw)/2:(oh-ih)/2")
        else:
            raise ValueError(f"Unsupported output_format: {output_format}")

        # --- Subtítulos (filtro de vídeo adicional) ---
        # Los filtros de FFmpeg (subtitles/ass) no aceptan rutas Windows con
        # drive colon (`C:\...`). Por ello se referencia el archivo por nombre
        # relativo y el proceso arranca con cwd en su carpeta.
        vfilter_cwd: str | None = None
        if captions_file:
            cf = Path(captions_file)
            vf_parts.append(self._subtitle_filter(cf.name))
            vfilter_cwd = str(cf.parent)

        # --- Construcción del comando ---
        # Nota: `-t` se coloca DESPUÉS de todos los inputs para que se aplique
        # al output (no al input). Si va antes de un `-i` extra (watermark),
        # ffmpeg lo interpreta como opción de input y no limita el clip.
        command = [
            self._binary,
            "-y",
            "-ss",
            f"{start:.3f}",
            "-i",
            str(input_path),
        ]

        has_watermark = bool(watermark and watermark.get("enabled"))

        if has_watermark:
            # El overlay usa 2 entradas (vídeo + logo): requiere -filter_complex.
            wm_file = watermark.get("file")
            if not wm_file:
                raise ValueError("watermark.file is required when watermark.enabled=true")
            command += ["-i", str(wm_file)]

        command += ["-t", f"{duration:.3f}"]

        if has_watermark:
            width = int(watermark.get("width", 180))
            x, y = self._position_xy(
                watermark.get("position", "top_right"),
                int(watermark.get("margin_x", 40)),
                int(watermark.get("margin_y", 40)),
            )
            enable = ""
            wm_start, wm_end = watermark.get("start"), watermark.get("end")
            if wm_start is not None or wm_end is not None:
                a = float(wm_start or 0.0)
                b = float(wm_end) if wm_end is not None else duration
                enable = f":enable='between(t,{a:.3f},{b:.3f})'"
            filter_complex = (
                f"[0:v]{','.join(vf_parts)}[vmain];"
                f"[1:v]scale={width}:-1[wm];"
                f"[vmain][wm]overlay={x}:{y}{enable}[vout]"
            )
            command += ["-filter_complex", filter_complex]
            # `0:a?`: no fallar si la fuente no tiene audio (el QA lo detecta).
            command += ["-map", "[vout]", "-map", "0:a?"]
        else:
            # Sin watermark: simple filtergraph con -vf (1 entrada / 1 salida).
            command += ["-vf", ",".join(vf_parts)]

        # Forzar framerate de salida. Se coloca DESPUÉS del filtergraph
        # para que se aplique al stream codificado (no al input). Esto
        # evita que clips con fuente 23.976/24/25/29.97 fallen la regla
        # QA de ``min_fps >= 24`` por margen de milésimas.
        if output_fps and output_fps > 0:
            command += ["-r", f"{output_fps:.3f}"]

        command += [
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            "23",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-b:a",
            "128k",
            "-movflags",
            "+faststart",
            str(output),
        ]

        logger.info(
            "rendering clip",
            input=str(input_path),
            output=str(output),
            start=start,
            end=end,
            format=output_format,
            watermark=has_watermark,
            output_fps=output_fps,
        )
        run_command(command, timeout=3600, cwd=vfilter_cwd)
        return output

    @staticmethod
    def _position_xy(position: str, margin_x: int, margin_y: int) -> tuple[str, str]:
        """Devuelve (x, y) para el overlay según la posición y márgenes."""
        if position == "top_left":
            return f"{margin_x}", f"{margin_y}"
        if position == "top_right":
            return f"W-w-{margin_x}", f"{margin_y}"
        if position == "bottom_left":
            return f"{margin_x}", f"H-h-{margin_y}"
        if position == "bottom_right":
            return f"W-w-{margin_x}", f"H-h-{margin_y}"
        if position == "center":
            return "(W-w)/2", "(H-h)/2"
        raise ValueError(f"Unsupported watermark position: {position}")

    @staticmethod
    def _subtitle_filter(path: str | Path) -> str:
        """Elige el filtro de FFmpeg correcto según la extensión del archivo.

        - `.srt`  → filtro `subtitles` (libass lo convierte).
        - `.ass`/`.ssa` → filtro `ass` (formatos estilizados avanzados).
        """
        ext = Path(path).suffix.lower()
        if ext in (".ass", ".ssa"):
            return f"ass={FFmpegTool._escape_path(path)}"
        # Por defecto (incluido .srt): usar el filtro subtitles.
        return f"subtitles={FFmpegTool._escape_path(path)}"

    @staticmethod
    def _escape_path(path: str | Path) -> str:
        """Escapa una ruta para usarla dentro de un filtro de FFmpeg."""
        return str(path).replace("\\", "/").replace(":", "\\:").replace("'", "\\'")
