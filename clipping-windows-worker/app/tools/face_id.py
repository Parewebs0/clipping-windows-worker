"""Face identity for reframe (T1b): who is the campaign's subject?

OpenCV ``FaceRecognizerSF`` with SFace (Apache-2.0) embeddings, faces aligned
with YuNet (MIT) landmarks. Both ONNX models come from opencv_zoo and run on
CPU in a few ms per face. No AGPL code.

Usage::

    fid = open_face_id(model_dir)
    refs = fid.reference_features([Path("ref1.jpg"), ...])
    for box, feature in fid.faces(bgr):           # all faces in a frame
        sim = fid.similarity(feature, refs)        # max cosine vs references
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import structlog

logger = structlog.get_logger(__name__)

YUNET = "face_detection_yunet_2023mar.onnx"
SFACE = "face_recognition_sface_2021dec.onnx"
# SFace's published cosine threshold for "same person".
MATCH_COSINE = 0.363
MIN_FACE_PX = 36


def model_paths(model_dir: Path | None = None) -> tuple[Path, Path] | None:
    """YuNet + SFace files: REFRAME_FACE_ID_DIR, then model_dir. None if missing."""
    dirs = [Path(p) for p in (os.environ.get("REFRAME_FACE_ID_DIR"), model_dir) if p]
    for base in dirs:
        yunet, sface = base / YUNET, base / SFACE
        if yunet.is_file() and sface.is_file():
            return yunet, sface
    return None


class FaceId:
    def __init__(self, yunet: Path, sface: Path):
        import cv2

        self._cv2 = cv2
        self._detector = cv2.FaceDetectorYN.create(str(yunet), "", (320, 320), 0.6, 0.3, 50)
        self._recognizer = cv2.FaceRecognizerSF.create(str(sface), "")

    def faces(self, bgr: Any) -> list[tuple[tuple[float, float, float, float], Any]]:
        """``[((x, y, w, h), feature)]`` for every face YuNet finds (pixels of ``bgr``)."""
        h, w = bgr.shape[:2]
        self._detector.setInputSize((w, h))
        _, found = self._detector.detect(bgr)
        out = []
        for row in found if found is not None else []:
            x, y, fw, fh = (float(v) for v in row[:4])
            if min(fw, fh) < MIN_FACE_PX:
                continue
            aligned = self._recognizer.alignCrop(bgr, row)
            out.append(((x, y, fw, fh), self._recognizer.feature(aligned).copy()))
        return out

    def reference_features(self, paths: list[Path]) -> list[Any]:
        """Largest face of each reference image. Unreadable or faceless images are skipped."""
        refs = []
        for path in paths:
            img = self._cv2.imread(str(path))
            if img is None:
                logger.warning("subject reference unreadable", path=str(path))
                continue
            found = self.faces(img)
            if not found:
                logger.warning("subject reference has no face", path=str(path))
                continue
            refs.append(max(found, key=lambda item: item[0][2] * item[0][3])[1])
        return refs

    def similarity(self, feature: Any, refs: list[Any]) -> float:
        if not refs:
            return 0.0
        return max(
            float(self._recognizer.match(feature, ref, self._cv2.FaceRecognizerSF_FR_COSINE)) for ref in refs
        )


def open_face_id(model_dir: Path | None = None) -> FaceId | None:
    paths = model_paths(model_dir)
    if paths is None:
        logger.warning("face id models missing", dir=os.environ.get("REFRAME_FACE_ID_DIR"))
        return None
    try:
        return FaceId(*paths)
    except Exception as exc:  # noqa: BLE001 — identity is optional, reframe keeps T1 behaviour
        logger.warning("face id unavailable", error=str(exc))
        return None
