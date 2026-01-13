from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class AppleVisionOcrLine:
    text: str
    confidence: float | None = None
    bbox: tuple[float, float, float, float] | None = None  # (x, y, w, h) in normalized coords


def apple_vision_available() -> bool:
    try:
        import Vision  # noqa: F401
        import Quartz  # noqa: F401
        from Foundation import NSData  # noqa: F401
        return True
    except Exception:
        return False


def _norm_bbox(box: Any) -> tuple[float, float, float, float] | None:
    try:
        origin = box.origin
        size = box.size
        return (float(origin.x), float(origin.y), float(size.width), float(size.height))
    except Exception:
        return None


def apple_vision_ocr_png_bytes(
    image_bytes: bytes,
    *,
    recognition_level: str = "accurate",
    use_language_correction: bool = False,
    min_confidence: float | None = None,
) -> list[AppleVisionOcrLine]:
    """
    OCR a PNG image using Apple's Vision framework (macOS only).

    Proven reference implementations live in:
    - `/Users/davidmontgomery/secondbrain/src/second_brain/ocr/apple_vision_ocr.py`
    - `/Users/davidmontgomery/secondbrain-ds/brain_scans/ocr_all_pages.py`
    """
    if not image_bytes:
        return []

    try:
        import objc
    except Exception:  # pragma: no cover
        objc = None

    def run() -> list[AppleVisionOcrLine]:
        from Foundation import NSData
        from Quartz import CGImageSourceCreateWithData, CGImageSourceCreateImageAtIndex
        from Vision import VNImageRequestHandler, VNRecognizeTextRequest

        # Build CGImage from PNG bytes (avoid temp files).
        data = NSData.dataWithBytes_length_(image_bytes, len(image_bytes))
        image_source = CGImageSourceCreateWithData(data, None)
        if not image_source:
            return []
        cg_image = CGImageSourceCreateImageAtIndex(image_source, 0, None)
        if not cg_image:
            return []

        request = VNRecognizeTextRequest.alloc().init()

        # PyObjC exposes VNRequestTextRecognitionLevel as ints; follow the known-good pattern:
        # - 0: fast
        # - 1: accurate
        if recognition_level.lower().strip() == "fast":
            request.setRecognitionLevel_(0)
        else:
            request.setRecognitionLevel_(1)

        request.setUsesLanguageCorrection_(bool(use_language_correction))

        handler = VNImageRequestHandler.alloc().initWithCGImage_options_(cg_image, None)
        ok, _err = handler.performRequests_error_([request], None)
        if not ok:
            return []

        results = request.results() or []
        out: list[AppleVisionOcrLine] = []
        for obs in results:
            try:
                candidates = obs.topCandidates_(1)
            except Exception:
                candidates = None
            if not candidates:
                continue
            cand = candidates[0]
            try:
                conf = float(cand.confidence())
            except Exception:
                conf = None
            if min_confidence is not None and conf is not None and conf < min_confidence:
                continue
            try:
                text = str(cand.string() or "").strip()
            except Exception:
                text = ""
            if not text:
                continue
            bbox = None
            try:
                bbox = _norm_bbox(obs.boundingBox())
            except Exception:
                bbox = None
            out.append(AppleVisionOcrLine(text=text, confidence=conf, bbox=bbox))
        return out

    if objc is not None:
        with objc.autorelease_pool():
            return run()
    return run()

