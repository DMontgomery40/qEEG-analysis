from __future__ import annotations

import base64
from dataclasses import dataclass
import json
import mimetypes
import os
import re
from pathlib import Path
from typing import Any

from pypdf import PdfReader

from .config import REPORTS_DIR
from .apple_vision_ocr import apple_vision_available, apple_vision_ocr_png_bytes


def _safe_suffix(filename: str) -> str:
    base = os.path.basename(filename or "")
    _, ext = os.path.splitext(base)
    ext = (ext or "").lower()
    if ext and len(ext) <= 10:
        return ext
    return ""


def _guess_mime_type(filename: str, provided: str | None) -> str:
    if provided and isinstance(provided, str) and provided.strip():
        return provided
    guessed, _ = mimetypes.guess_type(filename)
    return guessed or "application/octet-stream"


def _tesseract_ocr_available() -> bool:
    try:
        import shutil

        if shutil.which("tesseract") is None:
            return False
        import pytesseract  # noqa: F401
        from PIL import Image  # noqa: F401

        return True
    except Exception:
        return False


def _ocr_available() -> bool:
    """Check if ANY OCR engine is available for enhanced extraction."""
    # Apple Vision OCR is macOS-only; Tesseract is cross-platform (if installed).
    return apple_vision_available() or _tesseract_ocr_available()


def _render_page_image(pdf_path: Path, page_index: int, dpi_zoom: float = 3.0) -> bytes | None:
    """Render a PDF page as PNG bytes. Returns None on failure."""
    try:
        import fitz  # PyMuPDF

        doc = fitz.open(str(pdf_path))
        page = doc.load_page(page_index)
        pix = page.get_pixmap(matrix=fitz.Matrix(dpi_zoom, dpi_zoom), alpha=False)
        return pix.tobytes("png")
    except Exception:
        return None


def _ocr_image_bytes(image_bytes: bytes) -> str:
    """Run OCR on image bytes. Returns empty string on failure."""
    try:
        import pytesseract
        from PIL import Image
        import io

        img = Image.open(io.BytesIO(image_bytes))
        text = pytesseract.image_to_string(img)
        return (text or "").strip()
    except Exception:
        return ""


def _ocr_image_bytes_apple_vision(image_bytes: bytes) -> str:
    """Run Apple Vision OCR on image bytes. Returns empty string on failure."""
    try:
        min_confidence: float | None = None
        raw_min_conf = os.getenv("QEEG_APPLE_VISION_MIN_CONFIDENCE")
        if isinstance(raw_min_conf, str) and raw_min_conf.strip():
            try:
                candidate = float(raw_min_conf)
                if 0.0 <= candidate <= 1.0:
                    min_confidence = candidate
            except Exception:
                min_confidence = None

        lines = apple_vision_ocr_png_bytes(
            image_bytes,
            recognition_level=os.getenv("QEEG_APPLE_VISION_RECOGNITION_LEVEL", "accurate") or "accurate",
            use_language_correction=_truthy_env("QEEG_APPLE_VISION_LANGUAGE_CORRECTION", False),
            min_confidence=min_confidence,
        )
        return "\n".join(l.text for l in lines if l.text).strip()
    except Exception:
        return ""


def _truthy_env(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    raw = raw.strip().lower()
    if raw in {"1", "true", "yes", "y", "on"}:
        return True
    if raw in {"0", "false", "no", "n", "off"}:
        return False
    return default


def extract_text_from_pdf(pdf_path: Path) -> str:
    """
    Extracts text from PDF pages.

    Note: many clinical PDFs embed key tables/figures as images; in those cases, text extraction can be empty.
    If OCR is available, we attempt it on pages with no extracted text.
    """
    try:
        # Prefer the full multi-source extraction (PDF-native + OCR on all pages + rendered images).
        return extract_pdf_full(pdf_path).enhanced_text
    except Exception:
        pass

    reader = PdfReader(str(pdf_path))
    page_count = len(reader.pages)

    parts: list[str] = []
    for idx, page in enumerate(reader.pages, start=1):
        extracted = _extract_page_text(page)
        if not extracted.strip():
            extracted = _maybe_ocr_page(pdf_path, idx - 1)

        header = f"=== PAGE {idx} / {page_count} ==="
        body = extracted.strip() or "[NO TEXT EXTRACTED]"
        parts.append(f"{header}\n{body}".strip())

    return "\n\n".join(parts).strip()


@dataclass(frozen=True)
class PdfFullExtraction:
    enhanced_text: str
    page_images: list[dict[str, Any]]
    per_page_sources: list[dict[str, Any]]
    metadata: dict[str, Any]


def _float_env(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        val = float(raw)
    except Exception:
        return default
    return val if val > 0 else default


def _pymupdf_page_text(pdf_path: Path, page_index: int) -> str:
    # Deprecated: prefer reusing a single PyMuPDF document when extracting multiple pages.
    try:
        import fitz  # PyMuPDF

        doc = fitz.open(str(pdf_path))
        page = doc.load_page(page_index)
        return (page.get_text("text") or "").strip()
    except Exception:
        return ""


def extract_pdf_full(pdf_path: Path) -> PdfFullExtraction:
    """
    Extracts *redundant* per-page sources for maximum completeness and auditability:

    - PDF-native text via pypdf
    - PDF-native text via PyMuPDF
    - Apple Vision OCR (macOS) when available (unless disabled)
    - Tesseract OCR when available
    - Rendered page PNGs (for Stage-1 multimodal)

    Returns:
      PdfFullExtraction(enhanced_text, page_images, per_page_sources, metadata)
    """
    reader = PdfReader(str(pdf_path))
    page_count = len(reader.pages)

    vision_ok = apple_vision_available() and not _truthy_env("QEEG_DISABLE_APPLE_VISION_OCR", False)
    tesseract_ok = _tesseract_ocr_available()
    render_zoom = _float_env("QEEG_PDF_RENDER_ZOOM", 3.0)

    fitz_doc = None
    fitz_ok = False
    try:
        import fitz  # PyMuPDF

        fitz_doc = fitz.open(str(pdf_path))
        fitz_ok = True
    except Exception:
        fitz_doc = None
        fitz_ok = False

    parts: list[str] = []
    page_images: list[dict[str, Any]] = []
    per_page_sources: list[dict[str, Any]] = []
    pages_meta: list[dict[str, Any]] = []

    try:
        for idx, page in enumerate(reader.pages, start=1):
            # PDF-native text sources
            pypdf_text = _extract_page_text(page)
            pymupdf_text = ""

            # Render a page image once and reuse for OCR + multimodal
            image_bytes: bytes | None = None
            if fitz_doc is not None:
                try:
                    fpage = fitz_doc.load_page(idx - 1)
                    pymupdf_text = (fpage.get_text("text") or "").strip()
                    pix = fpage.get_pixmap(matrix=fitz.Matrix(render_zoom, render_zoom), alpha=False)
                    image_bytes = pix.tobytes("png")
                except Exception:
                    pymupdf_text = pymupdf_text or ""
                    image_bytes = None

            # OCR sources
            vision_text = ""
            ocr_text = ""
            if image_bytes:
                if vision_ok:
                    vision_text = _ocr_image_bytes_apple_vision(image_bytes)
                if tesseract_ok:
                    ocr_text = _ocr_image_bytes(image_bytes)

            p = (pypdf_text or "").strip()
            m = (pymupdf_text or "").strip()
            v = (vision_text or "").strip()
            o = (ocr_text or "").strip()

            def norm(s: str) -> str:
                return re.sub(r"\s+", " ", s).strip().lower()

            def combine_sources(sources: list[tuple[str, str]]) -> str:
                # Keep the strongest unique sources; preserve redundancy when sources differ.
                items = [(label, txt.strip(), norm(txt)) for label, txt in sources if isinstance(txt, str) and txt.strip()]
                if not items:
                    return ""
                # Deduplicate exact matches by normalized text; keep the longer variant.
                dedup: list[tuple[str, str, str]] = []
                for label, txt, n in items:
                    replaced = False
                    for i, (_l2, t2, n2) in enumerate(dedup):
                        if n and n == n2:
                            if len(txt) > len(t2):
                                dedup[i] = (label, txt, n)
                            replaced = True
                            break
                    if not replaced:
                        dedup.append((label, txt, n))
                # Drop subsets when another source strictly contains it (normalized).
                kept: list[tuple[str, str]] = []
                for i, (label, txt, n) in enumerate(dedup):
                    if n and any(i != j and n in n2 for j, (_l2, _t2, n2) in enumerate(dedup)):
                        continue
                    kept.append((label, txt))
                if not kept:
                    return ""
                if len(kept) == 1:
                    return kept[0][1]
                return "\n\n".join([f"--- {label} ---\n{txt}" for label, txt in kept])

            # IMPORTANT: This is *not* a "best source" selector; it's a redundancy-preserving union.
            # We order sources for human readability (cleaner OCR tends to come first), and keep all non-duplicate,
            # non-subset sources with explicit markers so downstream consumers can cross-check.
            final_text = combine_sources(
                [
                    ("PYPDF TEXT", p),
                    ("PYMUPDF TEXT", m),
                    ("TESSERACT OCR", o),
                    ("APPLE VISION OCR", v),
                ]
            )

            header = f"=== PAGE {idx} / {page_count} ==="
            body = final_text.strip() or "[NO TEXT EXTRACTED]"
            parts.append(f"{header}\n{body}".strip())

            per_page_sources.append(
                {
                    "page": idx,
                    "pypdf_text": p,
                    "pymupdf_text": m,
                    "vision_ocr_text": v,
                    "tesseract_ocr_text": o,
                }
            )

            pages_meta.append(
                {
                    "page": idx,
                    "pypdf_chars": len(p),
                    "pymupdf_chars": len(m),
                    "apple_vision_chars": len(v),
                    "tesseract_chars": len(o),
                    "render_zoom": render_zoom,
                    "has_png": bool(image_bytes),
                }
            )

            if image_bytes:
                page_images.append({"page": idx, "base64_png": base64.b64encode(image_bytes).decode("utf-8")})
    finally:
        try:
            if fitz_doc is not None:
                fitz_doc.close()
        except Exception:
            pass

    enhanced_text = "\n\n".join(parts).strip()
    metadata: dict[str, Any] = {
        "schema_version": 2,
        "page_count": page_count,
        "render_zoom": render_zoom,
        "engines": {
            "pypdf": True,
            "pymupdf": bool(fitz_ok),
            "apple_vision": bool(vision_ok),
            "tesseract": bool(tesseract_ok),
        },
        "pages": pages_meta,
    }
    return PdfFullExtraction(
        enhanced_text=enhanced_text,
        page_images=page_images,
        per_page_sources=per_page_sources,
        metadata=metadata,
    )


def extract_text_enhanced(pdf_path: Path) -> str:
    """
    Enhanced text extraction: runs OCR on ALL pages (not just empty ones) and
    combines pypdf text + OCR text to avoid losing tables/figures that are
    embedded as images.

    "No compromise on data availability" means we prefer redundancy over
    heuristic selection when both sources have content.
    """
    return extract_pdf_full(pdf_path).enhanced_text


def extract_pdf_with_images(pdf_path: Path) -> tuple[str, list[dict]]:
    """
    Extracts both text and page images from a PDF for multimodal processing.

    Returns:
        tuple: (enhanced_text, page_images)
        - enhanced_text: Full OCR-enhanced text extraction
        - page_images: List of dicts with {"page": int, "base64_png": str}
    """
    extracted = extract_pdf_full(pdf_path)
    return extracted.enhanced_text, extracted.page_images


def _extract_page_text(page) -> str:
    # Prefer the longer of plain/layout extraction.
    try:
        plain = (page.extract_text(extraction_mode="plain") or "").strip()
    except Exception:
        plain = ""
    try:
        layout = (page.extract_text(extraction_mode="layout") or "").strip()
    except Exception:
        layout = ""
    return layout if len(layout) >= len(plain) else plain


def _maybe_ocr_page(pdf_path: Path, page_index: int) -> str:
    # OCR is optional: if available, run Apple Vision OCR *and* Tesseract and keep redundancy.
    vision_ok = apple_vision_available() and not _truthy_env("QEEG_DISABLE_APPLE_VISION_OCR", False)
    tesseract_ok = _tesseract_ocr_available()
    if not (vision_ok or tesseract_ok):
        return ""

    image_bytes = _render_page_image(pdf_path, page_index)
    if not image_bytes:
        return ""

    sources: list[tuple[str, str]] = []
    if tesseract_ok:
        sources.append(("TESSERACT OCR", _ocr_image_bytes(image_bytes)))
    if vision_ok:
        sources.append(("APPLE VISION OCR", _ocr_image_bytes_apple_vision(image_bytes)))

    parts = [f"--- {label} ---\n{txt.strip()}" for label, txt in sources if isinstance(txt, str) and txt.strip()]
    return "\n\n".join(parts).strip()


def report_dir(patient_id: str, report_id: str) -> Path:
    return REPORTS_DIR / patient_id / report_id


def report_pages_dir(patient_id: str, report_id: str) -> Path:
    return report_dir(patient_id, report_id) / "pages"


def report_original_path(patient_id: str, report_id: str, filename: str) -> Path:
    ext = _safe_suffix(filename)
    if not ext:
        ext = ".bin"
    return report_dir(patient_id, report_id) / f"original{ext}"


def report_extracted_path(patient_id: str, report_id: str) -> Path:
    return report_dir(patient_id, report_id) / "extracted.txt"


def report_enhanced_path(patient_id: str, report_id: str) -> Path:
    return report_dir(patient_id, report_id) / "extracted_enhanced.txt"


def report_metadata_path(patient_id: str, report_id: str) -> Path:
    return report_dir(patient_id, report_id) / "metadata.json"


def save_report_upload(
    *,
    patient_id: str,
    report_id: str,
    filename: str,
    provided_mime_type: str | None,
    file_bytes: bytes,
) -> tuple[Path, Path, str, str]:
    """
    Saves original upload and extracted text; returns (original_path, extracted_path, mime_type, preview).

    For PDFs, also saves:
    - Enhanced OCR text to extracted_enhanced.txt
    - Page images to pages/ directory
    - Metadata to metadata.json
    """
    mime_type = _guess_mime_type(filename, provided_mime_type)

    out_dir = report_dir(patient_id, report_id)
    out_dir.mkdir(parents=True, exist_ok=True)

    original_path = report_original_path(patient_id, report_id, filename)
    original_path.write_bytes(file_bytes)

    extracted = ""
    if mime_type == "application/pdf" or original_path.suffix.lower() == ".pdf":
        # Prefer the enhanced multi-source extraction; fall back to basic extraction only if it fails.
        try:
            full = extract_pdf_full(original_path)
            enhanced_text = full.enhanced_text
            page_images = full.page_images

            # Save enhanced text
            enhanced_path = report_enhanced_path(patient_id, report_id)
            enhanced_path.write_text(enhanced_text, encoding="utf-8")

            # Also write extracted.txt as the enhanced union so the UI preview ("extracted") is never
            # constrained to a single OCR engine.
            extracted = enhanced_text

            # Save page images
            pages_dir = report_pages_dir(patient_id, report_id)
            pages_dir.mkdir(parents=True, exist_ok=True)
            for img_data in page_images:
                page_num = img_data["page"]
                img_bytes = base64.b64decode(img_data["base64_png"])
                img_path = pages_dir / f"page-{page_num}.png"
                img_path.write_bytes(img_bytes)

            # Save per-page source text (audit/debug)
            sources_dir = out_dir / "sources"
            sources_dir.mkdir(parents=True, exist_ok=True)
            for p in full.per_page_sources:
                page_num = p.get("page")
                if not isinstance(page_num, int):
                    continue
                try:
                    (sources_dir / f"page-{page_num}.pypdf.txt").write_text(p.get("pypdf_text", ""), encoding="utf-8")
                    (sources_dir / f"page-{page_num}.pymupdf.txt").write_text(p.get("pymupdf_text", ""), encoding="utf-8")
                    (sources_dir / f"page-{page_num}.apple_vision.txt").write_text(
                        p.get("vision_ocr_text", ""), encoding="utf-8"
                    )
                    (sources_dir / f"page-{page_num}.tesseract.txt").write_text(
                        p.get("tesseract_ocr_text", ""), encoding="utf-8"
                    )
                except Exception:
                    continue

            # Save metadata (includes per-page engine lengths + engine availability)
            metadata = dict(full.metadata)
            metadata.update(
                {
                    "has_enhanced_ocr": True,
                    "has_page_images": len(page_images) > 0,
                    "page_images_written": len(page_images),
                    "sources_dir": "sources",
                }
            )
            metadata_path = report_metadata_path(patient_id, report_id)
            metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

        except Exception:
            extracted = extract_text_from_pdf(original_path)
    else:
        extracted = file_bytes.decode("utf-8", errors="replace").strip()

    extracted_path = report_extracted_path(patient_id, report_id)
    extracted_path.write_text(extracted, encoding="utf-8")

    preview = extracted[:4000]
    return original_path, extracted_path, mime_type, preview


def get_page_images_base64(patient_id: str, report_id: str) -> list[str]:
    """
    Get base64-encoded page images for multimodal LLM input.
    Returns list of base64 PNG strings.
    """
    pages_dir = report_pages_dir(patient_id, report_id)
    if not pages_dir.exists():
        return []

    images: list[str] = []
    page_files = sorted(pages_dir.glob("page-*.png"), key=lambda p: int(p.stem.split("-")[1]))
    for page_file in page_files:
        try:
            img_bytes = page_file.read_bytes()
            images.append(base64.b64encode(img_bytes).decode("utf-8"))
        except Exception:
            continue

    return images


def get_enhanced_text(patient_id: str, report_id: str) -> str | None:
    """
    Get enhanced OCR text if available, otherwise return None.
    """
    enhanced_path = report_enhanced_path(patient_id, report_id)
    if enhanced_path.exists():
        return enhanced_path.read_text(encoding="utf-8", errors="replace")
    return None
