from __future__ import annotations

import base64
import json
import mimetypes
import os
from pathlib import Path

from pypdf import PdfReader

from .config import REPORTS_DIR


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


def _ocr_available() -> bool:
    """Check if OCR dependencies are available."""
    try:
        import shutil
        if shutil.which("tesseract") is None:
            return False
        import fitz  # PyMuPDF
        import pytesseract
        from PIL import Image
        return True
    except Exception:
        return False


def _render_page_image(pdf_path: Path, page_index: int, dpi_zoom: float = 3.0) -> bytes | None:
    """Render a PDF page as PNG bytes. Returns None on failure."""
    try:
        import fitz  # PyMuPDF
        from PIL import Image
        import io

        doc = fitz.open(str(pdf_path))
        page = doc.load_page(page_index)
        # Render at specified DPI (3.0 zoom from default 72 DPI = ~216 DPI)
        pix = page.get_pixmap(matrix=fitz.Matrix(dpi_zoom, dpi_zoom))
        mode = "RGBA" if pix.alpha else "RGB"
        img = Image.frombytes(mode, (pix.width, pix.height), pix.samples)
        if mode == "RGBA":
            img = img.convert("RGB")

        # Convert to PNG bytes
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        return buffer.getvalue()
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


def extract_text_from_pdf(pdf_path: Path) -> str:
    """
    Extracts text from PDF pages.

    Note: many clinical PDFs embed key tables/figures as images; in those cases, text extraction can be empty.
    If Tesseract OCR is available, we attempt OCR on pages with no extracted text.
    """
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


def extract_text_enhanced(pdf_path: Path) -> str:
    """
    Enhanced text extraction: runs OCR on ALL pages (not just empty ones).
    This captures tables and text rendered as images that pypdf misses.
    """
    reader = PdfReader(str(pdf_path))
    page_count = len(reader.pages)
    ocr_ok = _ocr_available()

    parts: list[str] = []
    for idx, page in enumerate(reader.pages, start=1):
        # First try pypdf text extraction
        pypdf_text = _extract_page_text(page)

        # Always try OCR as well (if available) to capture table images
        ocr_text = ""
        if ocr_ok:
            image_bytes = _render_page_image(pdf_path, idx - 1)
            if image_bytes:
                ocr_text = _ocr_image_bytes(image_bytes)

        # Combine: use OCR if it's significantly longer (captured tables/images)
        # or if pypdf extraction failed
        if not pypdf_text.strip():
            final_text = ocr_text
        elif len(ocr_text) > len(pypdf_text) * 1.2:  # OCR captured 20%+ more content
            final_text = ocr_text
        else:
            final_text = pypdf_text

        header = f"=== PAGE {idx} / {page_count} ==="
        body = final_text.strip() or "[NO TEXT EXTRACTED]"
        parts.append(f"{header}\n{body}".strip())

    return "\n\n".join(parts).strip()


def extract_pdf_with_images(pdf_path: Path) -> tuple[str, list[dict]]:
    """
    Extracts both text and page images from a PDF for multimodal processing.

    Returns:
        tuple: (enhanced_text, page_images)
        - enhanced_text: Full OCR-enhanced text extraction
        - page_images: List of dicts with {"page": int, "base64_png": str}
    """
    # Get enhanced text
    enhanced_text = extract_text_enhanced(pdf_path)

    # Get page images
    reader = PdfReader(str(pdf_path))
    page_count = len(reader.pages)

    page_images: list[dict] = []
    for idx in range(page_count):
        image_bytes = _render_page_image(pdf_path, idx)
        if image_bytes:
            page_images.append({
                "page": idx + 1,
                "base64_png": base64.b64encode(image_bytes).decode("utf-8")
            })

    return enhanced_text, page_images


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
    # OCR is optional: only run if both pytesseract and tesseract are available.
    if not _ocr_available():
        return ""

    image_bytes = _render_page_image(pdf_path, page_index)
    if image_bytes:
        return _ocr_image_bytes(image_bytes)
    return ""


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

    if mime_type == "application/pdf" or original_path.suffix.lower() == ".pdf":
        # Basic extraction (backward compatible)
        extracted = extract_text_from_pdf(original_path)

        # Enhanced extraction with images
        try:
            enhanced_text, page_images = extract_pdf_with_images(original_path)

            # Save enhanced text
            enhanced_path = report_enhanced_path(patient_id, report_id)
            enhanced_path.write_text(enhanced_text, encoding="utf-8")

            # Save page images
            pages_dir = report_pages_dir(patient_id, report_id)
            pages_dir.mkdir(parents=True, exist_ok=True)
            for img_data in page_images:
                page_num = img_data["page"]
                img_bytes = base64.b64decode(img_data["base64_png"])
                img_path = pages_dir / f"page-{page_num}.png"
                img_path.write_bytes(img_bytes)

            # Save metadata
            metadata = {
                "page_count": len(page_images),
                "has_enhanced_ocr": True,
                "has_page_images": len(page_images) > 0,
            }
            metadata_path = report_metadata_path(patient_id, report_id)
            metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

        except Exception:
            # Enhanced extraction failed, continue with basic
            pass
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
