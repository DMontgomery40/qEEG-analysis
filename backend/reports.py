from __future__ import annotations

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
    try:
        import shutil

        if shutil.which("tesseract") is None:
            return ""
        import fitz  # PyMuPDF
        import pytesseract
        from PIL import Image
    except Exception:
        return ""

    try:
        doc = fitz.open(str(pdf_path))
        page = doc.load_page(page_index)
        # Render at ~200 DPI (2.0 zoom from default 72 DPI ~= 144 DPI; 3.0 ~= 216 DPI)
        pix = page.get_pixmap(matrix=fitz.Matrix(3.0, 3.0))
        mode = "RGBA" if pix.alpha else "RGB"
        img = Image.frombytes(mode, (pix.width, pix.height), pix.samples)
        if mode == "RGBA":
            img = img.convert("RGB")
        text = pytesseract.image_to_string(img)
        return (text or "").strip()
    except Exception:
        return ""


def report_dir(patient_id: str, report_id: str) -> Path:
    return REPORTS_DIR / patient_id / report_id


def report_original_path(patient_id: str, report_id: str, filename: str) -> Path:
    ext = _safe_suffix(filename)
    if not ext:
        ext = ".bin"
    return report_dir(patient_id, report_id) / f"original{ext}"


def report_extracted_path(patient_id: str, report_id: str) -> Path:
    return report_dir(patient_id, report_id) / "extracted.txt"


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
    """
    mime_type = _guess_mime_type(filename, provided_mime_type)

    out_dir = report_dir(patient_id, report_id)
    out_dir.mkdir(parents=True, exist_ok=True)

    original_path = report_original_path(patient_id, report_id, filename)
    original_path.write_bytes(file_bytes)

    if mime_type == "application/pdf" or original_path.suffix.lower() == ".pdf":
        extracted = extract_text_from_pdf(original_path)
    else:
        extracted = file_bytes.decode("utf-8", errors="replace").strip()

    extracted_path = report_extracted_path(patient_id, report_id)
    extracted_path.write_text(extracted, encoding="utf-8")

    preview = extracted[:4000]
    return original_path, extracted_path, mime_type, preview
