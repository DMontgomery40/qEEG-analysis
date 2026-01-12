"""Tests for PDF text extraction functionality.

These tests use real PDF files from the examples/ directory to validate
that extraction actually works, not just that code runs without errors.

OCR-dependent assertions are conditional on tesseract availability.
"""

from __future__ import annotations

import uuid
from pathlib import Path

import pytest


def test_extract_text_from_pdf_returns_content(temp_data_dir, example_pdf_path: Path):
    """Basic extraction returns non-empty content with page markers."""
    from backend.reports import extract_text_from_pdf

    text = extract_text_from_pdf(example_pdf_path)

    # Always-true assertions (don't depend on OCR)
    assert len(text) > 100, "Extracted text should be substantial"
    assert "=== PAGE 1 /" in text, "Should have page markers"


def test_extract_text_from_pdf_has_multiple_pages(temp_data_dir, example_pdf_path: Path):
    """Extraction should capture multiple pages."""
    from backend.reports import extract_text_from_pdf

    text = extract_text_from_pdf(example_pdf_path)

    # Check for multiple page markers
    page_count = text.count("=== PAGE")
    assert page_count >= 2, f"Expected multiple pages, found {page_count}"


def test_extract_text_enhanced_runs(temp_data_dir, example_pdf_path: Path):
    """Enhanced extraction (OCR all pages) should run without error."""
    from backend.reports import extract_text_enhanced, _ocr_available

    if not _ocr_available():
        pytest.skip("Tesseract OCR not available")

    text = extract_text_enhanced(example_pdf_path)

    assert len(text) > 100, "Enhanced extraction should return content"
    assert "=== PAGE 1 /" in text, "Should have page markers"


def test_extract_pdf_with_images_returns_tuple(temp_data_dir, example_pdf_path: Path):
    """extract_pdf_with_images returns (text, page_images) tuple."""
    from backend.reports import extract_pdf_with_images

    result = extract_pdf_with_images(example_pdf_path)

    assert isinstance(result, tuple), "Should return a tuple"
    assert len(result) == 2, "Tuple should have (text, page_images)"

    text, page_images = result
    assert isinstance(text, str), "First element should be text string"
    assert isinstance(page_images, list), "Second element should be list of images"


def test_extract_pdf_with_images_creates_page_images(temp_data_dir, example_pdf_path: Path):
    """extract_pdf_with_images should create base64 page images."""
    from backend.reports import extract_pdf_with_images

    text, page_images = extract_pdf_with_images(example_pdf_path)

    # Should have at least one page image
    assert len(page_images) >= 1, "Should have at least one page image"

    # Each image should be a dict with base64_png and page keys
    for i, img in enumerate(page_images):
        assert isinstance(img, dict), f"Page image {i} should be a dict"
        assert "base64_png" in img, f"Page image {i} should have 'base64_png' key"
        assert "page" in img, f"Page image {i} should have 'page' key"
        assert len(img["base64_png"]) > 1000, f"Page image {i} should have substantial base64 content"


def test_save_report_upload_creates_files(temp_data_dir, example_pdf_bytes: bytes):
    """save_report_upload should create all expected files."""
    from backend.reports import save_report_upload

    patient_id = str(uuid.uuid4())
    report_id = str(uuid.uuid4())

    original_path, extracted_path, mime_type, preview = save_report_upload(
        patient_id=patient_id,
        report_id=report_id,
        filename="test.pdf",
        provided_mime_type="application/pdf",
        file_bytes=example_pdf_bytes,
    )

    # Verify return types
    assert isinstance(original_path, Path), "original_path should be Path"
    assert isinstance(extracted_path, Path), "extracted_path should be Path"
    assert isinstance(mime_type, str), "mime_type should be string"
    assert isinstance(preview, str), "preview should be string"

    # Verify files exist
    assert original_path.exists(), f"Original file should exist at {original_path}"
    assert extracted_path.exists(), f"Extracted text should exist at {extracted_path}"

    # Verify extracted text has content
    extracted_text = extracted_path.read_text()
    assert len(extracted_text) > 100, "Extracted text should have content"

    # Verify preview is non-empty
    assert len(preview) > 50, "Preview should have content"


def test_save_report_upload_creates_directory_structure(temp_data_dir, example_pdf_bytes: bytes):
    """save_report_upload should create proper directory structure."""
    from backend.reports import save_report_upload
    from backend.config import REPORTS_DIR

    patient_id = str(uuid.uuid4())
    report_id = str(uuid.uuid4())

    save_report_upload(
        patient_id=patient_id,
        report_id=report_id,
        filename="test.pdf",
        provided_mime_type="application/pdf",
        file_bytes=example_pdf_bytes,
    )

    # Check directory structure
    report_dir = REPORTS_DIR / patient_id / report_id
    assert report_dir.exists(), f"Report directory should exist at {report_dir}"

    # Check for expected files
    original_file = report_dir / "original.pdf"
    assert original_file.exists(), "original.pdf should exist"

    extracted_file = report_dir / "extracted.txt"
    assert extracted_file.exists(), "extracted.txt should exist"


def test_save_report_upload_creates_page_images_for_pdf(temp_data_dir, example_pdf_bytes: bytes):
    """save_report_upload should create page images for PDFs."""
    from backend.reports import save_report_upload
    from backend.config import REPORTS_DIR

    patient_id = str(uuid.uuid4())
    report_id = str(uuid.uuid4())

    save_report_upload(
        patient_id=patient_id,
        report_id=report_id,
        filename="test.pdf",
        provided_mime_type="application/pdf",
        file_bytes=example_pdf_bytes,
    )

    # Check for pages directory
    pages_dir = REPORTS_DIR / patient_id / report_id / "pages"
    assert pages_dir.exists(), f"Pages directory should exist at {pages_dir}"

    # Check for at least one page image
    page_files = list(pages_dir.glob("page-*.png"))
    assert len(page_files) >= 1, f"Should have at least one page image, found {len(page_files)}"


def test_save_report_upload_handles_txt_files(temp_data_dir):
    """save_report_upload should handle plain text files."""
    from backend.reports import save_report_upload

    patient_id = str(uuid.uuid4())
    report_id = str(uuid.uuid4())
    text_content = "This is a test qEEG report.\nP300 latency: 320 ms\nTheta/beta ratio: 2.1"

    original_path, extracted_path, mime_type, preview = save_report_upload(
        patient_id=patient_id,
        report_id=report_id,
        filename="report.txt",
        provided_mime_type="text/plain",
        file_bytes=text_content.encode("utf-8"),
    )

    # Text files should have same content for original and extracted
    assert original_path.exists()
    assert extracted_path.exists()
    assert mime_type == "text/plain"
    assert "qEEG report" in preview


def test_get_page_images_base64_retrieves_stored_images(temp_data_dir, example_pdf_bytes: bytes):
    """get_page_images_base64 should retrieve previously stored page images."""
    from backend.reports import save_report_upload, get_page_images_base64

    patient_id = str(uuid.uuid4())
    report_id = str(uuid.uuid4())

    # First, save the report (creates page images)
    save_report_upload(
        patient_id=patient_id,
        report_id=report_id,
        filename="test.pdf",
        provided_mime_type="application/pdf",
        file_bytes=example_pdf_bytes,
    )

    # Then retrieve the page images
    images = get_page_images_base64(patient_id, report_id)

    assert isinstance(images, list), "Should return a list"
    assert len(images) >= 1, "Should have at least one image"

    for i, img in enumerate(images):
        assert isinstance(img, str), f"Image {i} should be a string"
        assert len(img) > 1000, f"Image {i} should be substantial base64"


def test_get_enhanced_text_retrieves_stored_text(temp_data_dir, example_pdf_bytes: bytes):
    """get_enhanced_text should retrieve previously stored enhanced OCR text."""
    from backend.reports import save_report_upload, get_enhanced_text, _ocr_available

    if not _ocr_available():
        pytest.skip("Tesseract OCR not available - enhanced text may not be created")

    patient_id = str(uuid.uuid4())
    report_id = str(uuid.uuid4())

    # First, save the report (creates enhanced text if OCR available)
    save_report_upload(
        patient_id=patient_id,
        report_id=report_id,
        filename="test.pdf",
        provided_mime_type="application/pdf",
        file_bytes=example_pdf_bytes,
    )

    # Then retrieve the enhanced text
    text = get_enhanced_text(patient_id, report_id)

    if text is not None:
        assert isinstance(text, str), "Enhanced text should be a string"
        assert len(text) > 100, "Enhanced text should have content"
