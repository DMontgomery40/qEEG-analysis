from __future__ import annotations

import base64
from pathlib import Path

from ..reports import get_enhanced_text, get_page_images_base64
from ..storage import Report
from .types import PageImage


def _derive_report_dir(report: Report) -> Path:
    report_text_path = Path(report.extracted_text_path)
    report_dir = report_text_path.parent
    if report_dir.exists():
        return report_dir
    try:
        stored_dir = Path(report.stored_path).parent
        if stored_dir.exists():
            return stored_dir
    except Exception:
        pass
    return report_text_path.parent


def _load_best_report_text(report: Report, report_dir: Path) -> str:
    report_text_path = Path(report.extracted_text_path)
    report_text = report_text_path.read_text(encoding="utf-8", errors="replace")

    enhanced_path = report_dir / "extracted_enhanced.txt"
    enhanced_text: str | None = None
    if enhanced_path.exists():
        enhanced_text = enhanced_path.read_text(encoding="utf-8", errors="replace")
    else:
        try:
            enhanced_text = get_enhanced_text(report.patient_id, report.id)
        except Exception:
            enhanced_text = None

    if enhanced_text and enhanced_text.strip():
        report_text = enhanced_text

    # Never re-extract on the fly here: to avoid widening the error surface in later stages, we require
    # extraction artifacts to already be present and well-formed.
    if "=== PAGE 1 /" not in report_text and Path(report.stored_path).suffix.lower() == ".pdf":
        raise RuntimeError(
            "Report text is missing page markers (expected '=== PAGE 1 /').\n"
            f"Report: {report.filename} ({report.id})\n"
            f"Paths checked: {enhanced_path} and {report_text_path}\n"
            "Fix: re-generate extraction artifacts via POST /api/reports/{report_id}/reextract"
        )

    return report_text


def _load_page_images(report: Report, report_dir: Path) -> list[PageImage]:
    pages_dir = report_dir / "pages"
    if pages_dir.exists():
        page_files = sorted(
            pages_dir.glob("page-*.png"),
            key=lambda p: int(p.stem.split("-")[1]) if "-" in p.stem and p.stem.split("-")[1].isdigit() else 0,
        )
        out: list[PageImage] = []
        for p in page_files:
            try:
                page_num = int(p.stem.split("-")[1])
            except Exception:
                continue
            try:
                out.append(PageImage(page=page_num, base64_png=base64.b64encode(p.read_bytes()).decode("utf-8")))
            except Exception:
                continue
        if out:
            return out

    # Fallback for older artifact formats: no page numbers, assume sequential.
    try:
        images = get_page_images_base64(report.patient_id, report.id)
    except Exception:
        images = []
    return [PageImage(page=i + 1, base64_png=b64) for i, b64 in enumerate(images) if isinstance(b64, str)]

