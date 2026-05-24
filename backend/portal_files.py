from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any


_PORTAL_PATIENT_ID_RE = re.compile(
    r"^\s*(?P<mm>\d{1,2})-(?P<dd>\d{1,2})-(?P<yyyy>\d{4})-(?P<n>\d{1,3})\s*$"
)


def normalize_portal_patient_id(value: str) -> str | None:
    match = _PORTAL_PATIENT_ID_RE.match(value or "")
    if not match:
        return None
    mm = int(match.group("mm"))
    dd = int(match.group("dd"))
    yyyy = int(match.group("yyyy"))
    n = int(match.group("n"))
    if not (
        1 <= mm <= 12
        and 1 <= dd <= 31
        and 1900 <= yyyy <= 2100
        and 0 <= n <= 999
    ):
        return None
    return f"{mm:02d}-{dd:02d}-{yyyy:04d}-{n}"


def _portal_name(value: str | Path) -> str:
    return os.path.basename(str(value or "").strip()).lower()


def looks_generated_portal_pdf(patient_label: str, value: str | Path) -> bool:
    name = _portal_name(value)
    if not name.endswith(".pdf"):
        return True

    normalized_label = normalize_portal_patient_id(patient_label) or (
        patient_label or ""
    ).strip()
    lower_label = normalized_label.lower()

    if name in {f"{lower_label}.pdf", "main.pdf"}:
        return True
    if "__patient-facing__" in name or "patient-facing" in name:
        return True
    if "__single-agent" in name:
        return True
    if "__analysis__" in name or name.endswith("__analysis.pdf"):
        return True
    if name.endswith("_analysis_pdf.pdf"):
        return True
    return False


def is_source_report_pdf(patient_label: str, path: Path) -> bool:
    return (
        path.is_file()
        and path.suffix.lower() == ".pdf"
        and not looks_generated_portal_pdf(
            patient_label,
            path.name,
        )
    )


def report_asset_dir_from_paths(
    *,
    stored_path: str | Path | None,
    extracted_text_path: str | Path | None,
) -> Path:
    extracted = Path(str(extracted_text_path or ""))
    if extracted.name:
        extracted_dir = extracted.parent
        if extracted_dir.exists():
            return extracted_dir

    stored = Path(str(stored_path or ""))
    if stored.name:
        stored_dir = stored.parent
        if stored_dir.exists():
            return stored_dir

    return extracted.parent if extracted.name else stored.parent


def report_asset_dir_for_report(report: Any) -> Path:
    return report_asset_dir_from_paths(
        stored_path=getattr(report, "stored_path", None),
        extracted_text_path=getattr(report, "extracted_text_path", None),
    )
