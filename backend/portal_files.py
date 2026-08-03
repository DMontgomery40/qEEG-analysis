from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

from .patient_identity import parse_canonical_patient_id


def normalize_portal_patient_id(value: str) -> str | None:
    """Return the canonical clinic id a portal key routes on, or None.

    Folders, blob keys, filenames, and job keys all pass through here, so this
    is the single gate that keeps a date of birth from becoming a routing key.
    Surrounding whitespace is tolerated; case is not, because the id is compared
    case-insensitively in SQLite but lands on a case-preserving filesystem.
    """
    parsed = parse_canonical_patient_id(str(value or "").strip())
    return parsed.value if parsed is not None else None


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
    if re.match(
        rf"^{re.escape(lower_label)}__{re.escape(lower_label)}__v\d+__"
        r"\d{4}-\d{2}-\d{2}(?:__[^.]*)?\.pdf$",
        name,
    ):
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
    name = _portal_name(path.name)
    normalized_label = normalize_portal_patient_id(patient_label) or (
        patient_label or ""
    ).strip()
    lower_label = normalized_label.lower()
    if name == "guide.pdf" or name.startswith(f"{lower_label}__guide__"):
        return False

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
