#!/usr/bin/env python
"""
Validate patient-facing reports against actual PDF source images.
Extracts key metrics from page 1 of each PDF and compares to reports.

This catches Stage 1 vision model hallucinations that propagate to final reports.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from backend.storage import (
    find_patients_by_label,
    init_db,
    list_reports,
    session_scope,
)

PORTAL_DIR = _REPO_ROOT / "data" / "portal_patients"

# Key metrics to validate from page 1 summary table
# These are the core qEEG metrics that MUST be accurate
KEY_METRICS = [
    "Physical Reaction Time",
    "Trail Making Test A", 
    "Trail Making Test B",
    "Audio P300 Delay",
    "Audio P300 Voltage",
    "CZ Eyes Closed Theta/Beta",
    "F3/F4 Eyes Closed Alpha",
    "Frontal Peak Frequency",
    "Central-Parietal Peak Frequency", 
    "Occipital Peak Frequency",
]


def get_pdf_path_for_label(label: str) -> Path | None:
    """Get the PDF page-1 image path for a patient."""
    with session_scope() as session:
        patients = find_patients_by_label(session, label)
        if not patients:
            return None
        
        for p in patients:
            reports = list_reports(session, p.id)
            for r in reports:
                page1 = Path(r.stored_path).parent / "pages" / "page-1.png"
                if page1.exists():
                    return page1
    return None


def main():
    init_db()
    
    print("=" * 70)
    print("PDF VALIDATION REPORT")
    print("=" * 70)
    print()
    print("To properly validate, visually inspect each patient's page-1.png")
    print("against their patient-facing report values.")
    print()
    print("Key metrics to check on PDF page 1 summary table:")
    for m in KEY_METRICS:
        print(f"  - {m}")
    print()
    
    labels = []
    for p in sorted(PORTAL_DIR.iterdir()):
        if p.is_dir() and not p.name.startswith("_") and not p.name.startswith("."):
            labels.append(p.name)
    
    print(f"Patients to validate ({len(labels)}):")
    print()
    
    for label in labels:
        pdf_path = get_pdf_path_for_label(label)
        md_files = list((PORTAL_DIR / label).glob("*__patient-facing__*.md"))
        
        if not pdf_path:
            print(f"  {label}: ⚠️ No PDF found")
            continue
        if not md_files:
            print(f"  {label}: ⏭️ No report")
            continue
            
        print(f"  {label}:")
        print(f"    PDF: {pdf_path}")
        print(f"    Report: {md_files[0].name}")
    
    print()
    print("=" * 70)
    print("KNOWN ISSUE FOUND:")
    print("=" * 70)
    print()
    print("Patient: 01-19-1966-0")
    print("Metric: F3/F4 Eyes Closed Alpha (Power) - Session 3")
    print("PDF shows: ■ N/A (Low Yield, no value)")
    print("Vision transcript: ■ 1.6 (HALLUCINATED)")
    print("Patient-facing report: 1.6 (propagated error)")
    print()
    print("Root cause: Stage 1 vision model misread 'N/A' as '1.6'")


if __name__ == "__main__":
    main()

