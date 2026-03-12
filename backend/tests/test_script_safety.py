from __future__ import annotations

import json
from pathlib import Path

import pytest


def test_create_combined_report_rejects_duplicate_patient_labels(temp_data_dir):
    from backend import storage
    from scripts import create_combined_council_report as script

    with storage.session_scope() as session:
        storage.create_patient(session, label="09-05-1954-0", notes="")
        storage.create_patient(session, label="09-05-1954-0", notes="")

    with pytest.raises(RuntimeError, match="Multiple patients found for label"):
        script._patient_id_for_label("09-05-1954-0")


def test_single_agent_auto_discovery_rejects_ambiguous_matches(tmp_path: Path):
    from scripts import generate_single_agent_patient_report as script

    source_a = tmp_path / "session-a.pdf"
    source_b = tmp_path / "session-b.pdf"
    source_a.write_text("a", encoding="utf-8")
    source_b.write_text("b", encoding="utf-8")

    manifest_path = tmp_path / "combined.manifest.json"
    manifest = {
        "patient_label": "09-05-1954-0",
        "sources": [
            {"path": str(source_a)},
            {"path": str(source_b)},
        ],
    }
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    reports_root = tmp_path / "reports"
    for idx in (1, 2):
        report_dir = reports_root / f"patient-{idx}" / f"report-{idx}"
        report_dir.mkdir(parents=True, exist_ok=True)
        metadata = {
            "synthetic_combined": {
                "source_files": [
                    {"path": str(source_a.resolve())},
                    {"path": str(source_b.resolve())},
                ]
            }
        }
        (report_dir / "metadata.json").write_text(
            json.dumps(metadata), encoding="utf-8"
        )

    with pytest.raises(
        RuntimeError, match="Multiple combined report directories match"
    ):
        script._find_combined_report_dir(
            manifest=manifest,
            manifest_path=manifest_path,
            reports_root=reports_root,
        )
