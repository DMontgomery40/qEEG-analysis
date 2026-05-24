from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .portal_files import normalize_portal_patient_id


def portal_export_manifest_filename(patient_label: str) -> str:
    patient_id = normalize_portal_patient_id(patient_label) or "council-export"
    return f"{patient_id}__council-export-meta.json"


def council_export_manifest_payload(
    *,
    patient_label: str,
    run_id: str,
    report_id: str,
    selected_artifact: Any,
    export_md_path: Path,
    export_pdf_path: Path,
    portal_md_path: Path | None,
    portal_pdf_path: Path | None,
) -> dict[str, Any]:
    return {
        "schema": "qeeg.portal.council_export.v1",
        "patient_label": patient_label,
        "run_id": run_id,
        "report_id": report_id,
        "selected_artifact_id": getattr(selected_artifact, "id", None),
        "selected_artifact": {
            "stage_num": getattr(selected_artifact, "stage_num", None),
            "stage_name": getattr(selected_artifact, "stage_name", None),
            "kind": getattr(selected_artifact, "kind", None),
            "model_id": getattr(selected_artifact, "model_id", None),
            "content_type": getattr(selected_artifact, "content_type", None),
            "content_path": getattr(selected_artifact, "content_path", None),
        },
        "exports": {
            "final_md": str(export_md_path),
            "final_pdf": str(export_pdf_path),
            "portal_final_md": str(portal_md_path) if portal_md_path else None,
            "portal_final_pdf": str(portal_pdf_path) if portal_pdf_path else None,
        },
        "delivery_gaps": [],
        "exported_at": datetime.now(timezone.utc).isoformat(),
    }


def write_council_export_manifest(
    *,
    portal_patient_dir: Path,
    patient_label: str,
    payload: dict[str, Any],
) -> Path:
    portal_patient_dir.mkdir(parents=True, exist_ok=True)
    path = portal_patient_dir / portal_export_manifest_filename(patient_label)
    tmp_path = path.with_name(f".{path.name}.partial")
    tmp_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    tmp_path.replace(path)
    return path
