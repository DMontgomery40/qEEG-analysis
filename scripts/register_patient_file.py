#!/usr/bin/env python
from __future__ import annotations

import argparse
import shutil
import sys
import uuid
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from backend import storage  # noqa: E402
from backend.patient_files import patient_file_original_path  # noqa: E402


def register_patient_file(
    *,
    patient_label: str,
    src_path: Path,
    filename: str | None = None,
    mime_type: str = "application/octet-stream",
) -> dict[str, str | int]:
    source = Path(src_path).expanduser().resolve()
    if not source.exists() or source.stat().st_size <= 0:
        raise FileNotFoundError(f"Source file missing or empty: {source}")

    display_name = filename or source.name
    with storage.session_scope() as session:
        patient = storage.find_patients_by_label(session, patient_label)
        if not patient:
            raise ValueError(f"No qEEG patient found with label {patient_label!r}")
        patient_id = patient[0].id
        existing = (
            session.query(storage.PatientFile)
            .filter(
                storage.PatientFile.patient_id == patient_id,
                storage.PatientFile.filename == display_name,
            )
            .one_or_none()
        )
        file_id = existing.id if existing else str(uuid.uuid4())
        target_path = patient_file_original_path(patient_id, file_id, display_name)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target_path)
        size_bytes = int(target_path.stat().st_size)

        if existing:
            existing.mime_type = mime_type
            existing.size_bytes = size_bytes
            existing.stored_path = str(target_path)
            session.commit()
            session.refresh(existing)
            row = existing
        else:
            row = storage.create_patient_file(
                session,
                file_id=file_id,
                patient_id=patient_id,
                filename=display_name,
                mime_type=mime_type,
                size_bytes=size_bytes,
                stored_path=target_path,
            )
        return {
            "id": row.id,
            "patient_id": row.patient_id,
            "filename": row.filename,
            "mime_type": row.mime_type,
            "size_bytes": row.size_bytes,
            "stored_path": row.stored_path,
        }


def _main() -> int:
    parser = argparse.ArgumentParser(description="Register an existing file in qEEG patient_files.")
    parser.add_argument("--patient-label", required=True)
    parser.add_argument("--src", required=True)
    parser.add_argument("--filename", default="")
    parser.add_argument("--mime-type", default="application/octet-stream")
    args = parser.parse_args()
    result = register_patient_file(
        patient_label=args.patient_label,
        src_path=Path(args.src),
        filename=args.filename or None,
        mime_type=args.mime_type,
    )
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
