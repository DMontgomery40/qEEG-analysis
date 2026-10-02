"""The parked hub-upload records the clinic API lists and resolves.

An upload arrives from the hub before anyone knows whose it is. When the name
on it does not match the chart it lands next to, it parks, and a parked upload
nobody can list is a lost upload. These records are read through the shared
clinic upload catalogue; legacy JSON records written beside the pipeline job
status files in the engine's data directory are imported once and then served
from the database.
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any

from .orchestration import pipeline_job_status_dir

# Upload ids name a file, so they may not wander out of the directory.
UPLOAD_ID_RE = re.compile(r"^[A-Za-z0-9._-]{1,128}$")

STATUS_PENDING = "pending"
STATUS_REGISTERED = "registered"


def uploads_dir() -> Path:
    return pipeline_job_status_dir() / "uploads"


def is_valid_upload_id(value: Any) -> bool:
    raw = str(value or "").strip()
    return bool(UPLOAD_ID_RE.match(raw)) and raw not in {".", ".."}


def _record_path(upload_id: str) -> Path:
    return uploads_dir() / f"{upload_id}.json"


def read_upload(upload_id: str) -> dict[str, Any] | None:
    if not is_valid_upload_id(upload_id):
        return None
    from . import storage
    from .clinic_records import ClinicUpload, ClinicLegacyUpload
    from .clinic_intake import _upload_json

    with storage.session_scope() as session:
        current = session.get(ClinicUpload, upload_id)
        if current:
            record = _upload_json(session, current)
            record["resolution"] = (
                json.loads(current.resolution_json) if current.resolution_json else None
            )
            return record
        legacy = session.get(ClinicLegacyUpload, upload_id)
        if legacy:
            return json.loads(legacy.record_json)
    path = _record_path(upload_id)
    if not path.exists():
        return None
    from .clinic_upload_import import import_legacy_record

    return import_legacy_record(json.loads(path.read_text(encoding="utf-8")))


def write_upload(record: dict[str, Any]) -> Path:
    """Compatibility adapter. SQLite commits; old JSON is import evidence only."""
    from .clinic_catalogue import _write, _bump
    from .clinic_catalogue_reads import _json, _patient
    from .clinic_records import ClinicUpload, ClinicLegacyUpload
    from .clinic_intake import _resolution

    upload_id = str(record.get("uploadId") or "").strip()
    if not is_valid_upload_id(upload_id):
        raise ValueError("Invalid upload id")
    with _write() as session:
        current = session.get(ClinicUpload, upload_id)
        if current:
            answer = _resolution(record.get("resolution"))
            if answer and not current.patient_uuid:
                if answer.get("attachTo"):
                    _patient(session, answer["attachTo"])
                if current.resolution_json != _json(answer):
                    current.resolution_json = _json(answer)
                    _bump(session)
            return _record_path(upload_id)
        legacy = session.get(ClinicLegacyUpload, upload_id)
        if legacy is None:
            session.add(
                ClinicLegacyUpload(
                    id=upload_id, evidence_json=_json(record), record_json=_json(record)
                )
            )
            _bump(session)
        elif json.loads(legacy.record_json).get("status") != STATUS_REGISTERED:
            legacy.record_json = _json({**record, "updatedAt": int(time.time() * 1000)})
            _bump(session)
    return _record_path(upload_id)


def list_uploads() -> list[dict[str, Any]]:
    from sqlalchemy import select
    from . import storage
    from .clinic_records import ClinicUpload, ClinicLegacyUpload

    for path in uploads_dir().glob("*.json"):
        if not path.name.startswith("."):
            read_upload(path.stem)
    with storage.session_scope() as session:
        ids = set(session.scalars(select(ClinicUpload.id))) | set(
            session.scalars(select(ClinicLegacyUpload.id))
        )
    return sorted(
        (read_upload(i) for i in ids),
        key=lambda r: int(r.get("updatedAt") or r.get("uploadedAt") or 0),
        reverse=True,
    )
