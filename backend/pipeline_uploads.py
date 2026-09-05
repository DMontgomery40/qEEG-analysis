"""The queue of hub uploads that have no patient yet.

An upload arrives from the hub before anyone knows whose it is. Usually the
worker files it within one cycle and nobody needs to look. When the name on it
does not match the chart it lands next to, it parks — and a parked upload nobody
can list is a lost upload, so the worker writes a record here and the API serves
and resolves it.

Records live beside the pipeline job status files, in the engine's own data
directory. Blob credentials stay with the worker: it is the only thing that
talks to the store, and the operator's answer reaches it through this record.
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
STATUS_NEEDS_OPERATOR_ANSWER = "needs_operator_answer"
STATUS_REGISTERED = "registered"
STATUS_FAILED = "failed"


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


def record_seen(*, upload_id: str, identity: dict[str, Any]) -> None:
    existing = read_upload(upload_id)
    if existing:
        return
    write_upload(
        dict(
            uploadId=upload_id, identity=identity, status=STATUS_PENDING, conflict=None
        )
    )


def record_parked(
    *, upload_id: str, identity: dict[str, Any], conflict: dict[str, Any]
) -> None:
    """Park an upload whose name does not match the chart it would land on."""
    existing = read_upload(upload_id) or {}
    write_upload(
        {
            **existing,
            "uploadId": upload_id,
            "identity": identity,
            "status": STATUS_NEEDS_OPERATOR_ANSWER,
            "conflict": conflict,
            "resolution": None,
        }
    )


def record_registered(*, upload_id: str, patient_id: str) -> None:
    """Close an upload out. The record stays so a late answer can be told so."""
    existing = read_upload(upload_id) or {}
    write_upload(
        {
            **existing,
            "uploadId": upload_id,
            "status": STATUS_REGISTERED,
            "patientId": patient_id,
            "conflict": None,
            "resolution": None,
        }
    )


def pending_resolution(upload_id: str) -> dict[str, Any] | None:
    """The operator's answer, if one is waiting to be acted on."""
    record = read_upload(upload_id) or {}
    resolution = record.get("resolution")
    return resolution if isinstance(resolution, dict) and resolution else None


def record_failed(*, upload_id: str, error: str) -> None:
    """Note an upload that fell over, so it is visible rather than just gone.

    Best effort: recording a failure must never raise on top of the failure it
    is recording.
    """
    if not is_valid_upload_id(upload_id):
        return
    try:
        existing = read_upload(upload_id) or {}
        write_upload(
            {
                **existing,
                "uploadId": upload_id,
                "status": STATUS_FAILED,
                "error": error,
            }
        )
    except OSError:
        return
