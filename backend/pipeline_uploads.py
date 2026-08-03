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
    try:
        parsed = json.loads(_record_path(upload_id).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return parsed if isinstance(parsed, dict) else None


def write_upload(record: dict[str, Any]) -> Path:
    """Save a record atomically, so a crash never leaves half of one."""
    upload_id = str(record.get("uploadId") or "").strip()
    if not is_valid_upload_id(upload_id):
        raise ValueError(f"{upload_id!r} is not a usable upload id.")
    path = _record_path(upload_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {**record, "updatedAt": int(time.time() * 1000)}
    tmp_path = path.with_name(f".{path.name}.partial")
    tmp_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    tmp_path.replace(path)
    return path


def list_uploads() -> list[dict[str, Any]]:
    """Every upload the worker has seen, newest first."""
    directory = uploads_dir()
    if not directory.exists():
        return []
    records: list[dict[str, Any]] = []
    for path in sorted(directory.glob("*.json")):
        if path.name.startswith("."):
            continue
        try:
            parsed = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if isinstance(parsed, dict):
            records.append(parsed)
    return sorted(records, key=lambda r: int(r.get("updatedAt") or 0), reverse=True)


def record_seen(*, upload_id: str, identity: dict[str, Any]) -> None:
    """Note an upload the worker is about to work on."""
    existing = read_upload(upload_id) or {}
    write_upload(
        {
            **existing,
            "uploadId": upload_id,
            "identity": identity,
            "status": STATUS_PENDING,
            "conflict": None,
        }
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
