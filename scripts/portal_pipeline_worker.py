#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import mimetypes
import tempfile
import time
import uuid
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from backend import storage  # noqa: E402
from backend.orchestration import (  # noqa: E402
    derive_run_liveness,
    run_downstream_delivery_gaps,
    summarize_run_progress,
)
from backend.patient_identity import (  # noqa: E402
    PatientIdentityError,
    allocate_canonical_patient_id,
    parse_canonical_patient_id,
)
from backend import pipeline_uploads  # noqa: E402
from backend.patient_intake import (  # noqa: E402
    IdentityInput,
    IdentityNameConflict,
    find_patient_by_identity,
    identity_key,
)
from backend.portal_files import looks_generated_portal_pdf  # noqa: E402
from backend.reports import report_dir, save_report_upload  # noqa: E402
from scripts.register_patient_file import register_patient_file  # noqa: E402

META_NAME = "$meta.json"
INDEX_NAME = "$index.json"
JOB_PREFIX = "pipeline/jobs"
STATUS_PREFIX = "pipeline/status"


@dataclass(frozen=True)
class PortalReport:
    patient_id: str
    file_key: str
    original_name: str
    logical_name: str
    uploaded_at: int
    size: int
    content_type: str
    document_kind: str | None = None
    report_birthdate: str | None = None
    session_date: str | None = None
    uploaded_by: str | None = None
    from_job: bool = False
    local_name: str | None = None


@dataclass
class PatientWorkerResult:
    patient_id: str
    status: str
    downloaded: list[str]
    report_count: int
    ran_batch: bool
    would_download: list[str] = field(default_factory=list)
    returncode: int | None = None
    stdout_tail: str = ""
    stderr_tail: str = ""
    note: str = ""


def _now_ms() -> int:
    return int(time.time() * 1000)


def is_valid_patient_id(value: str) -> bool:
    return parse_canonical_patient_id(str(value or "").strip()) is not None


def patient_id_from_meta_key(key: str) -> str | None:
    match = re.fullmatch(r"patients/([^/]+)/\$\w+\.json", str(key or ""))
    if not match:
        return None
    patient_id = match.group(1)
    return patient_id if is_valid_patient_id(patient_id) else None


def patient_id_from_job_key(key: str) -> str | None:
    match = re.fullmatch(r"pipeline/jobs/([^/]+)/[^/]+\.json", str(key or ""))
    if not match:
        return None
    patient_id = match.group(1)
    return patient_id if is_valid_patient_id(patient_id) else None


def patient_id_from_file_key(key: str) -> str | None:
    match = re.fullmatch(r"patients/([^/]+)/files/[^/]+", str(key or ""))
    if not match:
        return None
    patient_id = match.group(1)
    return patient_id if is_valid_patient_id(patient_id) else None


def _normalized_patient_identity(value: Any) -> dict[str, Any] | None:
    identity = value if isinstance(value, dict) else {}
    first = str(identity.get("firstInitial") or "").strip().upper()
    last = str(identity.get("lastInitial") or "").strip().upper()
    if not re.fullmatch(r"[A-Z]", first) or not re.fullmatch(r"[A-Z]", last):
        return None
    return {
        "schemaVersion": int(identity.get("schemaVersion") or 1),
        "firstInitial": first,
        "lastInitial": last,
    }


def sync_remote_patient_identity(
    *,
    portal_dir: Path,
    patient_id: str,
    remote_meta: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """Persist hub initials beside the immutable local patient folder."""
    parsed = parse_canonical_patient_id(str(patient_id or "").strip())
    if parsed is None:
        raise ValueError("Invalid patient storage identifier")
    remote_identity = _normalized_patient_identity(
        remote_meta.get("identity") if isinstance(remote_meta, dict) else None
    )
    if remote_identity is None:
        return None

    patient_dir = portal_dir / patient_id
    meta_path = patient_dir / META_NAME
    try:
        existing = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        existing = {}
    local_identity = _normalized_patient_identity(
        existing.get("identity") if isinstance(existing, dict) else None
    )
    if local_identity and (
        local_identity["firstInitial"],
        local_identity["lastInitial"],
    ) != (
        remote_identity["firstInitial"],
        remote_identity["lastInitial"],
    ):
        raise ValueError("Remote patient initials conflict with local identity metadata")

    if (remote_identity["firstInitial"], remote_identity["lastInitial"]) != (
        parsed.first_initial,
        parsed.last_initial,
    ):
        raise ValueError("Remote patient initials conflict with the clinic patient id")

    patient_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "patientId": parsed.value,
        "birthdate": parsed.birthdate,
        "index": parsed.ordinal,
        "identity": remote_identity,
    }
    temp_path = meta_path.with_name(f".{META_NAME}.partial")
    temp_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temp_path.replace(meta_path)
    return remote_identity


def parse_blob_keys(payload: str) -> list[str]:
    parsed = json.loads(payload or "{}")
    blobs = parsed.get("blobs")
    if not isinstance(blobs, list):
        return []
    return [
        item.get("key")
        for item in blobs
        if isinstance(item, dict) and isinstance(item.get("key"), str)
    ]


def _safe_filename(value: str, fallback: str = "report.pdf") -> str:
    raw = Path(str(value or "").strip()).name or fallback
    cleaned = re.sub(r"[/\\\x00-\x1f\x7f]+", "_", raw)
    cleaned = re.sub(r"\s+", " ", cleaned).strip().strip(".")
    return (cleaned[:200] or fallback)


def _fallback_original_name_from_file_key(patient_id: str, file_key: str) -> str:
    raw = str(file_key or "")
    prefixes = [f"{patient_id}__", f"{patient_id}_"]
    for prefix in prefixes:
        if raw.startswith(prefix):
            raw = raw[len(prefix) :]
            break
    raw = re.sub(r"__v\d+__\d{4}-\d{2}-\d{2}(\.[^.]+)$", r"\1", raw)
    raw = re.sub(r"_v\d+_\d{4}-\d{2}-\d{2}(\.[^.]+)$", r"\1", raw)
    return _safe_filename(raw, "report.pdf")


def _looks_generated_pdf(patient_id: str, name: str) -> bool:
    return looks_generated_portal_pdf(patient_id, name)


def reports_from_index(
    patient_id: str, index_payload: dict[str, Any]
) -> list[PortalReport]:
    files = index_payload.get("files") if isinstance(index_payload, dict) else None
    if not isinstance(files, list):
        return []

    reports: list[PortalReport] = []
    seen: set[str] = set()
    for item in files:
        if not isinstance(item, dict):
            continue
        file_key = item.get("fileKey")
        if not isinstance(file_key, str) or file_key in seen:
            continue
        content_type = str(item.get("contentType") or "").lower()
        document_kind = item.get("documentKind")
        original_name = _safe_filename(
            str(item.get("originalName") or item.get("logicalName") or "")
            or _fallback_original_name_from_file_key(patient_id, file_key)
        )
        logical_name = _safe_filename(str(item.get("logicalName") or original_name))
        uploaded_by = str(item.get("uploadedBy") or "").strip() or None
        is_report_pdf = (
            document_kind == "report"
            and content_type == "application/pdf"
            and file_key.lower().endswith(".pdf")
        )
        if not is_report_pdf:
            is_report_pdf = (
                uploaded_by != "local-sync"
                and file_key.lower().endswith(".pdf")
                and not _looks_generated_pdf(patient_id, file_key)
                and not _looks_generated_pdf(patient_id, original_name)
            )
        if not is_report_pdf:
            continue
        reports.append(
            PortalReport(
                patient_id=patient_id,
                file_key=file_key,
                original_name=original_name,
                logical_name=logical_name,
                uploaded_at=int(item.get("uploadedAt") or 0),
                size=int(item.get("size") or 0),
                content_type=content_type or "application/pdf",
                document_kind=str(document_kind) if document_kind else None,
                report_birthdate=(
                    item.get("reportBirthdate")
                    if isinstance(item.get("reportBirthdate"), str)
                    else None
                ),
                session_date=(
                    item.get("sessionDate")
                    if isinstance(item.get("sessionDate"), str)
                    else None
                ),
                uploaded_by=uploaded_by,
                from_job=False,
                local_name=file_key,
            )
        )
        seen.add(file_key)

    by_logical_name: dict[str, list[PortalReport]] = {}
    for report in reports:
        key = (report.logical_name or report.original_name or report.file_key).lower()
        by_logical_name.setdefault(key, []).append(report)

    distinct_reports: list[PortalReport] = []
    for grouped in by_logical_name.values():
        preferred = [
            report for report in grouped if (report.uploaded_by or "") != "local-sync"
        ] or grouped
        if len(preferred) > 1:
            distinct_reports.extend(
                replace(report, local_name=report.file_key) for report in preferred
            )
        else:
            distinct_reports.extend(preferred)

    return sorted(
        distinct_reports, key=lambda report: (report.uploaded_at, report.file_key)
    )


def reports_from_job_payload(
    patient_id: str, job_payload: dict[str, Any]
) -> list[PortalReport]:
    raw_reports = (
        job_payload.get("reportFiles") if isinstance(job_payload, dict) else None
    )
    if not isinstance(raw_reports, list):
        return []
    reports: list[PortalReport] = []
    seen: set[str] = set()
    for item in raw_reports:
        if not isinstance(item, dict):
            continue
        file_key = item.get("fileKey")
        if not isinstance(file_key, str) or file_key in seen:
            continue
        if not file_key.lower().endswith(".pdf"):
            continue
        original_name = _safe_filename(
            str(item.get("originalName") or item.get("logicalName") or "")
            or _fallback_original_name_from_file_key(patient_id, file_key)
        )
        logical_name = _safe_filename(str(item.get("logicalName") or original_name))
        reports.append(
            PortalReport(
                patient_id=patient_id,
                file_key=file_key,
                original_name=original_name,
                logical_name=logical_name,
                uploaded_at=int(
                    item.get("uploadedAt") or job_payload.get("uploadedAt") or 0
                ),
                size=int(item.get("size") or 0),
                content_type=str(item.get("contentType") or "application/pdf").lower(),
                document_kind=str(item.get("documentKind") or "report"),
                report_birthdate=(
                    item.get("reportBirthdate")
                    if isinstance(item.get("reportBirthdate"), str)
                    else None
                ),
                session_date=(
                    item.get("sessionDate")
                    if isinstance(item.get("sessionDate"), str)
                    else None
                ),
                uploaded_by=(
                    str(
                        item.get("uploadedBy") or job_payload.get("uploadedBy") or ""
                    ).strip()
                    or None
                ),
                from_job=True,
            )
        )
        seen.add(file_key)
    return sorted(reports, key=lambda report: (report.uploaded_at, report.file_key))


def reports_from_file_keys(patient_id: str, blob_keys: list[str]) -> list[PortalReport]:
    reports: list[PortalReport] = []
    seen: set[str] = set()
    prefix = f"patients/{patient_id}/files/"
    for key in blob_keys:
        if not isinstance(key, str) or not key.startswith(prefix):
            continue
        file_key = key[len(prefix) :]
        if not file_key or file_key in seen:
            continue
        original_name = _fallback_original_name_from_file_key(patient_id, file_key)
        if _looks_generated_pdf(patient_id, file_key) or _looks_generated_pdf(
            patient_id, original_name
        ):
            continue
        reports.append(
            PortalReport(
                patient_id=patient_id,
                file_key=file_key,
                original_name=original_name,
                logical_name=original_name,
                uploaded_at=0,
                size=0,
                content_type="application/pdf",
                document_kind=None,
                from_job=False,
                local_name=file_key,
            )
        )
        seen.add(file_key)
    return sorted(reports, key=lambda report: (report.uploaded_at, report.file_key))


def analysis_artifacts_exist(patient_dir: Path, patient_id: str) -> bool:
    if not patient_dir.exists():
        return False
    for path in patient_dir.rglob("*"):
        if not path.is_file():
            continue
        try:
            rel = path.relative_to(patient_dir).as_posix()
        except Exception:
            continue
        name = path.name.lower()
        if rel.startswith("council/"):
            return True
        if "patient-facing" in name or "analysis" in name:
            return True
        if name == f"{patient_id.lower()}.md":
            return True
    return False


def source_local_filename(report: PortalReport) -> str:
    if report.local_name:
        return _safe_filename(report.local_name, "report.pdf")
    if report.from_job and report.file_key:
        return _safe_filename(report.file_key, "report.pdf")
    if report.document_kind == "report" and report.logical_name:
        return _safe_filename(report.logical_name, "report.pdf")
    return _safe_filename(
        report.original_name
        or report.logical_name
        or _fallback_original_name_from_file_key(report.patient_id, report.file_key),
        "report.pdf",
    )


def local_report_path(portal_dir: Path, report: PortalReport) -> Path:
    filename = source_local_filename(report)
    return portal_dir / report.patient_id / _safe_filename(filename, "report.pdf")


def _matching_active_run_exists(patient_label: str, filename: str) -> bool:
    with storage.session_scope() as session:
        patients = storage.find_patients_by_label(session, patient_label)
        for patient in patients:
            for report in storage.list_reports(session, patient.id):
                if (report.filename or "") != filename:
                    continue
                runs = (
                    session.query(storage.Run)
                    .filter(storage.Run.report_id == report.id)
                    .all()
                )
                for run in runs:
                    status = (run.status or "").strip()
                    if status not in {"created", "running"}:
                        continue
                    liveness = derive_run_liveness(
                        run, progress=summarize_run_progress(run)
                    )
                    if liveness["blocks_duplicate_work"]:
                        return True
    return False


def _matching_delivery_ready_run_exists(patient_label: str, filenames: set[str]) -> bool:
    with storage.session_scope() as session:
        patients = storage.find_patients_by_label(session, patient_label)
        for patient in patients:
            for report in storage.list_reports(session, patient.id):
                if (report.filename or "") not in filenames:
                    continue
                runs = (
                    session.query(storage.Run)
                    .filter(storage.Run.report_id == report.id)
                    .all()
                )
                for run in runs:
                    artifacts = storage.list_artifacts(session, run.id)
                    if not run_downstream_delivery_gaps(
                        run,
                        progress=summarize_run_progress(run),
                        artifacts=artifacts,
                        require_final_draft=True,
                    ):
                        return True
    return False


def _matching_complete_run_exists(patient_label: str, filenames: set[str]) -> bool:
    return _matching_delivery_ready_run_exists(patient_label, filenames)


def completion_candidate_filenames(report: PortalReport) -> set[str]:
    if report.from_job or report.local_name:
        return {source_local_filename(report)}
    return {
        name
        for name in {
            source_local_filename(report),
            _safe_filename(report.original_name, "report.pdf"),
            _safe_filename(report.logical_name, "report.pdf"),
            _fallback_original_name_from_file_key(report.patient_id, report.file_key),
        }
        if name
    }


def active_report_filenames(patient_id: str, reports: list[PortalReport]) -> set[str]:
    return {
        source_local_filename(report)
        for report in reports
        if _matching_active_run_exists(patient_id, source_local_filename(report))
    }


def incomplete_report_filenames(patient_id: str, reports: list[PortalReport]) -> set[str]:
    return {
        source_local_filename(report)
        for report in reports
        if not _matching_complete_run_exists(
            patient_id, completion_candidate_filenames(report)
        )
    }


def merge_reports(
    *,
    job_reports: list[PortalReport],
    index_reports: list[PortalReport],
    file_reports: list[PortalReport] | None = None,
) -> list[PortalReport]:
    by_file_key: dict[str, PortalReport] = {}
    for report in [*(file_reports or []), *index_reports, *job_reports]:
        key = report.file_key or source_local_filename(report).lower()
        current = by_file_key.get(key)
        current_score = (
            bool(current and current.from_job),
            bool(current and current.document_kind == "report"),
            bool(current and current.uploaded_at),
        )
        report_score = (
            report.from_job,
            report.document_kind == "report",
            bool(report.uploaded_at),
        )
        if current is None or report_score > current_score:
            by_file_key[key] = report
    return sorted(
        by_file_key.values(),
        key=lambda report: (report.uploaded_at, report.file_key, report.from_job),
    )


def should_run_pipeline_for_patient(
    *,
    portal_dir: Path,
    patient_id: str,
    reports: list[PortalReport],
    downloaded: list[str],
) -> tuple[bool, str]:
    if not reports:
        return False, "no report PDFs in portal index"
    active = sorted(active_report_filenames(patient_id, reports))
    if active and len(active) == len(reports):
        return False, f"matching run already active for {', '.join(active)}"
    if active:
        reports = [
            report for report in reports if source_local_filename(report) not in active
        ]
    incomplete_reports = sorted(incomplete_report_filenames(patient_id, reports))
    if downloaded:
        return True, "downloaded missing report PDFs"
    if incomplete_reports:
        note = f"report PDFs without delivery-ready runs: {', '.join(incomplete_reports)}"
        if active:
            note += "; active report(s) skipped this cycle: " + ", ".join(active)
        return True, note
    patient_dir = portal_dir / patient_id
    if not analysis_artifacts_exist(patient_dir, patient_id):
        return True, "no local analysis artifacts"
    return False, "matching delivery-ready runs already exist for all report PDFs"


class NetlifyBlobClient:
    def __init__(self, *, netlify_bin: str, store: str, cwd: Path) -> None:
        self.netlify_bin = netlify_bin
        self.store = store
        self.cwd = cwd

    def _run(self, args: list[str], *, text: bool = True) -> subprocess.CompletedProcess:
        return subprocess.run(
            [self.netlify_bin, *args],
            cwd=str(self.cwd),
            capture_output=True,
            text=text,
            check=False,
        )

    def list_keys(self, prefix: str) -> list[str]:
        proc = self._run(["blobs:list", self.store, "--json", "--prefix", prefix])
        if proc.returncode != 0:
            raise RuntimeError((proc.stderr or proc.stdout or "").strip())
        return parse_blob_keys(proc.stdout)

    def get_json(self, key: str) -> dict[str, Any] | None:
        proc = self._run(["blobs:get", self.store, key])
        if proc.returncode != 0:
            return None
        try:
            parsed = json.loads(proc.stdout or "{}")
        except Exception:
            return None
        return parsed if isinstance(parsed, dict) else None

    def download(self, key: str, dest: Path) -> None:
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_name(f".{dest.name}.partial")
        tmp.unlink(missing_ok=True)
        proc = self._run(["blobs:get", self.store, key, "--output", str(tmp)])
        if proc.returncode != 0:
            tmp.unlink(missing_ok=True)
            raise RuntimeError((proc.stderr or proc.stdout or "").strip())
        tmp.replace(dest)

    def set_json(self, key: str, payload: dict[str, Any]) -> None:
        proc = self._run(
            ["blobs:set", self.store, key, json.dumps(payload, sort_keys=True), "--force"]
        )
        if proc.returncode != 0:
            raise RuntimeError((proc.stderr or proc.stdout or "").strip())

    def delete(self, key: str) -> None:
        proc = self._run(["blobs:delete", self.store, key, "--force"])
        if proc.returncode != 0:
            raise RuntimeError((proc.stderr or proc.stdout or "").strip())


def discover_patient_ids(
    client: NetlifyBlobClient, *, include_labels: set[str]
) -> list[str]:
    patient_keys = client.list_keys("patients/")
    ids = {
        patient_id
        for key in patient_keys
        for patient_id in [patient_id_from_meta_key(key)]
        if patient_id
    }
    ids.update(
        patient_id
        for key in patient_keys
        for patient_id in [patient_id_from_file_key(key)]
        if patient_id
    )
    ids.update(
        patient_id
        for key in client.list_keys(f"{JOB_PREFIX}/")
        for patient_id in [patient_id_from_job_key(key)]
        if patient_id
    )
    if include_labels:
        ids = {patient_id for patient_id in ids if patient_id.lower() in include_labels}
    return sorted(ids)


def should_discover_all_patients(
    *, once: bool, include_labels: set[str]
) -> bool:
    """Keep the continuous worker scoped to explicit upload job markers."""
    return bool(once or include_labels)


def paid_runs_are_authorized(
    *, include_labels: set[str], allow_paid_runs: bool
) -> bool:
    """A patient selection or explicit flag is the confirmation boundary."""
    return bool(include_labels or allow_paid_runs)


def load_job_reports_by_patient(
    client: NetlifyBlobClient, *, include_labels: set[str]
) -> dict[str, list[PortalReport]]:
    reports_by_patient: dict[str, list[PortalReport]] = {}
    for key in client.list_keys(f"{JOB_PREFIX}/"):
        patient_id = patient_id_from_job_key(key)
        if not patient_id:
            continue
        if include_labels and patient_id.lower() not in include_labels:
            continue
        payload = client.get_json(key) or {}
        reports = reports_from_job_payload(patient_id, payload)
        if reports:
            reports_by_patient.setdefault(patient_id, []).extend(reports)
    return reports_by_patient


NEW_PATIENT_UPLOAD_KIND = "new_patient_upload"
PENDING_UPLOAD_PREFIX = "uploads/pending"


def _identity_from_job(payload: dict[str, Any]) -> IdentityInput:
    """Read the hub's identity block into the shape both intake paths use."""
    identity = payload.get("identity") if isinstance(payload, dict) else None
    identity = identity if isinstance(identity, dict) else {}
    resolution = payload.get("resolution") if isinstance(payload, dict) else None
    resolution = resolution if isinstance(resolution, dict) else {}
    return IdentityInput(
        first_name=identity.get("firstName"),
        last_name=identity.get("lastName"),
        birthdate=identity.get("birthdate"),
        first_initial=identity.get("firstInitial"),
        last_initial=identity.get("lastInitial"),
        attach_to=resolution.get("attachTo"),
        force_new=bool(resolution.get("forceNew")),
    )


def load_new_patient_upload_jobs(
    client: NetlifyBlobClient,
) -> list[tuple[str, dict[str, Any]]]:
    """Job markers for uploads that have no patient yet.

    These cannot be keyed by patient id — allocating it is the job — so they are
    found by reading each marker rather than by parsing the key. Markers already
    parked for an operator answer are skipped without downloading anything.
    """
    jobs: list[tuple[str, dict[str, Any]]] = []
    for key in client.list_keys(f"{JOB_PREFIX}/"):
        payload = client.get_json(key) or {}
        if payload.get("kind") != NEW_PATIENT_UPLOAD_KIND:
            continue
        upload_id = str(payload.get("uploadId") or "").strip()
        answer = pipeline_uploads.pending_resolution(upload_id)
        if answer:
            # The operator answered a parked upload. Their resolution wins over
            # whatever the hub originally sent, and the job runs again.
            payload = {
                **payload,
                "resolution": {**(payload.get("resolution") or {}), **answer},
            }
        elif payload.get("status") == "needs_operator_answer":
            continue
        jobs.append((key, payload))
    return jobs


def _register_new_patient_report(
    *, portal_dir: Path, patient_label: str, filename: str, file_bytes: bytes
) -> tuple[str, bool]:
    """File the report under an existing patient, or report it already was.

    Returns the report id and whether this call created it. Re-entry after a
    crash finds the report by filename instead of registering a second one.
    """
    with storage.session_scope() as session:
        patient = next(
            iter(storage.find_patients_by_label(session, patient_label)), None
        )
        if patient is None:
            raise RuntimeError(f"{patient_label} disappeared before registration")
        patient_uuid = patient.id
        existing = next(
            (
                report
                for report in storage.list_reports(session, patient_uuid)
                if (report.filename or "") == filename
            ),
            None,
        )
        if existing is not None:
            return existing.id, False

    report_id = str(uuid.uuid4())
    try:
        original_path, extracted_path, mime_type, _preview = save_report_upload(
            patient_id=patient_uuid,
            report_id=report_id,
            filename=filename,
            provided_mime_type="application/pdf",
            file_bytes=file_bytes,
        )
        with storage.session_scope() as session:
            storage.create_report(
                session,
                report_id=report_id,
                patient_id=patient_uuid,
                filename=filename,
                mime_type=mime_type,
                stored_path=original_path,
                extracted_text_path=extracted_path,
            )
    except Exception:
        # A retry mints a fresh report id, so the extracted files this attempt
        # wrote would sit there unreferenced forever. Take them with it.
        shutil.rmtree(report_dir(patient_uuid, report_id), ignore_errors=True)
        raise
    # The raw-sync watcher publishes whatever is in the patient folder, so
    # dropping the file there is the whole publication step.
    portal_patient_dir = portal_dir / patient_label
    portal_patient_dir.mkdir(parents=True, exist_ok=True)
    portal_path = portal_patient_dir / _safe_filename(filename, "report.pdf")
    tmp_path = portal_path.with_name(f".{portal_path.name}.partial")
    tmp_path.write_bytes(file_bytes)
    tmp_path.replace(portal_path)
    return report_id, True


def _allocate_patient_for_upload(identity: IdentityInput) -> str:
    """Create or find this person's chart and return their clinic id."""
    with storage.session_scope() as session:
        existing, keep_stored_name = find_patient_by_identity(session, identity)
        if existing is not None and keep_stored_name:
            return existing.label

        first_initial, last_initial, birthdate = identity_key(identity)
        canonical = allocate_canonical_patient_id(
            session,
            first_initial=first_initial,
            last_initial=last_initial,
            birthdate=birthdate,
            exclude_patient_uuid=existing.id if existing else None,
        )
        if existing is not None:
            storage.update_patient(
                session,
                existing.id,
                label=canonical,
                birthdate=birthdate,
                first_name=(identity.first_name or "").strip() or None,
                last_name=(identity.last_name or "").strip() or None,
                first_initial=first_initial,
                last_initial=last_initial,
            )
        else:
            storage.create_patient(
                session,
                label=canonical,
                notes="",
                birthdate=birthdate,
                first_name=(identity.first_name or "").strip(),
                last_name=(identity.last_name or "").strip(),
                first_initial=first_initial,
                last_initial=last_initial,
            )
        return canonical


def _delete_quietly(client: NetlifyBlobClient, key: str) -> str | None:
    """Drop a blob we are finished with, reporting rather than raising.

    The work these blobs describe is already durable by the time we get here, so
    a delete that fails — or one that already happened on an earlier attempt —
    must not undo it or stop the rest of the submission.
    """
    try:
        client.delete(key)
    except Exception as exc:
        return f"{key}: {str(exc).strip() or exc.__class__.__name__}"
    return None


def _file_remaining_submission_files(
    *,
    client: NetlifyBlobClient,
    portal_dir: Path,
    upload_id: str,
    patient_label: str,
    report_file_key: str,
) -> tuple[list[str], list[str], list[str]]:
    """File everything else the clinic sent with the report.

    One submission can carry intake forms, prior reports, anything. Nothing is
    deleted here: the whole submission's blobs stay put until every file has
    landed, so a failure part way through leaves a retry something to re-read.
    Re-filing is safe — ``register_patient_file`` reuses the row it already
    made for that filename.

    Returns the filenames filed, the keys they came from, and the failures.
    """
    prefix = f"{PENDING_UPLOAD_PREFIX}/{upload_id}/"
    try:
        keys = sorted(client.list_keys(prefix))
    except Exception as exc:
        return [], [], [f"listing the rest of the submission failed: {exc}"]

    filed: list[str] = []
    filed_keys: list[str] = []
    failures: list[str] = []
    for key in keys:
        relative = key[len(prefix) :]
        # The job names the report by its full blob key, so that is what it is
        # compared against. Comparing the relative name would let the report
        # through here and file it a second time as an attachment.
        if not relative or key == report_file_key:
            continue
        filename = _safe_filename(Path(relative).name, "attachment")
        try:
            with tempfile.TemporaryDirectory(prefix="qeeg-pending-extra-") as scratch:
                local_path = Path(scratch) / filename
                client.download(key, local_path)
                register_patient_file(
                    patient_label=patient_label,
                    src_path=local_path,
                    filename=filename,
                    mime_type=(
                        mimetypes.guess_type(filename)[0] or "application/octet-stream"
                    ),
                )
                # Same publication step as the report: the raw-sync watcher
                # sends on whatever is sitting in the patient folder.
                portal_patient_dir = portal_dir / patient_label
                portal_patient_dir.mkdir(parents=True, exist_ok=True)
                portal_path = portal_patient_dir / filename
                tmp_path = portal_path.with_name(f".{portal_path.name}.partial")
                tmp_path.write_bytes(local_path.read_bytes())
                tmp_path.replace(portal_path)
        except Exception as exc:
            failures.append(
                f"{relative}: {str(exc).strip() or exc.__class__.__name__}"
            )
            continue
        filed.append(filename)
        filed_keys.append(key)
    return filed, filed_keys, failures


def process_new_patient_upload(
    *,
    client: NetlifyBlobClient,
    portal_dir: Path,
    status_dir: Path,
    job_key: str,
    payload: dict[str, Any],
) -> PatientWorkerResult:
    """Give a hub upload a chart and a report. No paid analysis happens here.

    Creating the patient and registering the report are free, so they run
    without the paid-run approval flag. Analysis for the new patient stays
    behind the existing gate and is picked up by the normal per-patient pass.
    """
    upload_id = str(payload.get("uploadId") or "").strip()
    file_key = str(payload.get("fileKey") or "").strip()
    identity = _identity_from_job(payload)

    if not file_key or not pipeline_uploads.is_valid_upload_id(upload_id):
        return PatientWorkerResult(
            patient_id=upload_id or "unknown-upload",
            status="failed",
            downloaded=[],
            report_count=0,
            ran_batch=False,
            note="upload job is missing a usable uploadId or fileKey",
        )

    # The hub stores the file and names it by its full blob key. Anything else
    # is a job this worker cannot honour, and saying so beats reaching for a
    # key that was never going to exist.
    expected_prefix = f"{PENDING_UPLOAD_PREFIX}/{upload_id}/"
    if not file_key.startswith(expected_prefix):
        return PatientWorkerResult(
            patient_id=upload_id,
            status="failed",
            downloaded=[],
            report_count=0,
            ran_batch=False,
            note=(
                f"upload job names {file_key!r}, which is not under "
                f"{expected_prefix!r}"
            ),
        )

    identity_payload = payload.get("identity")
    try:
        return _file_new_patient_upload(
            client=client,
            portal_dir=portal_dir,
            status_dir=status_dir,
            job_key=job_key,
            payload=payload,
            upload_id=upload_id,
            file_key=file_key,
            identity=identity,
            identity_payload=(
                identity_payload if isinstance(identity_payload, dict) else {}
            ),
        )
    except Exception as exc:
        # One upload must never take the cycle down with it: the rest of the
        # queue and the per-patient pass still have to run. The store is remote
        # and the disk is not ours, so this is where a subprocess failure, a
        # write error, or anything else unforeseen stops.
        note = str(exc).strip() or exc.__class__.__name__
        pipeline_uploads.record_failed(upload_id=upload_id, error=note)
        result = PatientWorkerResult(
            patient_id=upload_id,
            status="failed",
            downloaded=[],
            report_count=0,
            ran_batch=False,
            note=note,
        )
        _write_local_status(status_dir, result)
        return result


def _file_new_patient_upload(
    *,
    client: NetlifyBlobClient,
    portal_dir: Path,
    status_dir: Path,
    job_key: str,
    payload: dict[str, Any],
    upload_id: str,
    file_key: str,
    identity: IdentityInput,
    identity_payload: dict[str, Any],
) -> PatientWorkerResult:
    """The upload's happy path. Its caller owns every failure."""
    pipeline_uploads.record_seen(upload_id=upload_id, identity=identity_payload)

    try:
        patient_label = _allocate_patient_for_upload(identity)
    except IdentityNameConflict as exc:
        # Park it. Nothing is downloaded and the marker is kept, so the operator
        # can answer with attachTo or forceNew and the next cycle picks it up.
        client.set_json(
            job_key,
            {**payload, "status": "needs_operator_answer", "conflict": exc.payload},
        )
        pipeline_uploads.record_parked(
            upload_id=upload_id,
            identity=identity_payload,
            conflict=exc.payload,
        )
        result = PatientWorkerResult(
            patient_id=upload_id,
            status="needs_operator_answer",
            downloaded=[],
            report_count=0,
            ran_batch=False,
            note=exc.payload["detail"],
        )
        _write_local_status(status_dir, result)
        return result
    except PatientIdentityError as exc:
        result = PatientWorkerResult(
            patient_id=upload_id,
            status="failed",
            downloaded=[],
            report_count=0,
            ran_batch=False,
            note=str(exc),
        )
        _write_local_status(status_dir, result)
        return result

    filename = _safe_filename(Path(file_key).name, "report.pdf")
    with tempfile.TemporaryDirectory(prefix="qeeg-pending-upload-") as scratch:
        local_path = Path(scratch) / filename
        # `fileKey` is the full blob key the hub stored the file under, not a
        # bare filename. Prepending the prefix again looked for
        # `uploads/pending/<id>/uploads/pending/<id>/<name>`, which is nothing.
        client.download(file_key, local_path)
        file_bytes = local_path.read_bytes()

    _report_id, registered = _register_new_patient_report(
        portal_dir=portal_dir,
        patient_label=patient_label,
        filename=filename,
        file_bytes=file_bytes,
    )

    # The rest of the submission goes in before the marker does, because the
    # marker is what brings us back here if any of it fails.
    cleanup_failures: list[str] = []
    extras, extra_keys, extra_failures = _file_remaining_submission_files(
        client=client,
        portal_dir=portal_dir,
        upload_id=upload_id,
        patient_label=patient_label,
        report_file_key=file_key,
    )
    cleanup_failures.extend(extra_failures)

    if not extra_failures:
        # Nothing under the pending prefix is dropped until the whole
        # submission has landed. Deleting the report first meant one failed
        # attachment left the next cycle re-downloading a blob that was already
        # gone — an exception every cycle, forever, while the upload record
        # said failed even though the report was registered.
        for key in [file_key, *extra_keys, job_key]:
            drop_failure = _delete_quietly(client, key)
            if drop_failure:
                cleanup_failures.append(drop_failure)
        pipeline_uploads.record_registered(
            upload_id=upload_id, patient_id=patient_label
        )

    note = f"new patient upload {upload_id} filed under {patient_label}"
    if extras:
        note += f"; also filed {', '.join(extras)}"
    if cleanup_failures:
        note += "; left for retry: " + "; ".join(cleanup_failures)

    result = PatientWorkerResult(
        patient_id=patient_label,
        status="registered" if registered else "already_registered",
        downloaded=[str(portal_dir / patient_label / filename)],
        report_count=1,
        ran_batch=False,
        note=note,
    )
    _write_local_status(status_dir, result)
    return result


def download_missing_reports(
    *, client: NetlifyBlobClient, portal_dir: Path, reports: list[PortalReport]
) -> list[str]:
    downloaded: list[str] = []
    for report in reports:
        dest = local_report_path(portal_dir, report)
        if dest.exists() and (report.size <= 0 or dest.stat().st_size == report.size):
            continue
        blob_key = f"patients/{report.patient_id}/files/{report.file_key}"
        client.download(blob_key, dest)
        downloaded.append(str(dest))
    return downloaded


def missing_report_paths(*, portal_dir: Path, reports: list[PortalReport]) -> list[str]:
    missing: list[str] = []
    for report in reports:
        dest = local_report_path(portal_dir, report)
        if dest.exists() and (report.size <= 0 or dest.stat().st_size == report.size):
            continue
        missing.append(str(dest))
    return missing


def _stage_reports_for_batch(reports: list[PortalReport], portal_dir: Path) -> Path:
    temp_root = Path(tempfile.mkdtemp(prefix="qeeg-portal-job-"))
    patient_dir = temp_root / reports[0].patient_id
    patient_dir.mkdir(parents=True, exist_ok=True)
    for report in reports:
        src = local_report_path(portal_dir, report)
        dest = patient_dir / source_local_filename(report)
        try:
            os.link(src, dest)
        except Exception:
            dest.write_bytes(src.read_bytes())
    return temp_root


def run_batch_for_patient(
    patient_id: str,
    *,
    portal_dir: Path | None = None,
    force: bool = False,
    reextract_existing: bool = False,
) -> subprocess.CompletedProcess:
    cmd = [
        sys.executable,
        str(_REPO_ROOT / "scripts" / "run_portal_council_batch.py"),
        "--include-label",
        patient_id,
    ]
    if portal_dir is not None:
        cmd.extend(["--portal-dir", str(portal_dir)])
    if force:
        cmd.append("--force")
    if reextract_existing:
        cmd.append("--reextract-existing")
    return subprocess.run(
        cmd,
        cwd=str(_REPO_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )


def _write_local_status(status_dir: Path, result: PatientWorkerResult) -> None:
    status_dir.mkdir(parents=True, exist_ok=True)
    path = status_dir / f"{result.patient_id}.json"
    tmp = path.with_name(f".{path.name}.partial")
    tmp.write_text(json.dumps(asdict(result), indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)


def process_patient(
    *,
    client: NetlifyBlobClient,
    portal_dir: Path,
    status_dir: Path,
    patient_id: str,
    job_reports: list[PortalReport],
    dry_run: bool,
    allow_paid_runs: bool = False,
) -> PatientWorkerResult:
    downloaded: list[str] = []
    reports: list[PortalReport] = []
    temp_batch_dir: Path | None = None
    try:
        if not dry_run:
            try:
                remote_meta = client.get_json(f"patients/{patient_id}/{META_NAME}")
            except Exception:
                remote_meta = None
            sync_remote_patient_identity(
                portal_dir=portal_dir,
                patient_id=patient_id,
                remote_meta=remote_meta,
            )
        index = client.get_json(f"patients/{patient_id}/{INDEX_NAME}") or {}
        index_reports = reports_from_index(patient_id, index)
        file_reports = reports_from_file_keys(
            patient_id, client.list_keys(f"patients/{patient_id}/files/")
        )
        reports = merge_reports(
            job_reports=job_reports,
            index_reports=index_reports,
            file_reports=file_reports,
        )
        active_filenames = active_report_filenames(patient_id, reports)
        runnable_reports = [
            report
            for report in reports
            if source_local_filename(report) not in active_filenames
        ]
        would_download = missing_report_paths(
            portal_dir=portal_dir, reports=runnable_reports
        )
        if not dry_run:
            downloaded = download_missing_reports(
                client=client, portal_dir=portal_dir, reports=runnable_reports
            )
        if runnable_reports:
            should_run, note = should_run_pipeline_for_patient(
                portal_dir=portal_dir,
                patient_id=patient_id,
                reports=runnable_reports,
                downloaded=would_download if dry_run else downloaded,
            )
        else:
            should_run = False
            note = "matching run already active for " + ", ".join(
                sorted(active_filenames)
            )
        if dry_run and would_download and note == "downloaded missing report PDFs":
            note = "would download missing report PDFs"
        if active_filenames and runnable_reports:
            note = (
                f"{note}; active report(s) skipped this cycle: "
                + ", ".join(sorted(active_filenames))
            )
        if dry_run:
            result = PatientWorkerResult(
                patient_id=patient_id,
                status="dry_run_run" if should_run else "dry_run_skip",
                downloaded=[],
                would_download=would_download,
                report_count=len(reports),
                ran_batch=False,
                note=note,
            )
        elif should_run and not allow_paid_runs:
            approval_note = "paid analysis requires explicit approval"
            result = PatientWorkerResult(
                patient_id=patient_id,
                status="awaiting_confirmation",
                downloaded=downloaded,
                report_count=len(reports),
                ran_batch=False,
                note=f"{note}; {approval_note}" if note else approval_note,
            )
        elif should_run:
            running_payload = {
                "schemaVersion": 1,
                "patientId": patient_id,
                "status": "running",
                "updatedAt": _now_ms(),
                "note": note,
                "reports": [asdict(report) for report in runnable_reports],
            }
            client.set_json(f"{STATUS_PREFIX}/{patient_id}.json", running_payload)
            has_job_report = any(report.from_job for report in runnable_reports)
            if has_job_report or active_filenames:
                temp_batch_dir = _stage_reports_for_batch(runnable_reports, portal_dir)
            proc = run_batch_for_patient(
                patient_id,
                portal_dir=temp_batch_dir,
                force=has_job_report,
                reextract_existing=has_job_report,
            )
            status = "complete" if proc.returncode == 0 else "failed"
            result = PatientWorkerResult(
                patient_id=patient_id,
                status=status,
                downloaded=downloaded,
                report_count=len(reports),
                ran_batch=True,
                returncode=proc.returncode,
                stdout_tail=(proc.stdout or "")[-4000:],
                stderr_tail=(proc.stderr or "")[-4000:],
                note=note,
            )
        else:
            result = PatientWorkerResult(
                patient_id=patient_id,
                status="skipped",
                downloaded=downloaded,
                report_count=len(reports),
                ran_batch=False,
                note=note,
            )
    except Exception as exc:
        result = PatientWorkerResult(
            patient_id=patient_id,
            status="failed",
            downloaded=downloaded,
            report_count=len(reports),
            ran_batch=False,
            note=str(exc),
        )
    finally:
        if temp_batch_dir is not None:
            shutil.rmtree(temp_batch_dir, ignore_errors=True)

    if not dry_run:
        remote_status_error = ""
        try:
            client.set_json(
                f"{STATUS_PREFIX}/{patient_id}.json",
                {
                    "schemaVersion": 1,
                    "patientId": patient_id,
                    "status": result.status,
                    "updatedAt": _now_ms(),
                    "reportCount": result.report_count,
                    "ranBatch": result.ran_batch,
                    "returncode": result.returncode,
                    "note": result.note,
                    "stdoutTail": result.stdout_tail,
                    "stderrTail": result.stderr_tail,
                },
            )
        except Exception as exc:
            remote_status_error = str(exc).strip() or exc.__class__.__name__
        if remote_status_error:
            warning = f"remote status publish failed: {remote_status_error}"
            result.note = f"{result.note}; {warning}" if result.note else warning
        _write_local_status(status_dir, result)
    return result


def _lock_path(status_dir: Path) -> Path:
    return status_dir / ".portal_pipeline_worker.lock"


def _write_worker_failure(status_dir: Path, message: str) -> None:
    status_dir.mkdir(parents=True, exist_ok=True)
    path = status_dir / "_worker.json"
    tmp = path.with_name(f".{path.name}.partial")
    tmp.write_text(
        json.dumps(
            {
                "status": "failed",
                "updatedAt": _now_ms(),
                "note": message,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    tmp.replace(path)


def _acquire_lock(lock_file):
    import fcntl

    fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)


def main() -> int:
    parser = argparse.ArgumentParser(description="Download clinic portal report PDFs and run qEEG Council jobs.")
    parser.add_argument("--once", action="store_true", help="Run one audit pass and exit.")
    parser.add_argument("--dry-run", action="store_true", help="Plan work without downloading or running jobs.")
    parser.add_argument("--include-label", action="append", default=[], help="Only process this patient label. Repeatable.")
    parser.add_argument(
        "--allow-paid-runs",
        action="store_true",
        help="Allow paid analysis for discovered upload jobs. A selected patient label also authorizes that patient.",
    )
    parser.add_argument("--poll-seconds", type=float, default=float(os.getenv("QEEG_PORTAL_PIPELINE_POLL_S", "60") or "60"))
    parser.add_argument("--store", default=os.getenv("QEEG_BLOBS_STORE", "qeeg-portal"))
    parser.add_argument("--netlify-bin", default=os.getenv("NETLIFY_BIN", "netlify"))
    parser.add_argument("--thrylen-repo", default=os.getenv("QEEG_PORTAL_SYNC_REPO", str(_REPO_ROOT.parent / "thrylen")))
    parser.add_argument("--portal-dir", default=os.getenv("QEEG_PORTAL_PATIENTS_DIR", str(_REPO_ROOT / "data" / "portal_patients")))
    parser.add_argument("--status-dir", default=str(_REPO_ROOT / "data" / "pipeline_jobs"))
    args = parser.parse_args()

    if args.poll_seconds <= 0:
        args.poll_seconds = 60.0

    portal_dir = Path(args.portal_dir).expanduser()
    status_dir = Path(args.status_dir).expanduser()
    status_dir.mkdir(parents=True, exist_ok=True)
    unusable_labels = [
        str(label).strip()
        for label in args.include_label
        if not is_valid_patient_id(str(label).strip())
    ]
    if unusable_labels:
        # Dropping these silently would empty include_labels and turn a
        # single-patient audit into full-portal discovery. A clinic id is
        # case-strict, which it never used to be, so say so.
        print(
            "Not clinic patient ids: "
            + ", ".join(unusable_labels)
            + ". Use XX_MM-DD-YYYY, matching case.",
            file=sys.stderr,
        )
        return 2
    include_labels = {str(label).strip().lower() for label in args.include_label}
    allow_paid_runs = paid_runs_are_authorized(
        include_labels=include_labels,
        allow_paid_runs=args.allow_paid_runs,
    )

    storage.init_db()
    client = NetlifyBlobClient(netlify_bin=args.netlify_bin, store=args.store, cwd=Path(args.thrylen_repo).expanduser())

    with _lock_path(status_dir).open("a+", encoding="utf-8") as lock_file:
        try:
            _acquire_lock(lock_file)
        except OSError:
            print("portal pipeline worker is already running", file=sys.stderr)
            return 2

        while True:
            try:
                new_patient_jobs = (
                    [] if args.dry_run else load_new_patient_upload_jobs(client)
                )
                job_reports_by_patient = load_job_reports_by_patient(
                    client, include_labels=include_labels
                )
                if should_discover_all_patients(
                    once=args.once, include_labels=include_labels
                ):
                    labels = discover_patient_ids(
                        client, include_labels=include_labels
                    )
                    labels = sorted(set(labels) | set(job_reports_by_patient))
                else:
                    labels = sorted(job_reports_by_patient)
            except Exception as exc:
                message = f"portal discovery failed: {exc}"
                _write_worker_failure(status_dir, message)
                print(message, file=sys.stderr, flush=True)
                if args.once:
                    return 1
                time.sleep(args.poll_seconds)
                continue
            print(f"Portal pipeline worker found {len(labels)} patient label(s).", flush=True)
            failures = 0
            for job_key, payload in new_patient_jobs:
                # Giving a hub upload its chart is free, so it runs regardless
                # of the paid-run gate. Analysis waits for the pass below.
                upload_result = process_new_patient_upload(
                    client=client,
                    portal_dir=portal_dir,
                    status_dir=status_dir,
                    job_key=job_key,
                    payload=payload,
                )
                if upload_result.status == "failed":
                    failures += 1
                print(
                    f"- {upload_result.patient_id}: {upload_result.status} "
                    f"({upload_result.note})",
                    flush=True,
                )
            for patient_id in labels:
                result = process_patient(
                    client=client,
                    portal_dir=portal_dir,
                    status_dir=status_dir,
                    patient_id=patient_id,
                    job_reports=job_reports_by_patient.get(patient_id, []),
                    dry_run=args.dry_run,
                    allow_paid_runs=allow_paid_runs,
                )
                if result.status == "failed":
                    failures += 1
                print(f"- {patient_id}: {result.status} ({result.note})", flush=True)
            if args.once:
                return 1 if failures else 0
            time.sleep(args.poll_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
