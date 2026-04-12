from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from . import config as cfg
from . import storage


_PORTAL_PATIENT_ID_RE = re.compile(
    r"^(?P<mm>\d{2})-(?P<dd>\d{2})-(?P<yyyy>\d{4})-(?P<n>\d+)$"
)

_GENERATED_ANALYSIS_PATTERNS = (
    "__patient-facing__",
    "__single-agent",
    "__analysis__",
    "__patient-facing.pdf",
)

_STAGE_LABELS = {
    1: "Initial Analysis",
    2: "Peer Review",
    3: "Revision",
    4: "Consolidation",
    5: "Final Review",
    6: "Final Draft",
}


def normalize_portal_patient_id(value: str) -> str | None:
    raw = (value or "").strip()
    match = _PORTAL_PATIENT_ID_RE.match(raw)
    if not match:
        return None
    mm = int(match.group("mm"))
    dd = int(match.group("dd"))
    yyyy = int(match.group("yyyy"))
    n = int(match.group("n"))
    if not (1 <= mm <= 12 and 1 <= dd <= 31 and 1900 <= yyyy <= 2100 and 0 <= n <= 999):
        return None
    return f"{mm:02d}-{dd:02d}-{yyyy:04d}-{n}"


def portal_patients_dir() -> Path:
    configured = (os.getenv("QEEG_PORTAL_PATIENTS_DIR") or "").strip()
    if configured:
        return Path(configured).expanduser()
    return cfg.DATA_DIR / "portal_patients"


def portal_sync_state_path() -> Path:
    return portal_patients_dir() / ".qeeg_portal_sync_state.json"


def pipeline_job_status_dir() -> Path:
    return cfg.DATA_DIR / "pipeline_jobs"


def cathode_projects_dir() -> Path:
    configured = (os.getenv("QEEG_CATHODE_PROJECTS_DIR") or "").strip()
    if configured:
        return Path(configured).expanduser()
    return Path(__file__).resolve().parents[2] / "cathode" / "projects"


def cathode_project_dir(patient_label: str) -> Path:
    return cathode_projects_dir() / patient_label


def cathode_handoff_payload_path(patient_label: str) -> Path:
    return cathode_project_dir(patient_label) / "qeeg_handoff_payload.json"


def cathode_handoff_source_path(patient_label: str) -> Path:
    return cathode_project_dir(patient_label) / "qeeg_council_source.md"


def cathode_handoff_meta_path(patient_label: str) -> Path:
    return cathode_project_dir(patient_label) / "qeeg_handoff.json"


def progress_jsonl_path(run_id: str) -> Path:
    return cfg.ARTIFACTS_DIR / run_id / "progress.jsonl"


def progress_log_path(run_id: str) -> Path:
    return cfg.ARTIFACTS_DIR / run_id / "progress.log"


def isoformat_or_none(value: datetime | None) -> str | None:
    return value.isoformat() if isinstance(value, datetime) else None


def _path_iso(path: Path | None) -> str | None:
    if path is None or not path.exists():
        return None
    return datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).isoformat()


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def _read_sync_state(patient_label: str) -> dict[str, Any] | None:
    payload = _read_json(portal_sync_state_path())
    if not payload:
        return None
    patients = payload.get("patients")
    if not isinstance(patients, dict):
        return None
    entry = patients.get(patient_label)
    return entry if isinstance(entry, dict) else None


def _latest_progress_payload(run_id: str) -> dict[str, Any] | None:
    payloads = _recent_progress_payloads(run_id)
    return payloads[-1] if payloads else None


def _recent_progress_payloads(run_id: str) -> list[dict[str, Any]]:
    path = progress_jsonl_path(run_id)
    if not path.exists():
        return []
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except Exception:
        return []
    payloads: list[dict[str, Any]] = []
    for line in reversed(lines[-200:]):
        line = line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except Exception:
            continue
        if isinstance(payload, dict):
            payloads.append(payload)
    payloads.reverse()
    return payloads


def summarize_run_progress(run: storage.Run) -> dict[str, Any]:
    payloads = _recent_progress_payloads(run.id)
    latest = payloads[-1] if payloads else None
    summary: dict[str, Any] = {
        "status": run.status,
        "phase_label": {
            "created": "Queued",
            "running": "Running",
            "complete": "Complete",
            "failed": "Failed",
            "needs_auth": "Needs Auth",
        }.get(run.status, run.status.title()),
        "determinate": False,
        "percent": 100.0 if run.status == "complete" else None,
        "stage_num": None,
        "stage_name": None,
        "task": None,
        "task_label": None,
        "model_id": None,
        "chunk_index": None,
        "chunk_count": None,
        "elapsed_s": None,
        "heartbeat_count": None,
        "timestamp": isoformat_or_none(run.completed_at if run.status == "complete" else run.started_at),
        "log_path": str(progress_log_path(run.id)),
        "progress_jsonl_path": str(progress_jsonl_path(run.id)),
    }
    if not latest:
        return summary

    latest_counts = next(
        (
            payload
            for payload in reversed(payloads)
            if isinstance(payload.get("requested_count"), int)
            and payload.get("requested_count", 0) > 0
            and isinstance(payload.get("success_count"), int)
        ),
        None,
    )
    latest_stage_payload = next(
        (
            payload
            for payload in reversed(payloads)
            if isinstance(payload.get("stage_num"), int)
            or isinstance(payload.get("stage_name"), str)
        ),
        None,
    )

    stage_num = (
        latest.get("stage_num")
        if latest.get("stage_num") is not None
        else (latest_stage_payload.get("stage_num") if isinstance(latest_stage_payload, dict) else None)
    )
    stage_name = (
        latest.get("stage_name")
        if latest.get("stage_name") is not None
        else (latest_stage_payload.get("stage_name") if isinstance(latest_stage_payload, dict) else None)
    )
    task = latest.get("task")
    model_id = latest.get("model_id")
    status = latest.get("status")
    chunk_index = latest.get("chunk_index")
    chunk_count = latest.get("chunk_count")
    elapsed_s = latest.get("elapsed_s")
    heartbeat_count = latest.get("heartbeat_count")
    timestamp = latest.get("timestamp")
    requested_count = (
        latest_counts.get("requested_count") if isinstance(latest_counts, dict) else latest.get("requested_count")
    )
    success_count = (
        latest_counts.get("success_count") if isinstance(latest_counts, dict) else latest.get("success_count")
    )

    summary.update(
        {
            "status": status or run.status,
            "stage_num": stage_num if isinstance(stage_num, int) else None,
            "stage_name": stage_name if isinstance(stage_name, str) else None,
            "task": task if isinstance(task, str) else None,
            "model_id": model_id if isinstance(model_id, str) else None,
            "chunk_index": chunk_index if isinstance(chunk_index, int) else None,
            "chunk_count": chunk_count if isinstance(chunk_count, int) else None,
            "elapsed_s": elapsed_s if isinstance(elapsed_s, int) else None,
            "elapsed_seconds": elapsed_s if isinstance(elapsed_s, int) else None,
            "heartbeat_count": heartbeat_count if isinstance(heartbeat_count, int) else None,
            "timestamp": timestamp if isinstance(timestamp, str) else summary["timestamp"],
            "requested_count": requested_count if isinstance(requested_count, int) else None,
            "success_count": success_count if isinstance(success_count, int) else None,
            "raw": latest,
        }
    )

    stage_label = _STAGE_LABELS.get(summary["stage_num"], summary["stage_name"] or "Run")
    task_bits = [bit for bit in [summary["task"], summary["model_id"]] if isinstance(bit, str) and bit]
    if isinstance(summary["chunk_index"], int) and isinstance(summary["chunk_count"], int) and summary["chunk_count"] > 0:
        task_bits.append(f"chunk {summary['chunk_index']}/{summary['chunk_count']}")
    partial_success = (
        isinstance(success_count, int)
        and isinstance(requested_count, int)
        and requested_count > 0
        and success_count < requested_count
    )
    if partial_success:
        task_bits.append(f"partial {success_count}/{requested_count}")
    summary["task_label"] = " · ".join(task_bits) if task_bits else None
    summary["phase_label"] = (
        f"{stage_label}"
        + (f" · {summary['task_label']}" if summary["task_label"] else "")
        + (f" · {summary['status']}" if isinstance(summary["status"], str) and summary["status"] not in {"start", "running"} else "")
    )

    percent: float | None = None
    determinate = False
    if run.status == "complete":
        percent = 100.0
        determinate = True
    elif isinstance(summary["stage_num"], int):
        completed_stages = max(0, min(summary["stage_num"] - 1, 6))
        if isinstance(summary["chunk_index"], int) and isinstance(summary["chunk_count"], int) and summary["chunk_count"] > 0:
            sub = min(max(summary["chunk_index"] / summary["chunk_count"], 0.0), 1.0)
            percent = round(((completed_stages + sub) / 6.0) * 100.0, 1)
            determinate = True
        elif (
            isinstance(requested_count, int)
            and requested_count > 0
            and isinstance(success_count, int)
        ):
            sub = min(max(success_count / requested_count, 0.0), 1.0)
            percent = round(((completed_stages + sub) / 6.0) * 100.0, 1)
            determinate = True

    summary["percent"] = percent
    summary["determinate"] = determinate
    summary["partial_success"] = partial_success
    summary["current_stage_num"] = summary["stage_num"]
    summary["current_stage_name"] = summary["stage_name"]
    summary["current_task"] = summary["task"]
    summary["current_model_id"] = summary["model_id"]
    return summary


def _looks_generated_pdf(patient_label: str, path: Path) -> bool:
    lower_name = path.name.lower()
    lower_label = patient_label.lower()
    if not lower_name.endswith(".pdf"):
        return True
    if lower_name in {f"{lower_label}.pdf", "main.pdf"}:
        return True
    if any(token in lower_name for token in _GENERATED_ANALYSIS_PATTERNS):
        return True
    if lower_name.endswith("_analysis_pdf.pdf"):
        return True
    return False


def classify_portal_files(patient_label: str) -> dict[str, Any]:
    patient_dir = portal_patients_dir() / patient_label
    info: dict[str, Any] = {
        "patient_dir": str(patient_dir),
        "exists": patient_dir.exists(),
        "source_reports": [],
        "patient_facing_pdfs": [],
        "final_exports": [],
        "council_run_ids": [],
        "latest_source_report": None,
        "latest_patient_facing_pdf": None,
        "latest_export": None,
        "last_modified": _path_iso(patient_dir) if patient_dir.exists() else None,
    }
    if not patient_dir.exists():
        return info

    source_reports: list[dict[str, Any]] = []
    patient_facing_pdfs: list[dict[str, Any]] = []
    final_exports: list[dict[str, Any]] = []

    pdf_paths = sorted(
        patient_dir.glob("*.pdf"),
        key=lambda candidate: candidate.stat().st_mtime,
        reverse=True,
    )
    for path in pdf_paths:
        row = {"name": path.name, "path": str(path), "mtime": _path_iso(path)}
        if "__patient-facing__" in path.name.lower():
            patient_facing_pdfs.append(row)
        elif not _looks_generated_pdf(patient_label, path):
            source_reports.append(row)
        else:
            final_exports.append(row)

    md_paths = sorted(
        patient_dir.glob("*.md"),
        key=lambda candidate: candidate.stat().st_mtime,
        reverse=True,
    )
    for path in md_paths:
        final_exports.append({"name": path.name, "path": str(path), "mtime": _path_iso(path)})

    council_dir = patient_dir / "council"
    council_run_ids: list[str] = []
    if council_dir.exists():
        council_run_ids = [p.name for p in sorted(council_dir.iterdir()) if p.is_dir()]

    info.update(
        {
            "source_reports": source_reports,
            "patient_facing_pdfs": patient_facing_pdfs,
            "final_exports": final_exports,
            "council_run_ids": council_run_ids,
            "latest_source_report": source_reports[0] if source_reports else None,
            "latest_patient_facing_pdf": patient_facing_pdfs[0] if patient_facing_pdfs else None,
            "latest_export": final_exports[0] if final_exports else None,
        }
    )
    return info


def load_pipeline_job_status(patient_label: str) -> dict[str, Any] | None:
    path = pipeline_job_status_dir() / f"{patient_label}.json"
    payload = _read_json(path)
    if payload is None:
        return None
    payload = dict(payload)
    payload["status_path"] = str(path)
    payload["updated_at"] = _path_iso(path)
    return payload


def _artifact_out(artifact: storage.Artifact) -> dict[str, Any]:
    return {
        "id": artifact.id,
        "run_id": artifact.run_id,
        "stage_num": artifact.stage_num,
        "stage_name": artifact.stage_name,
        "model_id": artifact.model_id,
        "kind": artifact.kind,
        "content_path": artifact.content_path,
        "content_type": artifact.content_type,
        "created_at": artifact.created_at.isoformat(),
    }


def _run_out(run: storage.Run) -> dict[str, Any]:
    try:
        council_model_ids = json.loads(run.council_model_ids_json)
    except Exception:
        council_model_ids = []
    try:
        label_map = json.loads(run.label_map_json or "{}")
    except Exception:
        label_map = {}
    return {
        "id": run.id,
        "patient_id": run.patient_id,
        "report_id": run.report_id,
        "status": run.status,
        "error_message": run.error_message,
        "council_model_ids": council_model_ids,
        "consolidator_model_id": run.consolidator_model_id,
        "label_map": label_map,
        "started_at": isoformat_or_none(run.started_at),
        "completed_at": isoformat_or_none(run.completed_at),
        "selected_artifact_id": run.selected_artifact_id,
        "created_at": run.created_at.isoformat(),
        "progress": summarize_run_progress(run),
    }


def choose_cathode_source_artifact(
    session: storage.Session,
    *,
    patient_id: str,
    preferred_run_id: str | None = None,
) -> tuple[storage.Run, storage.Artifact] | None:
    runs = storage.list_runs(session, patient_id)
    if preferred_run_id:
        runs = [run for run in runs if run.id == preferred_run_id] + [
            run for run in runs if run.id != preferred_run_id
        ]
    for run in runs:
        artifacts = storage.list_artifacts(session, run.id)
        for stage_num, kind in ((4, "consolidation"), (3, "revision")):
            candidates = [
                artifact
                for artifact in artifacts
                if artifact.stage_num == stage_num
                and artifact.kind == kind
                and artifact.content_type.startswith("text/")
            ]
            if candidates:
                candidates.sort(key=lambda artifact: artifact.created_at, reverse=True)
                return run, candidates[0]
    return None


def cathode_status(
    *,
    patient_label: str,
    source_artifact: tuple[storage.Run, storage.Artifact] | None = None,
) -> dict[str, Any]:
    project_dir = cathode_project_dir(patient_label)
    plan_path = project_dir / "plan.json"
    payload_path = cathode_handoff_payload_path(patient_label)
    handoff_meta_path = cathode_handoff_meta_path(patient_label)
    source_path = cathode_handoff_source_path(patient_label)
    payload = _read_json(payload_path)
    handoff_meta = _read_json(handoff_meta_path)

    video_path: str | None = None
    video_exists = False
    if plan_path.exists():
        plan = _read_json(plan_path) or {}
        meta = plan.get("meta") if isinstance(plan.get("meta"), dict) else {}
        candidate = meta.get("video_path")
        if isinstance(candidate, str) and candidate.strip():
            resolved = Path(candidate).expanduser()
            if not resolved.is_absolute():
                resolved = (project_dir / resolved).resolve()
            video_path = str(resolved)
            video_exists = resolved.exists()

    status = "missing"
    if payload_path.exists() or source_path.exists():
        status = "handoff_prepared"
    if plan_path.exists():
        status = "project_ready"
    if video_exists:
        status = "video_ready"

    recommended = None
    if source_artifact:
        run, artifact = source_artifact
        recommended = {
            "run_id": run.id,
            "artifact": _artifact_out(artifact),
        }

    return {
        "status": status,
        "project_dir": str(project_dir),
        "project_exists": project_dir.exists(),
        "plan_path": str(plan_path),
        "plan_exists": plan_path.exists(),
        "video_path": video_path,
        "video_exists": video_exists,
        "handoff_payload_path": str(payload_path),
        "handoff_payload_exists": payload_path.exists(),
        "handoff_payload": payload,
        "handoff_meta_path": str(handoff_meta_path),
        "handoff_meta_exists": handoff_meta_path.exists(),
        "handoff_meta": handoff_meta,
        "handoff_source_path": str(source_path),
        "handoff_source_exists": source_path.exists(),
        "updated_at": max(
            [value for value in (_path_iso(plan_path), _path_iso(payload_path), _path_iso(source_path)) if value],
            default=None,
        ),
        "recommended_source": recommended,
    }


def build_patient_orchestration_summary(
    session: storage.Session, patient: storage.Patient
) -> dict[str, Any]:
    runs = storage.list_runs(session, patient.id)
    latest_run = runs[0] if runs else None
    active_run = next((run for run in runs if run.status == "running"), None)
    pipeline_status = load_pipeline_job_status(patient.label)
    portal = classify_portal_files(patient.label)
    sync_entry = _read_sync_state(patient.label)
    source_artifact = choose_cathode_source_artifact(session, patient_id=patient.id)
    cathode = cathode_status(patient_label=patient.label, source_artifact=source_artifact)

    state = "idle"
    label = "No runs yet"
    progress: dict[str, Any] | None = None
    if active_run is not None:
        progress = summarize_run_progress(active_run)
        state = "running"
        label = progress.get("phase_label") or "Running"
    elif latest_run is not None and latest_run.status in {"failed", "needs_auth"}:
        progress = summarize_run_progress(latest_run)
        state = "attention"
        label = latest_run.error_message or progress.get("phase_label") or latest_run.status
    elif pipeline_status and pipeline_status.get("status") == "failed":
        state = "attention"
        label = str(pipeline_status.get("note") or pipeline_status.get("status"))
    elif latest_run is not None:
        progress = summarize_run_progress(latest_run)
        state = "ready" if latest_run.status == "complete" else latest_run.status
        label = progress.get("phase_label") or latest_run.status
    elif pipeline_status:
        state = str(pipeline_status.get("status") or "idle")
        label = str(pipeline_status.get("note") or state)

    return {
        "state": state,
        "label": label,
        "progress": progress,
        "pipeline_status": {
            "status": pipeline_status.get("status"),
            "updated_at": pipeline_status.get("updated_at"),
        }
        if pipeline_status
        else None,
        "portal": {
            "source_report_count": len(portal["source_reports"]),
            "patient_facing_count": len(portal["patient_facing_pdfs"]),
            "council_run_count": len(portal["council_run_ids"]),
            "last_modified": portal.get("last_modified"),
            "sync_known": sync_entry is not None,
        },
        "cathode": {
            "status": cathode["status"],
            "project_exists": cathode["project_exists"],
            "video_exists": cathode["video_exists"],
            "updated_at": cathode["updated_at"],
        },
    }


def build_patient_orchestration_detail(
    session: storage.Session, patient: storage.Patient
) -> dict[str, Any]:
    reports = storage.list_reports(session, patient.id)
    runs = storage.list_runs(session, patient.id)
    pipeline_status = load_pipeline_job_status(patient.label)
    portal = classify_portal_files(patient.label)
    sync_entry = _read_sync_state(patient.label)
    portal_patient_id = normalize_portal_patient_id(patient.label)
    source_artifact = choose_cathode_source_artifact(session, patient_id=patient.id)
    cathode = cathode_status(patient_label=patient.label, source_artifact=source_artifact)
    latest_complete_run = next((run for run in runs if run.status == "complete"), None)
    latest_patient_facing_pdf = portal.get("latest_patient_facing_pdf")

    patient_facing_status = "missing"
    patient_facing_summary = "No patient-facing PDF detected in the portal folder."
    if latest_patient_facing_pdf:
        patient_facing_status = "ready"
        patient_facing_summary = (
            f"Latest patient-facing PDF: {latest_patient_facing_pdf['name']}"
        )
        if latest_complete_run and latest_complete_run.completed_at:
            latest_pdf_mtime = latest_patient_facing_pdf.get("mtime")
            if (
                isinstance(latest_pdf_mtime, str)
                and latest_pdf_mtime < latest_complete_run.completed_at.isoformat()
            ):
                patient_facing_status = "stale"
                patient_facing_summary = (
                    f"Patient-facing PDF may be stale relative to run {latest_complete_run.id[:8]}"
                )

    runs_by_report: dict[str, list[storage.Run]] = {}
    for run in runs:
        runs_by_report.setdefault(run.report_id, []).append(run)

    report_rows: list[dict[str, Any]] = []
    for report in reports:
        report_runs = runs_by_report.get(report.id, [])
        latest_run = report_runs[0] if report_runs else None
        completed_runs = [run for run in report_runs if run.status == "complete"]
        extracted_exists = False
        try:
            extracted_path = Path(report.extracted_text_path)
            extracted_exists = extracted_path.exists() and bool(
                extracted_path.read_text(encoding="utf-8", errors="replace").strip()
            )
        except Exception:
            extracted_exists = False
        patient_facing_for_report = bool(
            latest_patient_facing_pdf
            and latest_complete_run is not None
            and latest_complete_run.report_id == report.id
        )
        cathode_for_report = bool(
            source_artifact
            and source_artifact[0].report_id == report.id
        )
        report_rows.append(
            {
                "report_id": report.id,
                "filename": report.filename,
                "mime_type": report.mime_type,
                "created_at": report.created_at.isoformat(),
                "latest_run": _run_out(latest_run) if latest_run else None,
                "complete_run_count": len(completed_runs),
                "has_complete_run": bool(completed_runs),
                "lifecycle": {
                    "uploaded": True,
                    "extracted": extracted_exists,
                    "council_status": latest_run.status if latest_run else "pending",
                    "patient_facing_status": "ready" if patient_facing_for_report else "pending",
                    "portal_sync_status": (
                        str(sync_entry.get("status") or sync_entry.get("state"))
                        if isinstance(sync_entry, dict)
                        else "unknown"
                    ),
                    "cathode_status": cathode.get("status") if cathode_for_report else "pending",
                },
            }
        )

    active_runs = [_run_out(run) for run in runs if run.status == "running"]
    latest_run = _run_out(runs[0]) if runs else None

    return {
        "patient_id": patient.id,
        "patient_label": patient.label,
        "portal_patient_id": portal_patient_id,
        "summary": build_patient_orchestration_summary(session, patient),
        "reports": report_rows,
        "active_runs": active_runs,
        "latest_run": latest_run,
        "pipeline_job": pipeline_status,
        "patient_facing": {
            "status": patient_facing_status,
            "summary": patient_facing_summary,
            "latest_pdf": latest_patient_facing_pdf,
            "latest_complete_run_id": latest_complete_run.id if latest_complete_run else None,
            "latest_complete_run_completed_at": isoformat_or_none(
                latest_complete_run.completed_at if latest_complete_run else None
            ),
        },
        "portal": {
            **portal,
            "sync_entry": sync_entry,
        },
        "cathode": cathode,
        "recommended_cathode_source": {
            "run_id": source_artifact[0].id,
            "artifact": _artifact_out(source_artifact[1]),
        }
        if source_artifact
        else None,
        "actions": {
            "refresh": {"enabled": True},
            "sync_portal": {
                "enabled": portal_patient_id is not None,
                "reason": ""
                if portal_patient_id is not None
                else "Patient label is not a canonical portal patient id.",
            },
            "rerun_pipeline": {
                "enabled": portal_patient_id is not None,
                "reason": ""
                if portal_patient_id is not None
                else "Patient label is not a canonical portal patient id.",
            },
            "regenerate_patient_facing": {
                "enabled": latest_complete_run is not None,
                "reason": ""
                if latest_complete_run is not None
                else "No complete run is available yet.",
            },
            "prepare_cathode_handoff": {
                "enabled": portal_patient_id is not None and source_artifact is not None,
                "reason": ""
                if portal_patient_id is not None and source_artifact is not None
                else (
                    "No complete council markdown artifact is available yet."
                    if portal_patient_id is not None
                    else "Patient label is not a canonical portal patient id."
                ),
            },
            "export_council_artifacts": {
                "enabled": latest_complete_run is not None
                and bool(latest_complete_run.selected_artifact_id),
                "reason": ""
                if latest_complete_run is not None and latest_complete_run.selected_artifact_id
                else "No complete run with a selected final draft is available yet.",
            },
        },
        "updated_at": max(
            [
                value
                for value in (
                    portal.get("last_modified"),
                    pipeline_status.get("updated_at") if pipeline_status else None,
                    cathode.get("updated_at"),
                    latest_run.get("created_at") if latest_run else None,
                )
                if value
            ],
            default=None,
        ),
    }
