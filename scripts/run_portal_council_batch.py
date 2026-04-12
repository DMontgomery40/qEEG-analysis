#!/usr/bin/env python
from __future__ import annotations

import argparse
import asyncio
import fcntl
import json
import os
import shutil
import sys
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sqlalchemy import select

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from backend import storage  # noqa: E402
from backend.config import (  # noqa: E402
    ARTIFACTS_DIR,
    CLIPROXY_API_KEY,
    CLIPROXY_BASE_URL,
    COUNCIL_MODELS,
    DEFAULT_CONSOLIDATOR,
    EXPORTS_DIR,
    ensure_data_dirs,
    set_discovered_model_ids,
)
from backend.council import QEEGCouncilWorkflow  # noqa: E402
from backend.exports import render_markdown_to_pdf  # noqa: E402
from backend.llm_client import AsyncOpenAICompatClient, UpstreamError  # noqa: E402
from backend.main import _auto_generate_patient_facing_for_run  # noqa: E402
from backend.model_selection import resolve_model_preference  # noqa: E402
from backend.portal_sync import sync_patient_to_thrylen  # noqa: E402
from backend.reports import (  # noqa: E402
    report_enhanced_path,
    report_metadata_path,
    report_pages_dir,
    report_dir,
    save_report_upload,
)


@dataclass(frozen=True)
class BatchTask:
    patient_label: str
    patient_dir: Path
    pdf_path: Path


@dataclass(frozen=True)
class TaskOutcome:
    patient_label: str
    pdf_name: str
    status: str
    patient_id: str | None = None
    report_id: str | None = None
    run_id: str | None = None
    note: str = ""


class _NullBroker:
    async def publish(self, _run_id: str, _payload: dict[str, Any]) -> None:
        return None


def _batch_lock_path() -> Path:
    return _REPO_ROOT / "data" / "pipeline_jobs" / ".run_portal_council_batch.lock"


def _acquire_batch_lock():
    path = _batch_lock_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("w", encoding="utf-8")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        handle.close()
        raise RuntimeError(
            f"Another qEEG council batch is already running (lock: {path}). "
            "Stop the active batch or wait for it before starting another."
        ) from exc
    handle.write(str(os.getpid()))
    handle.truncate()
    handle.flush()
    return handle


def _format_progress_event(payload: dict[str, Any]) -> str:
    stage_num = payload.get("stage_num")
    stage_name = payload.get("stage_name")
    stage_part = ""
    if stage_num is not None or stage_name:
        stage_part = f"stage={stage_num}:{stage_name}"

    status = str(payload.get("status") or "event")
    details: list[str] = []
    for key in ("task", "model_id"):
        value = payload.get(key)
        if isinstance(value, str) and value:
            details.append(f"{key}={value}")

    if isinstance(payload.get("chunk_index"), int):
        chunk_total = payload.get("chunk_count")
        if isinstance(chunk_total, int) and chunk_total > 0:
            details.append(f"chunk={payload['chunk_index']}/{chunk_total}")
        else:
            details.append(f"chunk={payload['chunk_index']}")

    pages = payload.get("pages")
    if isinstance(pages, list) and pages:
        details.append("pages=" + ",".join(str(page) for page in pages))

    for key in ("elapsed_s", "heartbeat_count", "success_count", "requested_count"):
        value = payload.get(key)
        if value is not None:
            details.append(f"{key}={value}")

    for key in ("reason", "error", "operatorHint"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            details.append(f"{key}={' '.join(value.strip().split())}")

    pieces = [part for part in [stage_part, status, *details] if part]
    return " ".join(pieces).strip()


def _normalize_portal_patient_id(label: str) -> str | None:
    import re

    m = re.fullmatch(
        r"\s*(?P<mm>\d{1,2})-(?P<dd>\d{1,2})-(?P<yyyy>\d{4})-(?P<n>\d{1,3})\s*",
        label or "",
    )
    if not m:
        return None
    mm = int(m.group("mm"))
    dd = int(m.group("dd"))
    yyyy = int(m.group("yyyy"))
    n = int(m.group("n"))
    if not (1 <= mm <= 12 and 1 <= dd <= 31 and 1900 <= yyyy <= 2100 and 0 <= n <= 999):
        return None
    return f"{mm:02d}-{dd:02d}-{yyyy:04d}-{n}"


def _portal_patients_dir() -> Path:
    configured = (os.getenv("QEEG_PORTAL_PATIENTS_DIR") or "").strip()
    if configured:
        return Path(configured).expanduser()
    return _REPO_ROOT / "data" / "portal_patients"


def _safe_portal_filename(value: str, *, fallback: str = "upload.bin") -> str:
    import re

    raw = os.path.basename(str(value or "").strip())
    cleaned = re.sub(r"[/\\\u0000-\u001F\u007F]+", "_", raw).strip()
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    cleaned = cleaned[:200]
    return cleaned or fallback


def _publish_file_to_portal_folder(
    *, patient_label: str, src_path: Path, filename: str
) -> Path | None:
    patient_id = _normalize_portal_patient_id(patient_label)
    if patient_id is None:
        return None

    try:
        out_dir = _portal_patients_dir() / patient_id
        out_dir.mkdir(parents=True, exist_ok=True)
        dest_path = out_dir / _safe_portal_filename(filename)
        if dest_path.exists():
            dest_path.unlink()

        if dest_path.suffix.lower() in {".mp4", ".pdf", ".zip", ".docx", ".rtf"}:
            try:
                os.link(src_path, dest_path)
                return dest_path
            except Exception:
                pass

        tmp_path = dest_path.with_name(f".{dest_path.name}.partial")
        tmp_path.unlink(missing_ok=True)
        shutil.copy2(src_path, tmp_path)
        tmp_path.replace(dest_path)
        return dest_path
    except Exception:
        return None


def _is_special_manifest_patient(patient_dir: Path) -> bool:
    return (patient_dir / "combined_5sessions.manifest.json").exists()


def _is_source_pdf(patient_label: str, path: Path) -> bool:
    if not path.is_file() or path.suffix.lower() != ".pdf":
        return False

    name = path.name
    lower_name = name.lower()
    lower_label = patient_label.lower()

    if lower_name == f"{lower_label}.pdf":
        return False
    if lower_name == "main.pdf":
        return False
    if "__patient-facing__" in lower_name:
        return False
    if "__single-agent" in lower_name:
        return False
    if "__analysis__" in lower_name or lower_name.endswith("__analysis.pdf"):
        return False
    if lower_name.endswith("_analysis_pdf.pdf"):
        return False
    if lower_name.endswith("__patient-facing.pdf"):
        return False
    return True


def _discover_batch_tasks(
    *,
    portal_dir: Path,
    exclude_labels: set[str],
    skip_manifest_special_cases: bool,
) -> list[BatchTask]:
    tasks: list[BatchTask] = []
    if not portal_dir.exists():
        return tasks

    normalized_excludes = {
        (_normalize_portal_patient_id(v) or v).lower() for v in exclude_labels
    }
    for patient_dir in sorted(p for p in portal_dir.iterdir() if p.is_dir()):
        label = patient_dir.name
        normalized = _normalize_portal_patient_id(label)
        if normalized is None:
            continue
        if normalized.lower() in normalized_excludes:
            continue
        if skip_manifest_special_cases and _is_special_manifest_patient(patient_dir):
            continue
        for pdf_path in sorted(patient_dir.glob("*.pdf")):
            if _is_source_pdf(label, pdf_path):
                tasks.append(
                    BatchTask(
                        patient_label=normalized,
                        patient_dir=patient_dir,
                        pdf_path=pdf_path,
                    )
                )
    return tasks


def _pick_model_id(preferred: str, discovered: list[str]) -> str | None:
    return resolve_model_preference(preferred, discovered)


def _portal_batch_model_allowed(model_id: str) -> bool:
    if (os.getenv("QEEG_ALLOW_OPUS_MODELS") or "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }:
        return True
    return "claude-opus" not in (model_id or "").strip().lower()


def _fallback_council_model_ids(discovered: list[str]) -> list[str]:
    preferred_tokens = [
        "claude-sonnet",
        "gemini-3-pro",
        "gemini-2.5-pro",
        "gpt-5.4",
        "gpt-5.3",
        "gpt-5.2",
        "gpt-5.1",
        "claude",
        "gemini",
        "gpt",
    ]
    picked: list[str] = []
    for token in preferred_tokens:
        match = _pick_model_id(token, discovered)
        if match and match not in picked and _portal_batch_model_allowed(match):
            picked.append(match)
        if len(picked) >= 3:
            break
    if not picked:
        picked.extend(
            model_id for model_id in discovered if _portal_batch_model_allowed(model_id)
        )
        picked = picked[:1]
    return picked


def _resolve_model_selection_for_run(
    session,
    *,
    patient_id: str,
    discovered_models: list[str],
) -> tuple[list[str], str]:
    def from_run(run: storage.Run) -> tuple[list[str], str] | None:
        try:
            raw_council = json.loads(run.council_model_ids_json or "[]")
        except Exception:
            return None
        if not isinstance(raw_council, list):
            return None
        mapped: list[str] = []
        for item in raw_council:
            if not isinstance(item, str):
                continue
            matched = _pick_model_id(item, discovered_models)
            if (
                matched
                and matched not in mapped
                and _portal_batch_model_allowed(matched)
            ):
                mapped.append(matched)
        consolidator = _pick_model_id(run.consolidator_model_id, discovered_models)
        if consolidator is not None and not _portal_batch_model_allowed(consolidator):
            consolidator = None
        if not mapped:
            return None
        if consolidator is None:
            consolidator = mapped[0]
        return mapped, consolidator

    configured: list[str] = []
    for model in COUNCIL_MODELS:
        matched = _pick_model_id(model.id, discovered_models)
        if (
            matched
            and matched not in configured
            and _portal_batch_model_allowed(matched)
        ):
            configured.append(matched)

    if configured:
        consolidator = _pick_model_id(DEFAULT_CONSOLIDATOR, discovered_models)
        if consolidator is not None and not _portal_batch_model_allowed(consolidator):
            consolidator = None
        if consolidator is None:
            consolidator = next(
                (
                    model_id
                    for model_id in configured
                    if model_id.lower().startswith("gpt-5")
                ),
                configured[0],
            )
        return configured, consolidator

    patient_runs = list(
        session.scalars(
            select(storage.Run)
            .where(storage.Run.patient_id == patient_id)
            .order_by(storage.Run.created_at.desc())
        )
    )
    for run in patient_runs:
        resolved = from_run(run)
        if resolved is not None:
            return resolved

    complete_runs = list(
        session.scalars(
            select(storage.Run)
            .where(storage.Run.status == "complete")
            .order_by(storage.Run.created_at.desc())
        )
    )
    for run in complete_runs:
        resolved = from_run(run)
        if resolved is not None:
            return resolved

    council_model_ids = _fallback_council_model_ids(discovered_models)
    if not council_model_ids:
        raise RuntimeError(
            "No compatible council models were discovered from CLIProxyAPI /v1/models"
        )
    consolidator = _pick_model_id(DEFAULT_CONSOLIDATOR, discovered_models)
    if consolidator is None:
        consolidator = next(
            (
                model_id
                for model_id in council_model_ids
                if "claude" in model_id.lower() or "gpt" in model_id.lower()
            ),
            council_model_ids[0],
        )
    return council_model_ids, consolidator


def _report_assets_ready(report: storage.Report) -> bool:
    extracted_path = Path(report.extracted_text_path)
    enhanced_path = report_enhanced_path(report.patient_id, report.id)
    metadata_path = report_metadata_path(report.patient_id, report.id)
    pages_dir = report_pages_dir(report.patient_id, report.id)
    page_images = list(pages_dir.glob("page-*.png")) if pages_dir.exists() else []
    return (
        extracted_path.exists()
        and enhanced_path.exists()
        and metadata_path.exists()
        and len(page_images) > 0
    )


def _score_report_candidate(session, report: storage.Report) -> tuple[int, int, str]:
    runs = list(
        session.scalars(
            select(storage.Run)
            .where(storage.Run.report_id == report.id)
            .order_by(storage.Run.created_at.desc())
        )
    )
    complete_runs = sum(1 for run in runs if run.status == "complete")
    latest_run = runs[0].created_at.isoformat() if runs else ""
    return (complete_runs, int(_report_assets_ready(report)), latest_run)


def _choose_existing_report(
    session,
    *,
    patient_id: str,
    filename: str,
) -> storage.Report | None:
    reports = [
        report
        for report in storage.list_reports(session, patient_id)
        if (report.filename or "") == filename
    ]
    if not reports:
        return None
    return max(reports, key=lambda report: _score_report_candidate(session, report))


def _score_patient_candidate(
    session,
    *,
    patient: storage.Patient,
    portal_pdf_names: set[str],
) -> tuple[int, int, int, str, str]:
    reports = storage.list_reports(session, patient.id)
    matching_reports = [
        report for report in reports if report.filename in portal_pdf_names
    ]
    all_runs = list(
        session.scalars(
            select(storage.Run)
            .where(storage.Run.patient_id == patient.id)
            .order_by(storage.Run.created_at.desc())
        )
    )
    matching_run_ids = {report.id for report in matching_reports}
    matching_complete_runs = sum(
        1
        for run in all_runs
        if run.report_id in matching_run_ids and run.status == "complete"
    )
    total_complete_runs = sum(1 for run in all_runs if run.status == "complete")
    latest_run = all_runs[0].created_at.isoformat() if all_runs else ""
    return (
        len(matching_reports),
        matching_complete_runs,
        total_complete_runs,
        latest_run,
        patient.created_at.isoformat(),
    )


def _resolve_patient_for_label(
    session,
    *,
    patient_label: str,
    portal_pdfs: list[Path],
) -> storage.Patient | None:
    existing = storage.find_patients_by_label(session, patient_label)
    if not existing:
        return None
    if len(existing) == 1:
        return existing[0]
    portal_pdf_names = {pdf.name for pdf in portal_pdfs}
    return max(
        existing,
        key=lambda patient: _score_patient_candidate(
            session, patient=patient, portal_pdf_names=portal_pdf_names
        ),
    )


def _get_or_create_patient_for_label(
    session,
    *,
    patient_label: str,
    portal_pdfs: list[Path],
) -> storage.Patient:
    patient = _resolve_patient_for_label(
        session,
        patient_label=patient_label,
        portal_pdfs=portal_pdfs,
    )
    if patient is not None:
        return patient
    return storage.create_patient(session, label=patient_label, notes="")


def _ensure_report_registered(
    *,
    patient_id: str,
    pdf_path: Path,
    reextract_existing: bool,
) -> tuple[str, str]:
    with storage.session_scope() as session:
        existing_report = _choose_existing_report(
            session,
            patient_id=patient_id,
            filename=pdf_path.name,
        )

    if existing_report is not None:
        if reextract_existing or not _report_assets_ready(existing_report):
            save_report_upload(
                patient_id=existing_report.patient_id,
                report_id=existing_report.id,
                filename=existing_report.filename,
                provided_mime_type="application/pdf",
                file_bytes=pdf_path.read_bytes(),
            )
            return existing_report.id, "reextracted"
        return existing_report.id, "reused"

    report_id = str(uuid.uuid4())
    report_path = report_dir(patient_id, report_id)
    try:
        original_path, extracted_path, mime_type, _preview = save_report_upload(
            patient_id=patient_id,
            report_id=report_id,
            filename=pdf_path.name,
            provided_mime_type="application/pdf",
            file_bytes=pdf_path.read_bytes(),
        )
        with storage.session_scope() as session:
            storage.create_report(
                session,
                report_id=report_id,
                patient_id=patient_id,
                filename=pdf_path.name,
                mime_type=mime_type,
                stored_path=original_path,
                extracted_text_path=extracted_path,
            )
    except Exception:
        shutil.rmtree(report_path, ignore_errors=True)
        raise
    return report_id, "uploaded"


def _latest_complete_run_for_report(report_id: str) -> storage.Run | None:
    with storage.session_scope() as session:
        runs = list(
            session.scalars(
                select(storage.Run)
                .where(
                    storage.Run.report_id == report_id,
                    storage.Run.status == "complete",
                )
                .order_by(storage.Run.created_at.desc())
            )
        )
        return runs[0] if runs else None


def _latest_resume_candidate_run_for_report(report_id: str) -> storage.Run | None:
    with storage.session_scope() as session:
        runs = list(
            session.scalars(
                select(storage.Run)
                .where(
                    storage.Run.report_id == report_id,
                    storage.Run.status.in_(("created", "running")),
                )
                .order_by(storage.Run.created_at.desc())
            )
        )
        return runs[0] if runs else None


def _resume_candidate_for_report(
    report_id: str, *, skip_complete: bool
) -> storage.Run | None:
    if skip_complete is False:
        return None
    return _latest_resume_candidate_run_for_report(report_id)


def _choose_stage6_artifact(session, run: storage.Run) -> storage.Artifact | None:
    artifacts = storage.list_artifacts(session, run.id)
    stage6 = [
        artifact
        for artifact in artifacts
        if artifact.stage_num == 6
        and artifact.kind == "final_draft"
        and artifact.content_type == "text/markdown"
    ]
    if not stage6:
        return None

    if run.selected_artifact_id:
        selected = next(
            (
                artifact
                for artifact in stage6
                if artifact.id == run.selected_artifact_id
            ),
            None,
        )
        if selected is not None:
            return selected

    try:
        model_order = json.loads(run.council_model_ids_json or "[]")
    except Exception:
        model_order = []
    for model_id in model_order:
        for artifact in stage6:
            if artifact.model_id == model_id:
                return artifact
    return stage6[0]


def _export_run(run_id: str) -> tuple[Path, Path]:
    with storage.session_scope() as session:
        run = storage.get_run(session, run_id)
        if run is None:
            raise RuntimeError(f"Run not found: {run_id}")
        patient = storage.get_patient(session, run.patient_id)
        if patient is None:
            raise RuntimeError(f"Patient not found for run: {run_id}")
        artifact = _choose_stage6_artifact(session, run)
        if artifact is None:
            raise RuntimeError(f"No Stage 6 final draft available for run: {run_id}")
        if run.selected_artifact_id != artifact.id:
            storage.select_artifact(session, run.id, artifact.id)
            run = storage.get_run(session, run_id)
            if run is None:
                raise RuntimeError(
                    f"Run disappeared after selecting artifact: {run_id}"
                )

    md = Path(artifact.content_path).read_text(encoding="utf-8", errors="replace")
    export_dir = EXPORTS_DIR / run_id
    export_dir.mkdir(parents=True, exist_ok=True)
    md_path = export_dir / "final.md"
    pdf_path = export_dir / "final.pdf"
    md_path.write_text(md, encoding="utf-8")
    render_markdown_to_pdf(md, pdf_path)

    _publish_file_to_portal_folder(
        patient_label=patient.label,
        src_path=md_path,
        filename=f"{patient.label}.md",
    )
    _publish_file_to_portal_folder(
        patient_label=patient.label,
        src_path=pdf_path,
        filename=f"{patient.label}.pdf",
    )
    return md_path, pdf_path


def _stage_run_artifacts(*, patient_label: str, run_id: str) -> int:
    source_root = ARTIFACTS_DIR / run_id
    if not source_root.exists():
        return 0

    patient_id = _normalize_portal_patient_id(patient_label)
    if patient_id is None:
        return 0

    dest_root = _portal_patients_dir() / patient_id / "council" / run_id
    files_to_copy: list[tuple[Path, Path]] = []

    stage1_dir = source_root / "stage-1"
    for name in ("_data_pack.json", "_vision_transcript.md"):
        src = stage1_dir / name
        if src.exists():
            files_to_copy.append((src, dest_root / "stage-1" / name))

    for stage_name in ("stage-4", "stage-6"):
        stage_dir = source_root / stage_name
        if stage_dir.exists():
            for src in sorted(stage_dir.glob("*.md")):
                files_to_copy.append((src, dest_root / stage_name / src.name))

    copied = 0
    for src, dest in files_to_copy:
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)
        copied += 1
    return copied


async def _run_pipeline_for_run(
    *,
    workflow: QEEGCouncilWorkflow,
    run_id: str,
) -> None:
    async def on_event(payload: dict[str, Any]) -> None:
        line = _format_progress_event(payload)
        print(f"[{run_id}] {line or 'event'}", flush=True)

    await workflow.run_pipeline(run_id, on_event=on_event)


async def _generate_patient_facing_outputs(run_id: str) -> None:
    await _auto_generate_patient_facing_for_run(run_id, _NullBroker())


async def _process_task(
    *,
    task: BatchTask,
    workflow: QEEGCouncilWorkflow,
    discovered_models: list[str],
    skip_complete: bool,
    reextract_existing: bool,
    dry_run: bool,
) -> TaskOutcome:
    portal_pdfs = [
        pdf_path
        for pdf_path in sorted(task.patient_dir.glob("*.pdf"))
        if _is_source_pdf(task.patient_label, pdf_path)
    ]

    with storage.session_scope() as session:
        patient = (
            _resolve_patient_for_label(
                session,
                patient_label=task.patient_label,
                portal_pdfs=portal_pdfs,
            )
            if dry_run
            else _get_or_create_patient_for_label(
                session,
                patient_label=task.patient_label,
                portal_pdfs=portal_pdfs,
            )
        )
        patient_id = patient.id if patient is not None else None
        existing_report = (
            _choose_existing_report(
                session,
                patient_id=patient.id,
                filename=task.pdf_path.name,
            )
            if patient is not None
            else None
        )

    if dry_run:
        report_action = "would_upload"
        report_id = existing_report.id if existing_report is not None else None
        if existing_report is not None:
            if reextract_existing or not _report_assets_ready(existing_report):
                report_action = "would_reextract"
            else:
                report_action = "would_reuse"
        existing_complete_run = (
            _latest_complete_run_for_report(existing_report.id)
            if existing_report is not None
            else None
        )
        status = "dry_run"
        dry_run_run_id: str | None = None
        if existing_complete_run is not None and skip_complete:
            status = "would_skip_complete"
            dry_run_run_id = existing_complete_run.id
        note = report_action
        if existing_complete_run is not None:
            note += f" run={existing_complete_run.id}"
        elif report_id is not None:
            resume_candidate = _resume_candidate_for_report(
                report_id,
                skip_complete=skip_complete,
            )
            if resume_candidate is not None:
                status = "would_resume"
                dry_run_run_id = resume_candidate.id
                note += f" run={resume_candidate.id}"
        return TaskOutcome(
            patient_label=task.patient_label,
            pdf_name=task.pdf_path.name,
            status=status,
            patient_id=patient_id,
            report_id=report_id,
            run_id=dry_run_run_id,
            note=note,
        )

    assert patient_id is not None

    if existing_report is not None:
        existing_complete_run = _latest_complete_run_for_report(existing_report.id)
        if existing_complete_run is not None and skip_complete:
            _export_run(existing_complete_run.id)
            _stage_run_artifacts(
                patient_label=task.patient_label,
                run_id=existing_complete_run.id,
            )
            await _generate_patient_facing_outputs(existing_complete_run.id)
            sync_patient_to_thrylen(task.patient_label)
            return TaskOutcome(
                patient_label=task.patient_label,
                pdf_name=task.pdf_path.name,
                status="skipped_complete",
                patient_id=patient_id,
                report_id=existing_report.id,
                run_id=existing_complete_run.id,
                note="reused",
            )

    report_id, report_action = _ensure_report_registered(
        patient_id=patient_id,
        pdf_path=task.pdf_path,
        reextract_existing=reextract_existing,
    )

    existing_complete_run = _latest_complete_run_for_report(report_id)
    if existing_report is None and existing_complete_run is not None and skip_complete:
        _export_run(existing_complete_run.id)
        _stage_run_artifacts(
            patient_label=task.patient_label,
            run_id=existing_complete_run.id,
        )
        await _generate_patient_facing_outputs(existing_complete_run.id)
        sync_patient_to_thrylen(task.patient_label)
        return TaskOutcome(
            patient_label=task.patient_label,
            pdf_name=task.pdf_path.name,
            status="skipped_complete",
            patient_id=patient_id,
            report_id=report_id,
            run_id=existing_complete_run.id,
            note=report_action,
        )

    resume_candidate = _resume_candidate_for_report(
        report_id,
        skip_complete=skip_complete,
    )
    if resume_candidate is not None:
        run_id = resume_candidate.id
        report_action = f"{report_action}; resumed {resume_candidate.status}"
        if resume_candidate.status == "created":
            with storage.session_scope() as session:
                storage.claim_run_start(session, run_id)
    else:
        try:
            with storage.session_scope() as session:
                council_model_ids, consolidator_model_id = _resolve_model_selection_for_run(
                    session,
                    patient_id=patient_id,
                    discovered_models=discovered_models,
                )
                run = storage.create_run(
                    session,
                    patient_id=patient_id,
                    report_id=report_id,
                    council_model_ids=council_model_ids,
                    consolidator_model_id=consolidator_model_id,
                )
                run_id = run.id
                storage.claim_run_start(session, run_id)
        except Exception as exc:
            return TaskOutcome(
                patient_label=task.patient_label,
                pdf_name=task.pdf_path.name,
                status="failed",
                patient_id=patient_id,
                report_id=report_id,
                run_id=None,
                note=str(exc),
            )

    try:
        await _run_pipeline_for_run(workflow=workflow, run_id=run_id)
    except Exception as exc:
        with storage.session_scope() as session:
            run = storage.get_run(session, run_id)
            run_status = (
                run.status if run is not None else "missing"
            ).strip() or "failed"
            error_message = (
                run.error_message if run is not None and run.error_message else str(exc)
            )
        return TaskOutcome(
            patient_label=task.patient_label,
            pdf_name=task.pdf_path.name,
            status=run_status,
            patient_id=patient_id,
            report_id=report_id,
            run_id=run_id,
            note=error_message,
        )

    with storage.session_scope() as session:
        run = storage.get_run(session, run_id)
        if run is None:
            return TaskOutcome(
                patient_label=task.patient_label,
                pdf_name=task.pdf_path.name,
                status="failed",
                patient_id=patient_id,
                report_id=report_id,
                run_id=run_id,
                note="Run not found after workflow completed",
            )
        if run.status != "complete":
            return TaskOutcome(
                patient_label=task.patient_label,
                pdf_name=task.pdf_path.name,
                status=run.status,
                patient_id=patient_id,
                report_id=report_id,
                run_id=run_id,
                note=run.error_message or run.status,
            )

    _export_run(run_id)
    _stage_run_artifacts(patient_label=task.patient_label, run_id=run_id)
    await _generate_patient_facing_outputs(run_id)
    sync_patient_to_thrylen(task.patient_label)
    return TaskOutcome(
        patient_label=task.patient_label,
        pdf_name=task.pdf_path.name,
        status="complete",
        patient_id=patient_id,
        report_id=report_id,
        run_id=run_id,
        note=report_action,
    )


def _print_task_plan(tasks: list[BatchTask]) -> None:
    print(f"Found {len(tasks)} source PDF(s) to process.", flush=True)
    for task in tasks:
        print(f"- {task.patient_label}: {task.pdf_path.name}", flush=True)


async def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run portal qEEG PDFs through the council pipeline sequentially."
    )
    parser.add_argument(
        "--portal-dir",
        default=str(_portal_patients_dir()),
        help="Portal patients directory (default: data/portal_patients)",
    )
    parser.add_argument(
        "--exclude-label",
        action="append",
        default=[],
        help="Patient label to exclude. Can be repeated.",
    )
    parser.add_argument(
        "--include-label",
        action="append",
        default=[],
        help="Only process these patient labels. Can be repeated.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optional max number of source PDFs to process.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Create a new run even when a report already has a completed run.",
    )
    parser.add_argument(
        "--reextract-existing",
        action="store_true",
        help="Regenerate extraction assets for matching reports before running.",
    )
    parser.add_argument(
        "--include-manifest-special-cases",
        action="store_true",
        help="Include patients that have combined multi-session manifests.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show the planned batch without creating runs.",
    )
    args = parser.parse_args()

    ensure_data_dirs()
    storage.init_db()

    portal_dir = Path(args.portal_dir).expanduser()
    tasks = _discover_batch_tasks(
        portal_dir=portal_dir,
        exclude_labels=set(args.exclude_label),
        skip_manifest_special_cases=not args.include_manifest_special_cases,
    )

    if args.include_label:
        wanted = {
            (_normalize_portal_patient_id(label) or label).lower()
            for label in args.include_label
        }
        tasks = [task for task in tasks if task.patient_label.lower() in wanted]

    if args.limit and args.limit > 0:
        tasks = tasks[: args.limit]

    _print_task_plan(tasks)
    if not tasks:
        return 0

    if args.dry_run:
        outcomes: list[TaskOutcome] = []
        for index, task in enumerate(tasks, start=1):
            print(
                f"\n[{index}/{len(tasks)}] {task.patient_label} :: {task.pdf_path.name}",
                flush=True,
            )
            outcome = await _process_task(
                task=task,
                workflow=None,  # type: ignore[arg-type]
                discovered_models=[],
                skip_complete=not args.force,
                reextract_existing=args.reextract_existing,
                dry_run=True,
            )
            outcomes.append(outcome)
            print(
                f"  -> {outcome.status}"
                + (f" ({outcome.note})" if outcome.note else ""),
                flush=True,
            )
        print("\nBatch summary:", flush=True)
        for outcome in outcomes:
            run_part = f" run={outcome.run_id}" if outcome.run_id else ""
            print(
                f"- {outcome.patient_label} :: {outcome.pdf_name} -> {outcome.status}{run_part}",
                flush=True,
            )
        return 0

    try:
        batch_lock = _acquire_batch_lock()
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr, flush=True)
        return 1

    llm = AsyncOpenAICompatClient(
        base_url=CLIPROXY_BASE_URL,
        api_key=CLIPROXY_API_KEY,
        timeout_s=600.0,
    )
    try:
        discovered_models = await llm.list_models()
    except UpstreamError as exc:
        await llm.aclose()
        print(
            f"Failed to list models from CLIProxyAPI at {CLIPROXY_BASE_URL}: {exc}",
            file=sys.stderr,
        )
        return 1

    set_discovered_model_ids(discovered_models)
    workflow = QEEGCouncilWorkflow(llm=llm)
    outcomes: list[TaskOutcome] = []

    try:
        try:
            for index, task in enumerate(tasks, start=1):
                print(
                    f"\n[{index}/{len(tasks)}] {task.patient_label} :: {task.pdf_path.name}",
                    flush=True,
                )
                outcome = await _process_task(
                    task=task,
                    workflow=workflow,
                    discovered_models=discovered_models,
                    skip_complete=not args.force,
                    reextract_existing=args.reextract_existing,
                    dry_run=args.dry_run,
                )
                outcomes.append(outcome)
                print(
                    f"  -> {outcome.status}"
                    + (f" ({outcome.note})" if outcome.note else ""),
                    flush=True,
                )
        finally:
            await llm.aclose()
    finally:
        batch_lock.close()

    print("\nBatch summary:", flush=True)
    for outcome in outcomes:
        run_part = f" run={outcome.run_id}" if outcome.run_id else ""
        print(
            f"- {outcome.patient_label} :: {outcome.pdf_name} -> {outcome.status}{run_part}",
            flush=True,
        )

    failures = [
        outcome
        for outcome in outcomes
        if outcome.status
        not in {"complete", "skipped_complete", "dry_run", "would_skip_complete"}
    ]
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
