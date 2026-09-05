#!/usr/bin/env python
from __future__ import annotations

import argparse
import asyncio
import json
import shutil
import sys
import uuid
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from backend.config import (  # noqa: E402
    CLIPROXY_API_KEY,
    CLIPROXY_BASE_URL,
    ensure_data_dirs,
    set_discovered_model_ids,
)
from backend.council import QEEGCouncilWorkflow  # noqa: E402
from backend.llm_client import AsyncOpenAICompatClient  # noqa: E402
from backend.reports import (  # noqa: E402
    report_dir,
    report_extracted_path,
    report_original_path,
)
from backend.storage import (  # noqa: E402
    create_report,
    create_run,
    find_patients_by_label,
    get_patient,
    get_report,
    get_run,
    init_db,
    list_runs,
    session_scope,
)


from backend.report_composition import (  # noqa: E402
    SourceSpec,
    Manifest,
    _extract_source,
    _write_combined_report,
)


def _load_manifest(path: Path) -> Manifest:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Manifest must be a JSON object")

    patient_label = payload.get("patient_label")
    combined_filename = payload.get("combined_filename", "combined_council_report.pdf")
    notes = payload.get("notes", "")
    raw_sources = payload.get("sources")
    raw_models = payload.get("council_model_ids", [])
    consolidator_model_id = payload.get("consolidator_model_id")

    if not isinstance(patient_label, str) or not patient_label.strip():
        raise ValueError("Manifest requires patient_label")
    if not isinstance(combined_filename, str) or not combined_filename.strip():
        raise ValueError("Manifest requires combined_filename")
    if not isinstance(notes, str):
        raise ValueError("Manifest notes must be a string")
    if not isinstance(raw_sources, list) or not raw_sources:
        raise ValueError("Manifest requires a non-empty sources array")
    if raw_models and not all(
        isinstance(item, str) and item.strip() for item in raw_models
    ):
        raise ValueError("council_model_ids must be an array of strings")
    if consolidator_model_id is not None and not isinstance(consolidator_model_id, str):
        raise ValueError("consolidator_model_id must be a string when provided")

    sources: list[SourceSpec] = []
    for item in raw_sources:
        if not isinstance(item, dict):
            raise ValueError("Each source must be an object")
        raw_path = item.get("path")
        raw_aliases = item.get("session_aliases")
        if not isinstance(raw_path, str) or not raw_path.strip():
            raise ValueError("Each source requires path")
        if not isinstance(raw_aliases, dict) or not raw_aliases:
            raise ValueError(f"Source {raw_path} requires session_aliases")

        aliases: dict[int, int] = {}
        for local_raw, global_raw in raw_aliases.items():
            try:
                local_idx = int(local_raw)
                global_idx = int(global_raw)
            except Exception as exc:
                raise ValueError(
                    f"Invalid session alias {local_raw} -> {global_raw} for {raw_path}"
                ) from exc
            aliases[local_idx] = global_idx

        source_path = Path(raw_path).expanduser()
        if not source_path.is_absolute():
            source_path = (_REPO_ROOT / source_path).resolve()
        sources.append(
            SourceSpec(path=source_path, session_aliases=dict(sorted(aliases.items())))
        )

    return Manifest(
        patient_label=patient_label.strip(),
        combined_filename=combined_filename.strip(),
        notes=notes.strip(),
        sources=sources,
        council_model_ids=[item.strip() for item in raw_models],
        consolidator_model_id=consolidator_model_id.strip()
        if isinstance(consolidator_model_id, str)
        else None,
    )


def _patient_id_for_label(label: str) -> str:
    with session_scope() as session:
        patients = find_patients_by_label(session, label)
        if not patients:
            raise RuntimeError(f"No patient found for label {label!r}")
        if len(patients) > 1:
            raise RuntimeError(
                f"Multiple patients found for label {label!r}: "
                + ", ".join(patient.id for patient in patients)
            )
        patient = patients[0]
        if get_patient(session, patient.id) is None:
            raise RuntimeError(
                f"Patient row disappeared while resolving label {label!r}"
            )
        return patient.id


def _default_model_selection(patient_id: str) -> tuple[list[str], str] | None:
    with session_scope() as session:
        runs = list_runs(session, patient_id)
        runs = [
            run
            for run in runs
            if run.council_model_ids_json and run.consolidator_model_id
        ]
        if not runs:
            return None
        run = runs[0]
        try:
            council_model_ids = json.loads(run.council_model_ids_json)
        except Exception:
            return None
        if not isinstance(council_model_ids, list) or not all(
            isinstance(mid, str) and mid.strip() for mid in council_model_ids
        ):
            return None
        consolidator_model_id = (run.consolidator_model_id or "").strip()
        if not consolidator_model_id:
            return None
        return ([mid.strip() for mid in council_model_ids], consolidator_model_id)


def _register_report(*, patient_id: str, report_id: str, manifest: Manifest) -> None:
    original_path = report_original_path(
        patient_id, report_id, manifest.combined_filename
    )
    extracted_path = report_extracted_path(patient_id, report_id)
    with session_scope() as session:
        create_report(
            session,
            report_id=report_id,
            patient_id=patient_id,
            filename=manifest.combined_filename,
            mime_type="application/pdf",
            stored_path=original_path,
            extracted_text_path=extracted_path,
        )


async def _maybe_set_discovered_models(llm: AsyncOpenAICompatClient) -> list[str]:
    try:
        discovered = await llm.list_models()
    except Exception:
        return []
    set_discovered_model_ids(discovered)
    return discovered


async def _maybe_create_and_run(
    *,
    patient_id: str,
    report_id: str,
    manifest: Manifest,
    create_run_flag: bool,
    start_run_flag: bool,
) -> tuple[str | None, str | None]:
    if not create_run_flag and not start_run_flag:
        return (None, None)

    model_selection = None
    if manifest.council_model_ids and manifest.consolidator_model_id:
        model_selection = (manifest.council_model_ids, manifest.consolidator_model_id)
    else:
        model_selection = _default_model_selection(patient_id)

    if model_selection is None:
        raise RuntimeError(
            "No council model selection available. Set council_model_ids/consolidator_model_id in the manifest "
            "or create a prior run for this patient first."
        )

    council_model_ids, consolidator_model_id = model_selection
    run_id: str
    with session_scope() as session:
        run = create_run(
            session,
            patient_id=patient_id,
            report_id=report_id,
            council_model_ids=council_model_ids,
            consolidator_model_id=consolidator_model_id,
        )
        run_id = run.id

    if not start_run_flag:
        return (run_id, None)

    llm = AsyncOpenAICompatClient(
        base_url=CLIPROXY_BASE_URL,
        api_key=CLIPROXY_API_KEY,
        timeout_s=600.0,
    )
    try:
        await _maybe_set_discovered_models(llm)
        workflow = QEEGCouncilWorkflow(llm=llm)
        await workflow.run_pipeline(run_id)
    finally:
        await llm.aclose()

    with session_scope() as session:
        run = get_run(session, run_id)
        if run is None:
            raise RuntimeError(f"Run {run_id} not found after execution")
        if run.status != "complete":
            raise RuntimeError(
                f"Run {run_id} finished with status={run.status}: {run.error_message or run.status}"
            )

    return (run_id, run_id)


async def main() -> int:
    parser = argparse.ArgumentParser(
        description="Create a synthetic combined qEEG report asset for one council run."
    )
    parser.add_argument(
        "--manifest",
        required=True,
        help="Path to JSON manifest describing source PDFs and session aliases",
    )
    parser.add_argument(
        "--report-id",
        default="",
        help="Optional report id to use instead of a random UUID",
    )
    parser.add_argument(
        "--create-run",
        action="store_true",
        help="Create a council run for the combined report",
    )
    parser.add_argument(
        "--start-run",
        action="store_true",
        help="Create and immediately run the council workflow",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate inputs and print the planned actions only",
    )
    args = parser.parse_args()

    manifest_path = Path(args.manifest).expanduser()
    if not manifest_path.is_absolute():
        manifest_path = (_REPO_ROOT / manifest_path).resolve()
    manifest = _load_manifest(manifest_path)

    ensure_data_dirs()
    init_db()

    patient_id = _patient_id_for_label(manifest.patient_label)
    report_id = args.report_id.strip() or str(uuid.uuid4())

    with session_scope() as session:
        if get_report(session, report_id) is not None:
            raise RuntimeError(f"Report id already exists: {report_id}")

    extracted_sources = [
        _extract_source(source_spec) for source_spec in manifest.sources
    ]

    if args.dry_run:
        total_pages = sum(len(source.page_sections) for source in extracted_sources)
        print(
            json.dumps(
                {
                    "patient_label": manifest.patient_label,
                    "patient_id": patient_id,
                    "report_id": report_id,
                    "combined_filename": manifest.combined_filename,
                    "source_files": [
                        str(source.spec.path) for source in extracted_sources
                    ],
                    "total_pages": total_pages,
                    "create_run": bool(args.create_run or args.start_run),
                    "start_run": bool(args.start_run),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    out_dir = report_dir(patient_id, report_id)
    try:
        _write_combined_report(
            patient_id=patient_id,
            report_id=report_id,
            manifest=manifest,
            manifest_path=manifest_path,
            extracted_sources=extracted_sources,
        )
        _register_report(patient_id=patient_id, report_id=report_id, manifest=manifest)
    except Exception:
        shutil.rmtree(out_dir, ignore_errors=True)
        raise

    created_run_id, started_run_id = await _maybe_create_and_run(
        patient_id=patient_id,
        report_id=report_id,
        manifest=manifest,
        create_run_flag=bool(args.create_run or args.start_run),
        start_run_flag=bool(args.start_run),
    )

    print(f"patient_id={patient_id}")
    print(f"report_id={report_id}")
    if created_run_id:
        print(f"run_id={created_run_id}")
    if started_run_id:
        print(f"run_started={started_run_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
