#!/usr/bin/env python
from __future__ import annotations

import argparse
import asyncio
import base64
import json
import sys
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import re
from typing import Any

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
from backend.council.report_text import _iter_page_sections, _page_section_body  # noqa: E402
from backend.llm_client import AsyncOpenAICompatClient  # noqa: E402
from backend.reports import (  # noqa: E402
    report_dir,
    report_enhanced_path,
    report_extracted_path,
    report_metadata_path,
    report_original_path,
    report_pages_dir,
    extract_pdf_full,
)
from backend.storage import (  # noqa: E402
    create_report,
    create_run,
    find_patients_by_label,
    get_patient,
    init_db,
    list_runs,
    session_scope,
)


@dataclass(frozen=True)
class SourceSpec:
    path: Path
    session_aliases: dict[int, int]


@dataclass(frozen=True)
class Manifest:
    patient_label: str
    combined_filename: str
    notes: str
    sources: list[SourceSpec]
    council_model_ids: list[str]
    consolidator_model_id: str | None


@dataclass(frozen=True)
class ExtractedSource:
    spec: SourceSpec
    page_sections: list[str]
    page_images: list[dict[str, Any]]
    per_page_sources: list[dict[str, Any]]
    metadata: dict[str, Any]
    session_dates: dict[int, str]


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
        patients = sorted(
            patients,
            key=lambda patient: patient.created_at.isoformat()
            if patient.created_at
            else "",
            reverse=True,
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


def _extract_session_dates(text: str) -> dict[int, str]:
    out: dict[int, str] = {}
    for match in re.finditer(
        r"Session\s+(\d+)\s*\((\d{1,2}/\d{1,2}/\d{4})\)", text or ""
    ):
        try:
            local_idx = int(match.group(1))
            iso_date = datetime.strptime(match.group(2), "%m/%d/%Y").date().isoformat()
        except Exception:
            continue
        out.setdefault(local_idx, iso_date)
    return out


def _label_for_page(
    *,
    source_name: str,
    source_page: int,
    session_aliases: dict[int, int],
    session_dates: dict[int, str],
) -> str:
    alias_parts: list[str] = []
    for local_idx, global_idx in sorted(session_aliases.items()):
        date_part = (
            f" ({session_dates[local_idx]})" if local_idx in session_dates else ""
        )
        alias_parts.append(
            f"local Session {local_idx}{date_part} => global Session {global_idx}"
        )
    aliases = "; ".join(alias_parts) if alias_parts else "no aliases"
    return f"source PDF: {source_name}; source page: {source_page}; {aliases}"


def _merged_pdf_bytes(source_paths: list[Path]) -> bytes:
    import fitz

    merged = fitz.open()
    try:
        for source_path in source_paths:
            src = fitz.open(str(source_path))
            try:
                merged.insert_pdf(src)
            finally:
                src.close()
        return merged.tobytes()
    finally:
        merged.close()


def _extract_source(spec: SourceSpec) -> ExtractedSource:
    if not spec.path.exists():
        raise RuntimeError(f"Source PDF not found: {spec.path}")
    extraction = extract_pdf_full(spec.path)
    page_sections = [
        _page_section_body(section)
        for _page, section in _iter_page_sections(extraction.enhanced_text)
    ]
    if len(page_sections) != len(extraction.page_images):
        raise RuntimeError(
            f"Page split mismatch for {spec.path.name}: text has {len(page_sections)} sections, "
            f"images has {len(extraction.page_images)} pages"
        )
    return ExtractedSource(
        spec=spec,
        page_sections=page_sections,
        page_images=extraction.page_images,
        per_page_sources=extraction.per_page_sources,
        metadata=extraction.metadata,
        session_dates=_extract_session_dates(extraction.enhanced_text),
    )


def _write_combined_report(
    *,
    patient_id: str,
    report_id: str,
    manifest: Manifest,
    manifest_path: Path,
    extracted_sources: list[ExtractedSource],
) -> None:
    out_dir = report_dir(patient_id, report_id)
    out_dir.mkdir(parents=True, exist_ok=True)

    total_pages = sum(len(source.page_sections) for source in extracted_sources)
    combined_sections: list[str] = []
    combined_pages_meta: list[dict[str, Any]] = []
    page_labels: dict[str, str] = {}
    page_map: list[dict[str, Any]] = []

    combined_page_num = 0
    pages_dir = report_pages_dir(patient_id, report_id)
    pages_dir.mkdir(parents=True, exist_ok=True)
    sources_dir = out_dir / "sources"
    sources_dir.mkdir(parents=True, exist_ok=True)

    for source in extracted_sources:
        source_name = source.spec.path.name
        source_pages_meta = source.metadata.get("pages")
        source_pages_meta = (
            source_pages_meta if isinstance(source_pages_meta, list) else []
        )

        for idx, section_body in enumerate(source.page_sections, start=1):
            combined_page_num += 1

            alias_lines: list[str] = []
            for local_idx, global_idx in sorted(source.spec.session_aliases.items()):
                alias_line = (
                    f"[[QEEG_SESSION_ALIAS local={local_idx} global={global_idx}"
                )
                date_value = source.session_dates.get(local_idx)
                if date_value:
                    alias_line += f" date={date_value}"
                alias_line += "]]"
                alias_lines.append(alias_line)

            combined_sections.append(
                "\n".join(
                    [
                        f"=== PAGE {combined_page_num} / {total_pages} ===",
                        *alias_lines,
                        section_body.strip() or "[NO TEXT EXTRACTED]",
                    ]
                ).strip()
            )

            page_label = _label_for_page(
                source_name=source_name,
                source_page=idx,
                session_aliases=source.spec.session_aliases,
                session_dates=source.session_dates,
            )
            page_labels[str(combined_page_num)] = page_label
            page_map.append(
                {
                    "combined_page": combined_page_num,
                    "source_file": source_name,
                    "source_page": idx,
                    "session_aliases": {
                        str(local): global_idx
                        for local, global_idx in source.spec.session_aliases.items()
                    },
                    "session_dates": {
                        str(local): value
                        for local, value in source.session_dates.items()
                    },
                }
            )

            image_item = source.page_images[idx - 1]
            image_bytes = base64.b64decode(image_item["base64_png"])
            (pages_dir / f"page-{combined_page_num}.png").write_bytes(image_bytes)

            source_payload = source.per_page_sources[idx - 1]
            (sources_dir / f"page-{combined_page_num}.pypdf.txt").write_text(
                source_payload.get("pypdf_text", ""),
                encoding="utf-8",
            )
            (sources_dir / f"page-{combined_page_num}.pymupdf.txt").write_text(
                source_payload.get("pymupdf_text", ""),
                encoding="utf-8",
            )
            (sources_dir / f"page-{combined_page_num}.apple_vision.txt").write_text(
                source_payload.get("vision_ocr_text", ""),
                encoding="utf-8",
            )
            (sources_dir / f"page-{combined_page_num}.tesseract.txt").write_text(
                source_payload.get("tesseract_ocr_text", ""),
                encoding="utf-8",
            )

            meta_payload = (
                source_pages_meta[idx - 1] if idx - 1 < len(source_pages_meta) else {}
            )
            combined_pages_meta.append(
                {
                    **(meta_payload if isinstance(meta_payload, dict) else {}),
                    "page": combined_page_num,
                    "source_file": source_name,
                    "source_page": idx,
                }
            )

    combined_text = "\n\n".join(combined_sections).strip() + "\n"
    report_extracted_path(patient_id, report_id).write_text(
        combined_text, encoding="utf-8"
    )
    report_enhanced_path(patient_id, report_id).write_text(
        combined_text, encoding="utf-8"
    )

    merged_pdf = _merged_pdf_bytes([source.spec.path for source in extracted_sources])
    report_original_path(patient_id, report_id, manifest.combined_filename).write_bytes(
        merged_pdf
    )

    engine_keys = ("pypdf", "pymupdf", "apple_vision", "tesseract")
    metadata = {
        "schema_version": 2,
        "page_count": total_pages,
        "render_zoom": max(
            [
                float(source.metadata.get("render_zoom", 0.0) or 0.0)
                for source in extracted_sources
            ]
            or [0.0]
        ),
        "engines": {
            key: all(
                bool(source.metadata.get("engines", {}).get(key))
                for source in extracted_sources
            )
            for key in engine_keys
        },
        "pages": combined_pages_meta,
        "has_enhanced_ocr": True,
        "has_page_images": total_pages > 0,
        "page_images_written": total_pages,
        "sources_dir": "sources",
        "synthetic_combined": {
            "manifest_path": str(manifest_path),
            "source_files": [
                {
                    "path": str(source.spec.path),
                    "session_aliases": {
                        str(local): global_idx
                        for local, global_idx in source.spec.session_aliases.items()
                    },
                    "session_dates": {
                        str(local): value
                        for local, value in source.session_dates.items()
                    },
                }
                for source in extracted_sources
            ],
            "page_labels": page_labels,
            "page_map": page_map,
        },
    }
    report_metadata_path(patient_id, report_id).write_text(
        json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8"
    )


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

    _write_combined_report(
        patient_id=patient_id,
        report_id=report_id,
        manifest=manifest,
        manifest_path=manifest_path,
        extracted_sources=extracted_sources,
    )
    _register_report(patient_id=patient_id, report_id=report_id, manifest=manifest)

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
