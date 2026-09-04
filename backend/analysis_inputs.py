"""Immutable, free analysis input admission and recoverable PDF composition.

The operation row is committed before composition. Its IDs and input snapshot are
reused after a lost response or process exit; only a fully promoted report can
be registered as a run. File locks coordinate workers without holding a SQLite
write transaction during PDF work.
"""

from __future__ import annotations

import base64
from contextlib import contextmanager
from dataclasses import replace
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
from typing import Any
import uuid

from fastapi import HTTPException
from sqlalchemy import select, text as sql_text

from . import reports, storage
from .council.report_text import (
    _facts_from_report_text_summary,
    _facts_from_report_text_n100_central_frontal,
    _page_count_from_markers,
    _iter_page_sections,
    _page_section_body,
)
from .report_composition import (
    ExtractedSource,
    Manifest,
    SourceSpec,
    _write_combined_report,
    _session_evidence,
)

COMPOSITION_VERSION = 1
_ENGINES = {
    "pypdf": "pypdf_text",
    "pymupdf": "pymupdf_text",
    "apple_vision": "vision_ocr_text",
    "tesseract": "tesseract_ocr_text",
}


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def _digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _fingerprint(value: Any) -> str:
    return _digest(_canonical(value).encode("utf-8"))


def normalize_source_ids(
    report_id: str | None, report_ids: list[str], *, list_supplied: bool = False
) -> list[str]:
    if report_id is not None and list_supplied and report_ids != [report_id]:
        raise HTTPException(
            400, "report_id and report_ids must name the same single source"
        )
    ids = (
        list(report_ids)
        if list_supplied or report_ids
        else ([report_id] if report_id else [])
    )
    if not ids or any(not rid or not rid.strip() for rid in ids):
        raise HTTPException(400, "A non-empty report_ids list or report_id is required")
    if len(set(ids)) != len(ids):
        raise HTTPException(400, "Duplicate report IDs are not allowed")
    return ids


def _source_snapshot(
    report: storage.Report,
) -> tuple[dict[str, Any], ExtractedSource | None]:
    """Use validated saved extraction; repair missing source assets in memory only."""
    original = Path(report.stored_path)
    directory = original.parent
    try:
        original_bytes = original.read_bytes()
        original_hash = _digest(original_bytes)
        if original.suffix.lower() != ".pdf":
            extracted = Path(report.extracted_text_path).read_bytes()
            return {
                "report_id": report.id,
                "original_sha256": original_hash,
                "asset_digests": {"extracted.txt": _digest(extracted)},
                "page_count": 0,
                "session_evidence": [],
                "session_aliases": {},
            }, None
        import fitz

        with fitz.open(original) as pdf:
            page_count = len(pdf)
        if not page_count:
            raise ValueError("PDF contains no pages")
        asset_bytes = {}
        repaired = False
        try:
            if (directory / ".analysis_input_repair_pending").exists():
                raise ValueError("Source extraction repair was interrupted")
            asset_bytes["extracted.txt"] = Path(report.extracted_text_path).read_bytes()
            asset_bytes["extracted_enhanced.txt"] = (
                directory / "extracted_enhanced.txt"
            ).read_bytes()
            asset_bytes["metadata.json"] = (directory / "metadata.json").read_bytes()
            metadata = json.loads(asset_bytes["metadata.json"])
            enhanced = asset_bytes["extracted_enhanced.txt"].decode("utf-8")
            sections = _iter_page_sections(enhanced)
            if (
                [p for p, _ in sections] != list(range(1, page_count + 1))
                or _page_count_from_markers(enhanced) != page_count
                or metadata.get("page_count") != page_count
                or metadata.get("schema_version") != 2
            ):
                raise ValueError("Incomplete extraction metadata/page markers")
            images, streams = [], []
            for n in range(1, page_count + 1):
                key = f"pages/page-{n}.png"
                asset_bytes[key] = (directory / key).read_bytes()
                fitz.Pixmap(asset_bytes[key])  # decode, not merely an existence check
                images.append(
                    {
                        "page": n,
                        "base64_png": base64.b64encode(asset_bytes[key]).decode(
                            "ascii"
                        ),
                    }
                )
                stream = {"page": n}
                for engine, field in _ENGINES.items():
                    key = f"sources/page-{n}.{engine}.txt"
                    asset_bytes[key] = (directory / key).read_bytes()
                    stream[field] = asset_bytes[key].decode("utf-8")
                streams.append(stream)
        except (OSError, ValueError, TypeError, KeyError, RuntimeError):
            repaired = True
            full = reports.extract_pdf_full(original)
            enhanced, images, streams, metadata = (
                full.enhanced_text,
                full.page_images,
                full.per_page_sources,
                full.metadata,
            )
            sections = _iter_page_sections(enhanced)
            if (
                [p for p, _ in sections] != list(range(1, page_count + 1))
                or len(images) != page_count
                or len(streams) != page_count
            ):
                raise ValueError("Extraction did not cover every PDF page")
            asset_bytes = {
                "extracted.txt": enhanced.encode(),
                "extracted_enhanced.txt": enhanced.encode(),
                "metadata.json": _canonical(metadata).encode(),
            }
            for n, (image, stream) in enumerate(zip(images, streams), 1):
                asset_bytes[f"pages/page-{n}.png"] = base64.b64decode(
                    image["base64_png"]
                )
                for engine, field in _ENGINES.items():
                    asset_bytes[f"sources/page-{n}.{engine}.txt"] = stream.get(
                        field, ""
                    ).encode()
        evidence = _session_evidence(enhanced)
        snapshot = {
            "report_id": report.id,
            "original_sha256": original_hash,
            "asset_digests": {k: _digest(v) for k, v in sorted(asset_bytes.items())},
            "page_count": page_count,
            "session_evidence": evidence,
            "session_aliases": {},
        }
        extracted = ExtractedSource(
            SourceSpec(original, {}, report.id, original_bytes),
            [_page_section_body(s) for _, s in sections],
            images,
            streams,
            metadata,
            {},
            asset_bytes if repaired else None,
        )
        return snapshot, extracted
    except (OSError, ValueError, TypeError, RuntimeError) as exc:
        raise HTTPException(
            400,
            {
                "code": "ANALYSIS_SOURCE_INVALID",
                "report_id": report.id,
                "message": str(exc),
            },
        ) from exc


def _measurements(source: ExtractedSource, local: int) -> dict[str, set[str]]:
    text = "\n\n".join(
        f"=== PAGE {n} / {len(source.page_sections)} ===\n{s}"
        for n, s in enumerate(source.page_sections, 1)
    )
    indices = [r["local_session_index"] for r in _session_evidence(text)]
    facts = _facts_from_report_text_summary(text, expected_sessions=indices)
    facts += _facts_from_report_text_n100_central_frontal(
        text, expected_sessions=indices
    )
    result: dict[str, set[str]] = {}
    for fact in facts:
        if fact.get("session_index") == local:
            key = _canonical(
                [
                    fact.get(k)
                    for k in ("fact_type", "metric", "electrode", "condition", "unit")
                ]
            )
            result.setdefault(key, set()).add(
                _canonical([fact.get(k) for k in ("value", "sd_plus_minus")])
            )
    return result


def _mapping_error(snapshots: list[dict], reason: str) -> None:
    raise HTTPException(
        409,
        {
            "code": "ANALYSIS_SESSION_MAPPING_REQUIRED",
            "reason": reason,
            "sources": [
                {"report_id": s["report_id"], "sessions": s["session_evidence"]}
                for s in snapshots
            ],
        },
    )


def resolve_aliases(
    snapshots: list[dict],
    extracted: list[ExtractedSource | None],
    explicit: dict[str, dict[str, int]],
) -> list[ExtractedSource | None]:
    ids = {s["report_id"] for s in snapshots}
    if set(explicit) - ids:
        _mapping_error(snapshots, "Aliases name a report outside this request")
    multi = len(snapshots) > 1
    rows = [
        (i, row) for i, snap in enumerate(snapshots) for row in snap["session_evidence"]
    ]
    if multi and any(src is None for src in extracted):
        raise HTTPException(400, "Combined analysis sources must be PDFs")
    if multi and any(not s["session_evidence"] for s in snapshots):
        _mapping_error(snapshots, "A source has no observed local sessions")
    aliases: dict[tuple[int, int], int] = {}
    if explicit:
        for i, snap in enumerate(snapshots):
            raw = explicit.get(snap["report_id"], {})
            observed = {str(r["local_session_index"]) for r in snap["session_evidence"]}
            if set(raw) != observed or any(
                type(v) is not int or v <= 0 for v in raw.values()
            ):
                _mapping_error(
                    snapshots,
                    "Explicit aliases must cover every observed local session with positive integer labels",
                )
            aliases.update({(i, int(k)): v for k, v in raw.items()})
    elif multi:
        if any(len(r["dates"]) != 1 or r["invalid_dates"] for _, r in rows):
            _mapping_error(
                snapshots,
                "Missing, invalid, or conflicting session dates leave chronology unresolved",
            )
        dates = sorted({r["dates"][0] for _, r in rows})
        aliases = {
            (i, r["local_session_index"]): dates.index(r["dates"][0]) + 1
            for i, r in rows
        }
    else:
        aliases = {
            (i, r["local_session_index"]): r["local_session_index"] for i, r in rows
        }
    # A repeated date is corroborated with shared measured values. Different
    # measurements remain distinct visits unless the operator supplies a map.
    groups: dict[int, list[tuple[int, dict]]] = {}
    for i, row in rows:
        groups.setdefault(aliases[i, row["local_session_index"]], []).append((i, row))
    for group in groups.values():
        for n, (i, row) in enumerate(group):
            for j, other in group[:n]:
                if i == j:
                    _mapping_error(
                        snapshots,
                        "Distinct local sessions in one source cannot be merged",
                    )
                if (
                    row["dates"]
                    and other["dates"]
                    and set(row["dates"]) != set(other["dates"])
                ):
                    _mapping_error(
                        snapshots,
                        "A shared global session has conflicting source dates",
                    )
                left = _measurements(extracted[i], row["local_session_index"])
                right = _measurements(extracted[j], other["local_session_index"])
                shared = left.keys() & right.keys()
                if any(
                    left[key] != right[key] or len(left[key]) != 1 for key in shared
                ):
                    _mapping_error(
                        snapshots,
                        "A shared global session has conflicting measured values",
                    )
                if not shared and not explicit:
                    _mapping_error(
                        snapshots,
                        "A repeated date lacks consistent shared measurements to establish one visit",
                    )
    result = []
    for i, (snap, src) in enumerate(zip(snapshots, extracted)):
        mapping = {
            r["local_session_index"]: aliases[i, r["local_session_index"]]
            for r in snap["session_evidence"]
        }
        snap["session_aliases"] = {str(k): v for k, v in mapping.items()}
        snap["mapping_provenance"] = (
            "operator"
            if explicit
            else (
                "source_dates_and_measurements" if multi else "original_local_sessions"
            )
        )
        dates = {
            r["local_session_index"]: r["dates"][0]
            for r in snap["session_evidence"]
            if len(r["dates"]) == 1 and not r["invalid_dates"]
        }
        result.append(
            replace(
                src,
                spec=replace(src.spec, session_aliases=mapping),
                session_dates=dates,
            )
            if src
            else None
        )
    return result


@contextmanager
def _operation_lock(operation_id: str):
    directory = storage.DATA_DIR / "analysis_input_locks"
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / _digest(operation_id.encode())).open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def _asset_inventory(directory: Path) -> dict[str, str]:
    return {
        str(p.relative_to(directory)): _digest(p.read_bytes())
        for p in sorted(directory.rglob("*"))
        if p.is_file() and p.name != "analysis_input_ready.json"
    }


def _ready(directory: Path, fingerprint: str) -> bool:
    try:
        ready = json.loads((directory / "analysis_input_ready.json").read_text())
        return ready["fingerprint"] == fingerprint and ready[
            "assets"
        ] == _asset_inventory(directory)
    except (OSError, ValueError, KeyError):
        return False


def _compose(
    manifest: dict, extracted: list[ExtractedSource | None], fingerprint: str
) -> Path:
    destination = reports.report_dir(
        manifest["patient_id"], manifest["execution_report_id"]
    )
    if _ready(destination, fingerprint):
        return destination
    # Stable staging and backup locations make promotion recoverable on restart.
    stage = destination.with_name(destination.name + ".staging")
    backup = destination.with_name(destination.name + ".previous")
    if not _ready(stage, fingerprint):
        if stage.exists():
            shutil.rmtree(stage)
        recipe = Manifest(
            "", "combined_council_report.pdf", "", [s.spec for s in extracted], [], None
        )
        _write_combined_report(
            patient_id=manifest["patient_id"],
            report_id=manifest["execution_report_id"],
            manifest=recipe,
            manifest_path=destination / "analysis_input_manifest.json",
            extracted_sources=extracted,
            out_dir=stage,
        )
        (stage / "analysis_input_manifest.json").write_text(_canonical(manifest))
        metadata = json.loads((stage / "metadata.json").read_text())
        if (
            metadata["page_count"] != len(manifest["page_map"])
            or metadata["synthetic_combined"]["page_map"] != manifest["page_map"]
        ):
            raise RuntimeError("Composed page map failed validation")
        ready = {"fingerprint": fingerprint, "assets": _asset_inventory(stage)}
        (stage / "analysis_input_ready.json").write_text(_canonical(ready))
    if destination.exists():
        if backup.exists():
            shutil.rmtree(backup)
        os.replace(destination, backup)
    os.replace(stage, destination)
    if backup.exists():
        shutil.rmtree(backup)
    return destination


def _page_map(extracted: list[ExtractedSource | None]) -> list[dict]:
    result = []
    for src in extracted:
        if src is None:
            continue
        for page in range(1, len(src.page_sections) + 1):
            result.append(
                {
                    "combined_page": len(result) + 1,
                    "source_file": src.spec.report_id,
                    "source_report_id": src.spec.report_id,
                    "source_page": page,
                    "session_aliases": {
                        str(k): v for k, v in src.spec.session_aliases.items()
                    },
                    "session_dates": {str(k): v for k, v in src.session_dates.items()},
                }
            )
    return result


def _operation_conflict() -> HTTPException:
    return HTTPException(
        409,
        {
            "code": "ANALYSIS_OPERATION_CONFLICT",
            "message": "This operation was reserved for different immutable inputs or models",
        },
    )


def admit_run(
    *,
    patient_id: str,
    source_ids: list[str],
    special_instructions: str,
    source_session_aliases: dict[str, dict[str, int]],
    operation_id: str | None,
    model_fields: dict[str, Any],
) -> storage.Run:
    if operation_id is not None and not operation_id.strip():
        raise HTTPException(400, "operation_id must be non-empty when supplied")
    key = operation_id if operation_id is not None else str(uuid.uuid4())
    envelope_fingerprint = _fingerprint(
        {
            "patient_id": patient_id,
            "source_ids": source_ids,
            "special_instructions": special_instructions,
            "requested_model_ids": model_fields["requested_model_ids"],
            "resolved_model_ids": model_fields["resolved_model_ids"],
        }
    )
    with _operation_lock(key):
        with storage.session_scope() as session:
            prior = session.get(storage.AnalysisInputReservation, key)
            if prior and prior.envelope_fingerprint != envelope_fingerprint:
                raise _operation_conflict()
            if storage.get_patient(session, patient_id) is None:
                raise HTTPException(404, "Patient not found")
            originals = []
            for rid in source_ids:
                report = storage.get_report(session, rid)
                if report is None:
                    raise HTTPException(404, "Report not found")
                if report.patient_id != patient_id:
                    raise HTTPException(400, "Report does not belong to patient")
                originals.append(report)
        try:
            snapshots, extracted = map(
                list, zip(*[_source_snapshot(r) for r in originals])
            )
        except HTTPException as exc:
            if prior:
                raise _operation_conflict() from exc
            raise
        extracted = resolve_aliases(snapshots, extracted, source_session_aliases)
        # Resolution provenance is descriptive, not a different immutable mapping.
        fingerprint = _fingerprint(
            {
                "composition_version": COMPOSITION_VERSION,
                "patient_id": patient_id,
                "sources": [
                    {k: v for k, v in s.items() if k != "mapping_provenance"}
                    for s in snapshots
                ],
                "special_instructions": special_instructions,
            }
        )
        request_hash = _fingerprint(
            {
                "analysis_input_fingerprint": fingerprint,
                "requested_model_ids": model_fields["requested_model_ids"],
                "resolved_model_ids": model_fields["resolved_model_ids"],
            }
        )
        composed = len(source_ids) > 1 or any(
            int(k) != v for s in snapshots for k, v in s["session_aliases"].items()
        )
        with storage.session_scope() as session:
            session.execute(sql_text("BEGIN IMMEDIATE"))
            reservation = session.get(storage.AnalysisInputReservation, key)
            if reservation:
                if reservation.request_fingerprint != request_hash:
                    raise HTTPException(
                        409,
                        {
                            "code": "ANALYSIS_OPERATION_CONFLICT",
                            "message": "This operation was reserved for different immutable inputs or models",
                        },
                    )
                run = storage.get_run(session, reservation.run_id)
                if run:
                    return run
            else:
                report_id = str(uuid.uuid4()) if composed else source_ids[0]
                manifest = {
                    "schema_version": 1,
                    "composition_version": COMPOSITION_VERSION,
                    "legacy": False,
                    "patient_id": patient_id,
                    "execution_report_id": report_id,
                    "source_report_ids": source_ids,
                    "sources": snapshots,
                    "page_map": _page_map(extracted),
                    "analysis_input_fingerprint": fingerprint,
                }
                reservation = storage.AnalysisInputReservation(
                    operation_id=key,
                    request_fingerprint=request_hash,
                    envelope_fingerprint=envelope_fingerprint,
                    manifest_json=_canonical(manifest),
                    report_id=report_id,
                    run_id=str(uuid.uuid4()),
                )
                session.add(reservation)
            session.commit()
        manifest = json.loads(reservation.manifest_json)
        for original, src in zip(originals, extracted):
            if src and src.repaired_assets:
                pending = (
                    Path(original.stored_path).parent / ".analysis_input_repair_pending"
                )
                pending.touch()
                for name, content in src.repaired_assets.items():
                    target = (
                        Path(original.extracted_text_path)
                        if name == "extracted.txt"
                        else Path(original.stored_path).parent / name
                    )
                    target.parent.mkdir(parents=True, exist_ok=True)
                    temporary = target.with_name(
                        target.name + ".admission-" + uuid.uuid4().hex
                    )
                    temporary.write_bytes(content)
                    os.replace(temporary, target)
                pending.unlink()
        if composed:
            directory = _compose(manifest, extracted, fingerprint)
        with storage.session_scope() as session:
            if composed and storage.get_report(session, reservation.report_id) is None:
                storage.create_report(
                    session,
                    report_id=reservation.report_id,
                    patient_id=patient_id,
                    filename="combined_council_report.pdf",
                    mime_type="application/pdf",
                    stored_path=directory / "original.pdf",
                    extracted_text_path=directory / "extracted.txt",
                )
            return storage.create_run(
                session,
                patient_id=patient_id,
                report_id=reservation.report_id,
                source_report_ids=source_ids,
                source_manifest=manifest,
                special_instructions=special_instructions,
                analysis_input_fingerprint=fingerprint,
                operation_id=operation_id,
                run_id=reservation.run_id,
                **model_fields,
            )


def repair_combined_report(
    report: storage.Report, *, run_id: str | None = None
) -> bool:
    """Repair only through the banked original-source snapshot; never OCR the merge."""
    with storage.session_scope() as session:
        run = (
            storage.get_run(session, run_id)
            if run_id
            else session.scalar(
                select(storage.Run).where(storage.Run.report_id == report.id)
            )
        )
        manifest = json.loads(run.source_manifest_json) if run else {}
        if manifest.get("legacy", True) or manifest.get(
            "execution_report_id"
        ) in manifest.get("source_report_ids", []):
            return False
        originals = [
            storage.get_report(session, rid) for rid in manifest["source_report_ids"]
        ]
    with _operation_lock(run.operation_id or run.id):
        if _ready(Path(report.stored_path).parent, run.analysis_input_fingerprint):
            return True
        extracted = []
        for original, saved in zip(originals, manifest["sources"]):
            if original is None or original.patient_id != report.patient_id:
                raise RuntimeError(
                    "The original report is unavailable for combined asset recovery"
                )
            current, src = _source_snapshot(original)
            if any(
                current[k] != saved[k]
                for k in ("report_id", "original_sha256", "asset_digests")
            ):
                raise RuntimeError(
                    "Original source assets changed since admission; preserve this run and create a new authorized run"
                )
            aliases = {int(k): v for k, v in saved["session_aliases"].items()}
            dates = {
                r["local_session_index"]: r["dates"][0]
                for r in saved["session_evidence"]
                if len(r["dates"]) == 1 and not r["invalid_dates"]
            }
            extracted.append(
                replace(
                    src,
                    spec=replace(src.spec, session_aliases=aliases),
                    session_dates=dates,
                )
            )
        _compose(manifest, extracted, run.analysis_input_fingerprint)
        return True


def saved_operator_instructions(run_id: str) -> str:
    with storage.session_scope() as session:
        run = storage.get_run(session, run_id)
        return run.special_instructions if run else ""


def validate_admitted_run(run: storage.Run) -> None:
    """An admitted source snapshot cannot silently change before a new start."""
    manifest = json.loads(run.source_manifest_json)
    if manifest.get("legacy", True):
        return
    with storage.session_scope() as session:
        for saved in manifest["sources"]:
            original = storage.get_report(session, saved["report_id"])
            if original is None or original.patient_id != run.patient_id:
                raise HTTPException(
                    409,
                    {"code": "ANALYSIS_INPUT_CHANGED", "report_id": saved["report_id"]},
                )
            try:
                current, _ = _source_snapshot(original)
            except HTTPException as exc:
                raise HTTPException(
                    409,
                    {"code": "ANALYSIS_INPUT_CHANGED", "report_id": saved["report_id"]},
                ) from exc
            if any(
                current[k] != saved[k]
                for k in ("report_id", "original_sha256", "asset_digests")
            ):
                raise HTTPException(
                    409,
                    {"code": "ANALYSIS_INPUT_CHANGED", "report_id": saved["report_id"]},
                )
        execution = storage.get_report(session, run.report_id)
    if execution is None:
        raise HTTPException(
            409, {"code": "ANALYSIS_INPUT_CHANGED", "report_id": run.report_id}
        )
    repair_combined_report(execution, run_id=run.id)
