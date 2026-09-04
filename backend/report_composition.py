"""Shared free PDF asset composition for the API and standalone utility."""

from __future__ import annotations

import base64
import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from .council.report_text import _iter_page_sections, _page_section_body
from .reports import extract_pdf_full, report_dir, report_original_path_in_dir


@dataclass(frozen=True)
class SourceSpec:
    path: Path
    session_aliases: dict[int, int]
    report_id: str = ""
    original_bytes: bytes | None = None


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
    repaired_assets: dict[str, bytes] | None = None


def _session_evidence(enhanced: str) -> list[dict[str, Any]]:
    evidence: dict[int, dict[str, Any]] = {}
    for page, section in _iter_page_sections(enhanced):
        for match in re.finditer(
            r"\bSession\s+(\d+)\b(?:[ \t]*\(([^)\n]+)\)|[ \t]*(?:Date[ \t]*:?|:|-)?[ \t]*(\d{4}-\d{1,2}-\d{1,2}|\d{1,2}/\d{1,2}/\d{4}))?",
            _page_section_body(section),
            re.I,
        ):
            local = int(match.group(1))
            row = evidence.setdefault(
                local,
                {
                    "local_session_index": local,
                    "dates": [],
                    "invalid_dates": [],
                    "pages": [],
                },
            )
            if page not in row["pages"]:
                row["pages"].append(page)
            raw = match.group(2) or match.group(3)
            if raw:
                parsed = None
                for fmt in ("%Y-%m-%d", "%m/%d/%Y"):
                    try:
                        parsed = datetime.strptime(raw.strip(), fmt).date().isoformat()
                        break
                    except ValueError:
                        pass
                field, value = ("dates", parsed) if parsed else ("invalid_dates", raw)
                if value not in row[field]:
                    row[field].append(value)
    return [evidence[k] for k in sorted(evidence)]


def _extract_session_dates(text: str) -> dict[int, str]:
    return {
        row["local_session_index"]: row["dates"][0]
        for row in _session_evidence(text)
        if len(row["dates"]) == 1 and not row["invalid_dates"]
    }


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


def _merged_pdf_bytes(source_paths: list[Path | bytes]) -> bytes:
    import fitz

    merged = fitz.open()
    try:
        for source_path in source_paths:
            src = (
                fitz.open(stream=source_path, filetype="pdf")
                if isinstance(source_path, bytes)
                else fitz.open(str(source_path))
            )
            try:
                merged.insert_pdf(src)
            finally:
                src.close()
        return merged.tobytes(no_new_id=True)
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
    out_dir: Path | None = None,
) -> None:
    out_dir = out_dir or report_dir(patient_id, report_id)
    out_dir.mkdir(parents=True, exist_ok=True)

    total_pages = sum(len(source.page_sections) for source in extracted_sources)
    combined_sections: list[str] = []
    combined_pages_meta: list[dict[str, Any]] = []
    page_labels: dict[str, str] = {}
    page_map: list[dict[str, Any]] = []

    combined_page_num = 0
    pages_dir = out_dir / "pages"
    pages_dir.mkdir(parents=True, exist_ok=True)
    sources_dir = out_dir / "sources"
    sources_dir.mkdir(parents=True, exist_ok=True)

    for source in extracted_sources:
        source_name = source.spec.report_id or source.spec.path.name
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
                    "source_report_id": source.spec.report_id,
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
                    "source_report_id": source.spec.report_id,
                    "source_page": idx,
                }
            )

    combined_text = "\n\n".join(combined_sections).strip() + "\n"
    (out_dir / "extracted.txt").write_text(combined_text, encoding="utf-8")
    (out_dir / "extracted_enhanced.txt").write_text(combined_text, encoding="utf-8")

    merged_pdf = _merged_pdf_bytes(
        [
            source.spec.original_bytes
            if source.spec.original_bytes is not None
            else source.spec.path
            for source in extracted_sources
        ]
    )
    report_original_path_in_dir(out_dir, manifest.combined_filename).write_bytes(merged_pdf)

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
    (out_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8"
    )
