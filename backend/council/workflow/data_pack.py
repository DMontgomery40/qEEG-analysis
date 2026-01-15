from __future__ import annotations

import json
import os
import random
import re
from pathlib import Path
from typing import Any

from ...storage import Artifact, Report, create_artifact
from ...storage import session_scope
from ..constants import DATA_PACK_SCHEMA_VERSION, STAGES
from ..json_utils import _json_loads_loose
from ..paths import _data_pack_path, _stage_dir, _vision_transcript_path
from ..prompts import _data_pack_prompt
from ..report_text import (
    _expected_session_indices,
    _facts_from_report_text_n100_central_frontal,
    _facts_from_report_text_summary,
    _find_p300_rare_comparison_pages,
)
from ..utils import _chunked
from ..vision import _save_debug_images, _try_build_p300_cp_site_crops, _try_build_summary_table_crops


class _DataPackMixin:
    @staticmethod
    def _dedupe_facts(facts: list[dict[str, Any]]) -> list[dict[str, Any]]:
        deduped: list[dict[str, Any]] = []
        seen: set[str] = set()
        for fact in facts:
            ftype = fact.get("fact_type")
            if not isinstance(ftype, str):
                continue
            key_parts = [ftype]
            for k in ("metric", "site", "region", "session_index"):
                v = fact.get(k)
                if v is None:
                    continue
                key_parts.append(f"{k}={v}")
            key_parts.append(f"page={fact.get('source_page')}")
            key = "|".join(key_parts)
            if key in seen:
                continue
            seen.add(key)
            deduped.append(fact)
        return deduped

    @staticmethod
    def _fact_key(fact: dict[str, Any]) -> str | None:
        ftype = fact.get("fact_type")
        if not isinstance(ftype, str):
            return None
        key_parts = [ftype]
        for k in ("metric", "site", "region", "session_index"):
            v = fact.get(k)
            if v is None:
                continue
            key_parts.append(f"{k}={v}")
        key_parts.append(f"page={fact.get('source_page')}")
        return "|".join(key_parts)

    @staticmethod
    def _fact_numeric_signature(fact: dict[str, Any]) -> tuple[Any, ...] | None:
        """
        Return a tuple representing the numeric payload of a fact for conflict detection.

        We intentionally ignore incidental fields (e.g., labels, units, target ranges) and focus on required numbers.
        """
        def coerce(val: Any) -> Any:
            if isinstance(val, bool):
                return val
            if isinstance(val, (int, float)):
                if isinstance(val, float) and val.is_integer():
                    return int(val)
                return val
            if isinstance(val, str):
                s = val.strip().replace("−", "-")
                if re.fullmatch(r"-?\d+(?:\.\d+)?", s):
                    try:
                        num = float(s)
                    except Exception:
                        return val
                    return int(num) if num.is_integer() else num
            return val

        shown_as = fact.get("shown_as")
        if isinstance(shown_as, str) and shown_as.strip().upper() == "N/A":
            return ("NA",)
        ftype = fact.get("fact_type")
        if ftype in {"performance_metric", "evoked_potential", "state_metric", "peak_frequency"}:
            val = coerce(fact.get("value"))
            if val is None:
                return None
            return ("value", val)
        if ftype == "p300_cp_site":
            uv = coerce(fact.get("uv"))
            ms = coerce(fact.get("ms"))
            if uv is None or ms is None:
                return None
            return ("uv_ms", uv, ms)
        if ftype == "n100_central_frontal_average":
            uv = coerce(fact.get("uv"))
            ms = coerce(fact.get("ms"))
            if uv is None or ms is None:
                return None
            return ("uv_ms", uv, ms)
        # Default: best-effort, include common numeric fields if present.
        payload: list[Any] = ["generic"]
        for k in ("value", "uv", "ms", "yield"):
            if k in fact:
                payload.append(coerce(fact.get(k)))
        return tuple(payload)

    @classmethod
    def _find_fact_conflicts(cls, facts: list[dict[str, Any]]) -> list[dict[str, Any]]:
        by_key: dict[str, list[dict[str, Any]]] = {}
        for f in facts:
            if not isinstance(f, dict):
                continue
            key = cls._fact_key(f)
            if not key:
                continue
            by_key.setdefault(key, []).append(f)

        conflicts: list[dict[str, Any]] = []
        for key, items in by_key.items():
            if len(items) < 2:
                continue
            sigs: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
            for f in items:
                sig = cls._fact_numeric_signature(f)
                if sig is None:
                    continue
                sigs.setdefault(sig, []).append(f)
            if len(sigs) > 1:
                conflicts.append(
                    {
                        "key": key,
                        "variants": [
                            {
                                "signature": list(sig),
                                "facts": group,
                            }
                            for sig, group in sigs.items()
                        ],
                    }
                )
        return conflicts

    async def _ensure_data_pack(
        self,
        *,
        run_id: str,
        report: Report,
        report_text: str,
        page_images: list[PageImage],
        candidate_extractor_model_ids: list[str],
        strict: bool,
    ) -> dict[str, Any] | None:
        """
        Build (or load) a structured data pack that makes image-only metrics available to ALL stages.

        Notes:
        - This runs multi-pass multimodal extraction across ALL available page images.
        - If strict is True, missing required numeric fields raises an error (no silent degradation).
        """
        out_path = _data_pack_path(run_id)
        expected_sessions = _expected_session_indices(report_text)
        expected_pages = sorted({img.page for img in page_images if isinstance(img.page, int)})

        # If an existing data pack is present, upgrade it in-place with deterministic facts and derived views.
        if out_path.exists():
            try:
                existing = json.loads(out_path.read_text(encoding="utf-8"))
            except Exception:
                existing = None

            if isinstance(existing, dict) and existing.get("schema_version") == DATA_PACK_SCHEMA_VERSION:
                # If the report was re-extracted or page images changed, ensure the cached data pack actually
                # covers the current pages (older runs could have partial page coverage).
                try:
                    existing_pages = existing.get("pages_processed") or existing.get("pages_seen") or []
                    existing_pages_set = {p for p in existing_pages if isinstance(p, int)}
                except Exception:
                    existing_pages_set = set()
                existing_page_count = existing.get("page_count")
                if expected_pages and (
                    existing_page_count != len(expected_pages) or not existing_pages_set.issuperset(set(expected_pages))
                ):
                    existing = None

            if isinstance(existing, dict) and existing.get("schema_version") == DATA_PACK_SCHEMA_VERSION:
                facts = existing.get("facts")
                if isinstance(facts, list):
                    base_facts = [f for f in facts if isinstance(f, dict)]
                    add_facts: list[dict[str, Any]] = []
                    add_facts.extend(_facts_from_report_text_summary(report_text, expected_sessions=expected_sessions))
                    add_facts.extend(_facts_from_report_text_n100_central_frontal(report_text, expected_sessions=expected_sessions))
                    for f in add_facts:
                        if isinstance(f, dict):
                            f.setdefault("extraction_method", "deterministic_report_text")
                    for f in base_facts:
                        if isinstance(f, dict):
                            f.setdefault("extraction_method", f.get("extraction_method") or "vision_llm")
                    conflicts = self._find_fact_conflicts(add_facts + base_facts)
                    if conflicts and strict:
                        # Cached pack is inconsistent; rebuild from scratch so strict runs are never silently
                        # grounded in conflicting numbers.
                        existing = None
                    else:
                        # Deterministic facts come first so they override duplicates from older model output.
                        existing["facts"] = self._dedupe_facts(add_facts + base_facts)

            if isinstance(existing, dict) and existing.get("schema_version") == DATA_PACK_SCHEMA_VERSION:
                existing["derived"] = self._derive_data_pack_views(existing, expected_sessions=expected_sessions)
                missing = self._missing_required_fields(existing, expected_sessions=expected_sessions)
                if not (missing and strict):
                    try:
                        out_path.parent.mkdir(parents=True, exist_ok=True)
                        out_path.write_text(json.dumps(existing, indent=2, sort_keys=True), encoding="utf-8")
                    except Exception:
                        pass
                    return existing

        if not page_images or not candidate_extractor_model_ids:
            if strict:
                report_dir = _derive_report_dir(report)
                raise RuntimeError(
                    "Strict data availability requested, but Stage 1 multimodal extraction cannot run.\n"
                    f"Report: {report.filename} ({report.id})\n"
                    f"Vision-capable extractor models selected: {candidate_extractor_model_ids}\n"
                    f"Page images available: {len(page_images)}\n"
                    f"Expected page images directory: {report_dir / 'pages'}\n"
                    "If report assets look incomplete, re-run: POST /api/reports/{report_id}/reextract"
                )
            return None

        chunk_size = int(os.getenv("QEEG_VISION_PAGES_PER_CALL", "8") or "8")
        if chunk_size <= 0:
            chunk_size = 8
        # Hard requirement: PDFs >10 pages must be processed in 2+ multimodal passes.
        if len(page_images) > 10 and chunk_size > 10:
            chunk_size = 10

        debug_dir: Path | None = None
        attempt_log: list[dict[str, Any]] = []
        if strict or _truthy_env("QEEG_SAVE_DATA_PACK_DEBUG", False):
            try:
                debug_dir = _stage_dir(run_id, 1) / "_data_pack_debug"
                debug_dir.mkdir(parents=True, exist_ok=True)
            except Exception:
                debug_dir = None

        def apply_deterministic_facts(merged: dict[str, Any]) -> list[dict[str, Any]]:
            merged_facts = merged.get("facts")
            if not isinstance(merged_facts, list):
                return []
            model_facts: list[dict[str, Any]] = [f for f in merged_facts if isinstance(f, dict)]
            for f in model_facts:
                f.setdefault("extraction_method", "vision_llm")

            add_facts: list[dict[str, Any]] = []
            add_facts.extend(_facts_from_report_text_summary(report_text, expected_sessions=expected_sessions))
            add_facts.extend(_facts_from_report_text_n100_central_frontal(report_text, expected_sessions=expected_sessions))
            for f in add_facts:
                if isinstance(f, dict):
                    f.setdefault("extraction_method", "deterministic_report_text")

            combined = add_facts + model_facts
            if combined:
                # Deterministic facts come first so they override duplicates from model output.
                merged["facts"] = self._dedupe_facts(combined)
            return self._find_fact_conflicts(combined)

        # Multi-pass extraction across all pages.
        errors: list[str] = []
        for extractor_model_id in candidate_extractor_model_ids:
            try:
                parts: list[dict[str, Any]] = []
                for chunk_index, chunk in enumerate(_chunked(page_images, chunk_size), start=1):
                    pages = [img.page for img in chunk]
                    prompt_text = _data_pack_prompt(
                        pages=pages,
                        focus=(
                            "Extract required numeric facts (performance, P300 delay/voltage, CP per-site P300, N100, "
                            "theta/beta, alpha ratio, peak frequency) from ONLY these pages."
                        ),
                    )
                    raw = await self._call_model_multimodal(
                        model_id=extractor_model_id,
                        prompt_text=prompt_text,
                        images=chunk,
                        temperature=0.0,
                        max_tokens=3500,
                        allow_text_fallback=False,
                    )
                    if debug_dir is not None:
                        try:
                            safe_model = "".join(
                                c if c.isalnum() or c in {"-", "_", "."} else "_" for c in extractor_model_id
                            ) or "model"
                            model_dir = debug_dir / safe_model
                            model_dir.mkdir(parents=True, exist_ok=True)
                            pages_key = "-".join(str(p) for p in pages)
                            stem = f"pass-{chunk_index:02d}-pages-{pages_key}"
                            (model_dir / f"{stem}.prompt.txt").write_text(prompt_text, encoding="utf-8")
                            (model_dir / f"{stem}.raw.txt").write_text(raw, encoding="utf-8")
                            image_paths = _save_debug_images(model_dir=model_dir, stem=stem, images=chunk)
                            attempt_log.append(
                                {
                                    "kind": "chunk",
                                    "model_id": extractor_model_id,
                                    "chunk_index": chunk_index,
                                    "pages": pages,
                                    "raw_path": str(model_dir / f"{stem}.raw.txt"),
                                    "image_paths": image_paths,
                                }
                            )
                        except Exception:
                            pass
                    part = _json_loads_loose(raw)
                    if not isinstance(part, dict):
                        raise ValueError("Data pack part must be a JSON object")
                    if part.get("schema_version") != DATA_PACK_SCHEMA_VERSION:
                        raise ValueError("Data pack schema_version mismatch")
                    if debug_dir is not None:
                        try:
                            safe_model = "".join(
                                c if c.isalnum() or c in {"-", "_", "."} else "_" for c in extractor_model_id
                            ) or "model"
                            model_dir = debug_dir / safe_model
                            pages_key = "-".join(str(p) for p in pages)
                            stem = f"pass-{chunk_index:02d}-pages-{pages_key}"
                            (model_dir / f"{stem}.parsed.json").write_text(
                                json.dumps(part, indent=2, sort_keys=True), encoding="utf-8"
                            )
                        except Exception:
                            pass
                    parts.append(part)

                merged = self._merge_data_pack_parts(
                    parts,
                    meta={
                        "schema_version": DATA_PACK_SCHEMA_VERSION,
                        "run_id": run_id,
                        "report_id": report.id,
                        "report_filename": report.filename,
                        "extraction_model_id": extractor_model_id,
                        "expected_session_indices": expected_sessions,
                        "page_count": len(page_images),
                        "pages_processed": [img.page for img in page_images],
                        "pages_per_call": chunk_size,
                    },
                )

                # Fill in deterministic facts from OCR/PDF text (page-grounded markers) to reduce flakiness.
                fact_conflicts = apply_deterministic_facts(merged)

                merged["derived"] = self._derive_data_pack_views(merged, expected_sessions=expected_sessions)

                missing = self._missing_required_fields(merged, expected_sessions=expected_sessions)
                if missing:
                    # Targeted retry for the most common failure mode: CP per-site P300 values.
                    candidate_pages = _find_p300_rare_comparison_pages(report_text, page_count=len(page_images))
                    merged = await self._targeted_retry_missing(
                        extractor_model_id=extractor_model_id,
                        page_images=page_images,
                        merged=merged,
                        expected_sessions=expected_sessions,
                        missing=missing,
                        candidate_pages=candidate_pages,
                        debug_dir=debug_dir,
                        attempt_log=attempt_log,
                    )
                    fact_conflicts = apply_deterministic_facts(merged)
                    merged["derived"] = self._derive_data_pack_views(merged, expected_sessions=expected_sessions)
                    missing = self._missing_required_fields(merged, expected_sessions=expected_sessions)

                if missing:
                    merged = await self._targeted_retry_summary_missing(
                        extractor_model_id=extractor_model_id,
                        page_images=page_images,
                        merged=merged,
                        expected_sessions=expected_sessions,
                        missing=missing,
                        debug_dir=debug_dir,
                        attempt_log=attempt_log,
                    )
                    fact_conflicts = apply_deterministic_facts(merged)
                    merged["derived"] = self._derive_data_pack_views(merged, expected_sessions=expected_sessions)
                    missing = self._missing_required_fields(merged, expected_sessions=expected_sessions)

                if strict and (missing or fact_conflicts):
                    failure_path = _stage_dir(run_id, 1) / "_data_pack_failure.json"
                    try:
                        failure_payload = {
                            "run_id": run_id,
                            "report_id": report.id,
                            "report_filename": report.filename,
                            "extraction_model_id": extractor_model_id,
                            "expected_session_indices": expected_sessions,
                            "pages_processed": [img.page for img in page_images],
                            "pages_per_call": chunk_size,
                            "missing_fields": sorted(missing),
                            "conflicts": fact_conflicts,
                            "debug_dir": str(debug_dir) if debug_dir is not None else None,
                            "attempt_log": attempt_log,
                            "partial_data_pack": merged,
                        }
                        failure_path.write_text(json.dumps(failure_payload, indent=2, sort_keys=True), encoding="utf-8")
                    except Exception:
                        pass

                    chunk_pages = [pages for pages in _chunked([img.page for img in page_images], chunk_size)]
                    retry_kinds = sorted(
                        {
                            a.get("kind")
                            for a in attempt_log
                            if isinstance(a, dict) and isinstance(a.get("kind"), str) and a.get("kind") != "chunk"
                        }
                    )
                    msg_lines = [
                        "Required data could not be extracted from the PDF images.",
                        f"Missing fields: {', '.join(sorted(missing)) if missing else '(none)'}",
                        f"Conflicts: {len(fact_conflicts)}" if fact_conflicts else "Conflicts: (none)",
                        f"Expected sessions: {expected_sessions}",
                        f"Pages processed: {[img.page for img in page_images]}",
                        f"Multimodal passes (pages_per_call={chunk_size}): {chunk_pages}",
                        f"Targeted retries: {', '.join(retry_kinds) if retry_kinds else 'none'}",
                        f"Extractor model: {extractor_model_id}",
                        f"Failure artifact: {failure_path}",
                    ]
                    if debug_dir is not None:
                        msg_lines.append(f"Debug artifacts: {debug_dir}")
                    msg_lines.append("If report assets look incomplete, re-run: POST /api/reports/{report_id}/reextract")
                    raise RuntimeError("\n".join(msg_lines))

                out_path.parent.mkdir(parents=True, exist_ok=True)
                out_path.write_text(json.dumps(merged, indent=2, sort_keys=True), encoding="utf-8")

                # Store as an artifact for traceability/debugging.
                with session_scope() as session:
                    create_artifact(
                        session,
                        run_id=run_id,
                        stage_num=1,
                        stage_name=STAGES[0].name,
                        model_id="_data_pack",
                        kind="data_pack",
                        content_path=out_path,
                        content_type="application/json",
                    )

                return merged
            except Exception as e:
                errors.append(f"{extractor_model_id}: {e}")
                continue

        if strict:
            msg = "All extractor models failed to build a data pack."
            if errors:
                msg = f"{msg}\n\nAttempts:\n- " + "\n- ".join(errors)
            raise RuntimeError(msg)

        return None

    async def _ensure_vision_transcript(
        self,
        *,
        run_id: str,
        report: Report,
        page_images: list[PageImage],
        transcript_model_id: str | None,
        strict: bool,
    ) -> str | None:
        """
        Build a run-level multimodal transcript so later stages (2-6) can access image-only tables/figures.

        This is intentionally separate from the structured data pack:
        - Data pack is strict + normalized for required metrics.
        - Vision transcript is broad + best-effort for everything visible on the pages.
        """
        out_path = _vision_transcript_path(run_id)
        if out_path.exists():
            try:
                existing = out_path.read_text(encoding="utf-8", errors="replace")
                if existing.strip():
                    # Ensure cached transcripts cover all current pages (older runs could have partial coverage).
                    expected_pages = sorted({img.page for img in page_images if isinstance(img.page, int)})
                    found_pages = {
                        int(m.group(1))
                        for m in re.finditer(r"(?m)^##\\s*Page\\s+(\\d+)\\b", existing)
                        if m.group(1).isdigit()
                    }
                    if not expected_pages or found_pages.issuperset(set(expected_pages)):
                        return existing
            except Exception:
                pass

        if not page_images or not transcript_model_id:
            if strict:
                report_dir = _derive_report_dir(report)
                raise RuntimeError(
                    "Strict data availability requested, but no vision transcript can be generated.\n"
                    f"Report: {report.filename} ({report.id})\n"
                    f"Transcript model: {transcript_model_id}\n"
                    f"Page images available: {len(page_images)}\n"
                    f"Expected page images directory: {report_dir / 'pages'}\n"
                    "If report assets look incomplete, re-run: POST /api/reports/{report_id}/reextract"
                )
            return None

        chunk_size = int(os.getenv("QEEG_VISION_TRANSCRIPT_PAGES_PER_CALL", "2") or "2")
        if chunk_size <= 0:
            chunk_size = 2
        # Hard requirement: PDFs >10 pages must be processed in 2+ multimodal passes.
        if len(page_images) > 10 and chunk_size > 10:
            chunk_size = 10

        max_tokens = int(os.getenv("QEEG_VISION_TRANSCRIPT_MAX_TOKENS", "4000") or "4000")
        if max_tokens <= 0:
            max_tokens = 4000

        parts: list[str] = []
        errors: list[str] = []
        chunks = _chunked(page_images, chunk_size)
        for chunk_index, chunk in enumerate(chunks, start=1):
            pages = [img.page for img in chunk]
            pages_str = ", ".join(str(p) for p in pages) if pages else "(unknown)"
            prompt_text = (
                "You are performing LOSSLESS MULTIMODAL TRANSCRIPTION from qEEG report page images.\n\n"
                "Goal: Ensure ALL information embedded in page graphics/tables is available downstream as text.\n\n"
                "Output format: Markdown.\n\n"
                "Rules:\n"
                "- Do NOT interpret or summarize. Do NOT diagnose.\n"
                "- Do NOT invent numbers. Transcribe only what is visible.\n"
                "- For each page, start a section with: \"## Page <n>\".\n"
                "- Under each page, transcribe every table/figure caption and any clearly printed numeric values.\n"
                "- When a page contains a table, reproduce it as a markdown table (keep row/column labels).\n"
                "- Preserve units, target ranges, N/A markings, and any symbols (e.g., *, #) when shown.\n"
                "- If a color key maps sessions to colors, transcribe the key explicitly.\n\n"
                f"Pages in this pass: {pages_str}\n"
                f"Pass {chunk_index} of {len(chunks)}\n"
            )
            try:
                text = await self._call_model_multimodal(
                    model_id=transcript_model_id,
                    prompt_text=prompt_text,
                    images=chunk,
                    temperature=0.0,
                    max_tokens=max_tokens,
                    allow_text_fallback=not strict,
                )
                if isinstance(text, str) and text.strip():
                    parts.append(text.strip())
            except Exception as e:
                errors.append(f"pass {chunk_index} pages {pages_str}: {e}")
                if strict:
                    raise RuntimeError(
                        "Strict data availability requested, but multimodal vision transcript generation failed.\n"
                        f"Model: {transcript_model_id}\n"
                        f"Failed pass: {chunk_index} / {len(chunks)}\n"
                        f"Pages: {pages_str}\n"
                        f"Output path: {out_path}\n"
                        "If report assets look incomplete, re-run: POST /api/reports/{report_id}/reextract"
                    ) from e
                continue

        transcript_body = "\n\n".join(parts).strip()
        if not transcript_body:
            if strict:
                msg = "Strict data availability requested, but vision transcript output was empty."
                if errors:
                    msg = f"{msg}\n\nAttempts:\n- " + "\n- ".join(errors)
                raise RuntimeError(msg)
            return None

        header = (
            "# Multimodal Vision Transcript\n"
            f"- run_id: {run_id}\n"
            f"- report: {report.filename} ({report.id})\n"
            f"- model_id: {transcript_model_id}\n"
            f"- pages_per_call: {chunk_size}\n"
            f"- page_count: {len(page_images)}\n\n"
        )
        transcript = f"{header}{transcript_body}\n"

        try:
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(transcript, encoding="utf-8")
        except Exception:
            pass

        try:
            with session_scope() as session:
                create_artifact(
                    session,
                    run_id=run_id,
                    stage_num=1,
                    stage_name=STAGES[0].name,
                    model_id="_vision_transcript",
                    kind="vision_transcript",
                    content_path=out_path,
                    content_type="text/markdown",
                )
        except Exception:
            pass

        return transcript

    @staticmethod
    def _merge_data_pack_parts(parts: list[dict[str, Any]], meta: dict[str, Any]) -> dict[str, Any]:
        pages_seen: set[int] = set()
        page_inventory: list[dict[str, Any]] = []
        facts: list[dict[str, Any]] = []
        unparsed_required: list[dict[str, Any]] = []

        for part in parts:
            for p in part.get("pages_seen", []) or []:
                if isinstance(p, int):
                    pages_seen.add(p)
            inv = part.get("page_inventory")
            if isinstance(inv, list):
                page_inventory.extend([x for x in inv if isinstance(x, dict)])
            f = part.get("facts")
            if isinstance(f, list):
                facts.extend([x for x in f if isinstance(x, dict)])
            ur = part.get("unparsed_required")
            if isinstance(ur, list):
                unparsed_required.extend([x for x in ur if isinstance(x, dict)])

        deduped = _DataPackMixin._dedupe_facts(facts)

        return {
            **meta,
            "pages_seen": sorted(pages_seen),
            "page_inventory": page_inventory,
            "facts": deduped,
            "unparsed_required": unparsed_required,
        }

    @staticmethod
    def _derive_data_pack_views(data_pack: dict[str, Any], *, expected_sessions: list[int]) -> dict[str, Any]:
        facts = data_pack.get("facts")
        if not isinstance(facts, list):
            return {}

        def fmt_num(val: Any) -> str:
            if isinstance(val, int):
                return str(val)
            if isinstance(val, float):
                # Keep small clinical values readable without over-rounding.
                return f"{val:.2f}".rstrip("0").rstrip(".")
            return ""

        def shown_as_or_na(f: dict[str, Any]) -> str | None:
            shown_as = f.get("shown_as")
            if isinstance(shown_as, str) and shown_as.strip():
                return shown_as.strip()
            return None

        def get_fact(*, fact_type: str, metric: str, session_index: int) -> dict[str, Any] | None:
            for f in facts:
                if not isinstance(f, dict):
                    continue
                if f.get("fact_type") != fact_type:
                    continue
                if f.get("metric") != metric:
                    continue
                if f.get("session_index") != session_index:
                    continue
                return f
            return None

        def short_shown_value(f: dict[str, Any]) -> str | None:
            shown = shown_as_or_na(f)
            if not shown:
                return None
            if shown.strip().upper() == "N/A":
                return "N/A"
            if re.match(r"^[<>≥≤]?\s*-?\d+(?:\.\d+)?\s*(?:ms|µV|uV|Hz|sec|s|ratio)?\s*$", shown):
                return shown
            return None

        # Summary tables (PAGE 1 style).
        perf_specs = [
            ("Physical Reaction Time", "physical_reaction_time"),
            ("Trail Making Test A", "trail_making_test_a"),
            ("Trail Making Test B", "trail_making_test_b"),
        ]
        perf_headers = ["Metric"] + [f"Session {i}" for i in expected_sessions] + ["Target"]
        perf_rows = ["| " + " | ".join(perf_headers) + " |", "|" + "|".join(["---"] * len(perf_headers)) + "|"]
        for label, metric in perf_specs:
            target = ""
            cells: list[str] = [label]
            for sess in expected_sessions:
                f = get_fact(fact_type="performance_metric", metric=metric, session_index=sess)
                if not f:
                    cells.append("MISSING")
                    continue
                if not target and isinstance(f.get("target_range"), str):
                    target = f["target_range"]
                shown = short_shown_value(f)
                if shown == "N/A":
                    cells.append("N/A")
                    continue
                val = f.get("value")
                if val is None:
                    cells.append(shown_as_or_na(f) or "N/A")
                    continue
                unit = f.get("unit") if isinstance(f.get("unit"), str) else ""
                sd = f.get("sd_plus_minus")
                if metric == "physical_reaction_time" and isinstance(sd, (int, float)):
                    cells.append(f"{fmt_num(val)} ±{fmt_num(sd)} {unit}".strip())
                else:
                    cells.append(f"{fmt_num(val)} {unit}".strip())
            cells.append(target)
            perf_rows.append("| " + " | ".join(cells) + " |")
        performance_table = "\n".join(perf_rows)

        evoked_specs = [
            ("Audio P300 Delay", "audio_p300_delay"),
            ("Audio P300 Voltage", "audio_p300_voltage"),
        ]
        evoked_headers = ["Metric"] + [f"Session {i}" for i in expected_sessions] + ["Target"]
        evoked_rows = ["| " + " | ".join(evoked_headers) + " |", "|" + "|".join(["---"] * len(evoked_headers)) + "|"]
        for label, metric in evoked_specs:
            target = ""
            cells = [label]
            for sess in expected_sessions:
                f = get_fact(fact_type="evoked_potential", metric=metric, session_index=sess)
                if not f:
                    cells.append("MISSING")
                    continue
                if not target and isinstance(f.get("target_range"), str):
                    target = f["target_range"]
                shown = short_shown_value(f)
                if shown == "N/A":
                    cells.append("N/A")
                    continue
                val = f.get("value")
                if val is None:
                    cells.append(shown_as_or_na(f) or "N/A")
                    continue
                unit = f.get("unit") if isinstance(f.get("unit"), str) else ""
                cells.append(f"{fmt_num(val)} {unit}".strip())
            cells.append(target)
            evoked_rows.append("| " + " | ".join(cells) + " |")
        evoked_table = "\n".join(evoked_rows)

        state_specs = [
            ("CZ Eyes Closed Theta/Beta (Power)", "cz_theta_beta_ratio_ec"),
            ("F3/F4 Eyes Closed Alpha (Power)", "f3_f4_alpha_ratio_ec"),
        ]
        state_headers = ["Metric"] + [f"Session {i}" for i in expected_sessions] + ["Target"]
        state_rows = ["| " + " | ".join(state_headers) + " |", "|" + "|".join(["---"] * len(state_headers)) + "|"]
        for label, metric in state_specs:
            target = ""
            cells = [label]
            for sess in expected_sessions:
                f = get_fact(fact_type="state_metric", metric=metric, session_index=sess)
                if not f:
                    cells.append("MISSING")
                    continue
                if not target and isinstance(f.get("target_range"), str):
                    target = f["target_range"]
                shown = short_shown_value(f)
                if shown:
                    cells.append(shown)
                    continue
                val = f.get("value")
                cells.append(fmt_num(val) if val is not None else (shown_as_or_na(f) or "N/A"))
            cells.append(target)
            state_rows.append("| " + " | ".join(cells) + " |")
        state_table = "\n".join(state_rows)

        peak_specs = [
            ("Frontal", "frontal_peak_frequency_ec"),
            ("Central-Parietal", "central_parietal_peak_frequency_ec"),
            ("Occipital", "occipital_peak_frequency_ec"),
        ]
        peak_headers = ["Region"] + [f"Session {i}" for i in expected_sessions] + ["Target"]
        peak_rows = ["| " + " | ".join(peak_headers) + " |", "|" + "|".join(["---"] * len(peak_headers)) + "|"]
        for label, metric in peak_specs:
            target = ""
            cells = [label]
            for sess in expected_sessions:
                f = get_fact(fact_type="peak_frequency", metric=metric, session_index=sess)
                if not f:
                    cells.append("MISSING")
                    continue
                if not target and isinstance(f.get("target_range"), str):
                    target = f["target_range"]
                shown = short_shown_value(f)
                if shown == "N/A":
                    cells.append("N/A")
                    continue
                val = f.get("value")
                if val is None:
                    cells.append(shown_as_or_na(f) or "N/A")
                    continue
                unit = f.get("unit") if isinstance(f.get("unit"), str) else ""
                cells.append(f"{fmt_num(val)} {unit}".strip())
            cells.append(target)
            peak_rows.append("| " + " | ".join(cells) + " |")
        peak_table = "\n".join(peak_rows)

        # CP per-site P300 table (C3, CZ, C4, P3, PZ, P4).
        cp_sites = ["C3", "CZ", "C4", "P3", "PZ", "P4"]
        cp_map: dict[str, dict[int, dict[str, Any]]] = {s: {} for s in cp_sites}
        for f in facts:
            if f.get("fact_type") != "p300_cp_site":
                continue
            site = f.get("site")
            sess = f.get("session_index")
            if site not in cp_map or not isinstance(sess, int):
                continue
            cp_map[site][sess] = f

        headers = ["Site"] + [f"Session {i} (µV / ms)" for i in expected_sessions]
        rows = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
        for site in cp_sites:
            cells: list[str] = [site]
            for sess in expected_sessions:
                f = cp_map.get(site, {}).get(sess)
                if not f:
                    cells.append("MISSING")
                    continue
                uv = f.get("uv")
                ms = f.get("ms")
                if uv is None or ms is None:
                    cells.append(shown_as_or_na(f) or "N/A")
                else:
                    cells.append(f"{fmt_num(uv)} / {fmt_num(ms)}")
            rows.append("| " + " | ".join(cells) + " |")
        p300_cp_table = "\n".join(rows)

        # CENTRAL-FRONTAL AVERAGE N100 table.
        n100_by_sess: dict[int, dict[str, Any]] = {}
        for f in facts:
            if f.get("fact_type") != "n100_central_frontal_average":
                continue
            sess = f.get("session_index")
            if not isinstance(sess, int):
                continue
            n100_by_sess[sess] = f

        n100_headers = ["Metric"] + [f"Session {i}" for i in expected_sessions]
        n100_rows = ["| " + " | ".join(n100_headers) + " |", "|" + "|".join(["---"] * len(n100_headers)) + "|"]
        cells = ["Central-frontal N100 (yield, µV, ms)"]
        for sess in expected_sessions:
            f = n100_by_sess.get(sess)
            if not f:
                cells.append("MISSING")
                continue
            shown = shown_as_or_na(f)
            if shown and shown.strip().upper() == "N/A":
                yv = f.get("yield")
                cells.append(f"{fmt_num(yv)} / N/A" if yv is not None else "N/A")
                continue
            yv = f.get("yield")
            uv = f.get("uv")
            ms = f.get("ms")
            if uv is None or ms is None:
                cells.append(shown or "N/A")
            else:
                ytxt = fmt_num(yv) if yv is not None else ""
                # Keep a consistent "yield / µV / ms" shape.
                cells.append(f"{ytxt} / {fmt_num(uv)} / {fmt_num(ms)}".strip(" /"))
        n100_rows.append("| " + " | ".join(cells) + " |")
        n100_table = "\n".join(n100_rows)

        return {
            "summary_performance_table_markdown": performance_table,
            "summary_evoked_table_markdown": evoked_table,
            "summary_state_table_markdown": state_table,
            "peak_frequency_table_markdown": peak_table,
            "p300_cp_table_markdown": p300_cp_table,
            "n100_central_frontal_table_markdown": n100_table,
        }

    @staticmethod
    def _missing_required_fields(data_pack: dict[str, Any], *, expected_sessions: list[int]) -> set[str]:
        facts = data_pack.get("facts")
        if not isinstance(facts, list):
            return {"facts"}

        missing: set[str] = set()

        def has_values_or_na(f: dict[str, Any], *, fields: tuple[str, ...]) -> bool:
            shown_as = f.get("shown_as")
            if isinstance(shown_as, str) and shown_as.strip().upper() == "N/A":
                return True
            return all(f.get(field) is not None for field in fields)

        def has_fact(predicate) -> bool:
            for f in facts:
                if not isinstance(f, dict):
                    continue
                if predicate(f):
                    return True
            return False

        # Performance metrics (must exist for each expected session).
        perf_metrics = [
            ("performance_metric", "physical_reaction_time"),
            ("performance_metric", "trail_making_test_a"),
            ("performance_metric", "trail_making_test_b"),
        ]
        for _, metric in perf_metrics:
            for sess in expected_sessions:
                if not has_fact(
                    lambda f, m=metric, s=sess: f.get("fact_type") == "performance_metric"
                    and f.get("metric") == m
                    and f.get("session_index") == s
                    and has_values_or_na(f, fields=("value",))
                ):
                    missing.add(f"performance:{metric}:session_{sess}")

        # Summary P300 delay/voltage.
        for metric in ("audio_p300_delay", "audio_p300_voltage"):
            for sess in expected_sessions:
                if not has_fact(
                    lambda f, m=metric, s=sess: f.get("fact_type") == "evoked_potential"
                    and f.get("metric") == m
                    and f.get("session_index") == s
                    and has_values_or_na(f, fields=("value",))
                ):
                    missing.add(f"evoked:{metric}:session_{sess}")

        # Background EEG summary metrics.
        for metric in ("cz_theta_beta_ratio_ec", "f3_f4_alpha_ratio_ec"):
            for sess in expected_sessions:
                if not has_fact(
                    lambda f, m=metric, s=sess: f.get("fact_type") == "state_metric"
                    and f.get("metric") == m
                    and f.get("session_index") == s
                    and has_values_or_na(f, fields=("value",))
                ):
                    missing.add(f"state:{metric}:session_{sess}")

        # Peak frequency by region (Frontal, Central-Parietal, Occipital).
        for metric in (
            "frontal_peak_frequency_ec",
            "central_parietal_peak_frequency_ec",
            "occipital_peak_frequency_ec",
        ):
            for sess in expected_sessions:
                if not has_fact(
                    lambda f, m=metric, s=sess: f.get("fact_type") == "peak_frequency"
                    and f.get("metric") == m
                    and f.get("session_index") == s
                    and has_values_or_na(f, fields=("value",))
                ):
                    missing.add(f"peak_frequency:{metric}:session_{sess}")

        # CP per-site P300 table.
        cp_sites = ["C3", "CZ", "C4", "P3", "PZ", "P4"]
        for site in cp_sites:
            for sess in expected_sessions:
                if not has_fact(
                    lambda f, st=site, s=sess: f.get("fact_type") == "p300_cp_site"
                    and f.get("site") == st
                    and f.get("session_index") == s
                    and has_values_or_na(f, fields=("uv", "ms"))
                ):
                    missing.add(f"p300_cp_site:{site}:session_{sess}")

        # CENTRAL-FRONTAL AVERAGE N100.
        for sess in expected_sessions:
            if not has_fact(
                lambda f, s=sess: f.get("fact_type") == "n100_central_frontal_average"
                and f.get("session_index") == s
                and has_values_or_na(f, fields=("uv", "ms"))
            ):
                missing.add(f"n100:central_frontal_average:session_{sess}")

        return missing

    async def _targeted_retry_missing(
        self,
        *,
        extractor_model_id: str,
        page_images: list[PageImage],
        merged: dict[str, Any],
        expected_sessions: list[int],
        missing: set[str],
        candidate_pages: list[int] | None = None,
        debug_dir: Path | None = None,
        attempt_log: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        # Focused retry for P300 Rare Comparison page(s).
        need_cp = any(k.startswith("p300_cp_site:") for k in missing)
        need_n100 = any(k.startswith("n100:") for k in missing)
        if not (need_cp or need_n100):
            return merged

        pages_to_try: list[int] = []
        if candidate_pages:
            pages_to_try = [p for p in candidate_pages if isinstance(p, int)]

        # Fallback: use model page inventory if present, else default to page 2.
        if not pages_to_try:
            pages_to_try = [2]
            inv = merged.get("page_inventory")
            if isinstance(inv, list):
                pages = []
                for item in inv:
                    if not isinstance(item, dict):
                        continue
                    title = item.get("title")
                    page = item.get("page")
                    if isinstance(title, str) and "p300" in title.lower() and isinstance(page, int):
                        pages.append(page)
                if pages:
                    pages_to_try = sorted(set(pages))

        img_by_page = {img.page: img for img in page_images}
        base_pages = [img_by_page[p] for p in pages_to_try if p in img_by_page]
        if not base_pages:
            return merged

        retry_images: list[PageImage] = []
        for page_img in base_pages:
            crops = _try_build_p300_cp_site_crops(page_img)
            if crops:
                retry_images = crops
                break
        if not retry_images:
            retry_images = base_pages

        focus = (
            "RETRY: Extract central-parietal per-site P300 values (C3, CZ, C4, P3, PZ, P4) for every session shown.\n"
            "- If images are cropped panels (labels like C3_panel), each panel corresponds to ONE site.\n"
            "- Use the legend crop (if provided) to map Session 1/2/3 to colors.\n"
            "- For each site+session, extract yield (#), uv (µV), and ms.\n"
            "- If a value is shown as N/A, set uv/ms to null and shown_as=\"N/A\".\n"
            "Also extract CENTRAL-FRONTAL AVERAGE N100 (yield, uv, ms) if present (look for a crop labeled "
            "central_frontal_avg if provided)."
        )
        prompt_text = _data_pack_prompt(pages=[img.page for img in retry_images], focus=focus)

        raw = await self._call_model_multimodal(
            model_id=extractor_model_id,
            prompt_text=prompt_text,
            images=retry_images,
            temperature=0.0,
            max_tokens=2500,
            allow_text_fallback=False,
        )

        if debug_dir is not None:
            try:
                safe_model = "".join(c if c.isalnum() or c in {"-", "_", "."} else "_" for c in extractor_model_id) or "model"
                model_dir = debug_dir / safe_model
                model_dir.mkdir(parents=True, exist_ok=True)
                pages_key = "-".join(str(p) for p in sorted({img.page for img in retry_images}))
                stem = f"retry-p300-{pages_key}"
                (model_dir / f"{stem}.prompt.txt").write_text(prompt_text, encoding="utf-8")
                (model_dir / f"{stem}.raw.txt").write_text(raw, encoding="utf-8")
                image_paths = _save_debug_images(model_dir=model_dir, stem=stem, images=retry_images)
                if attempt_log is not None:
                    attempt_log.append(
                        {
                            "kind": "retry_p300_cp_n100",
                            "model_id": extractor_model_id,
                            "pages": sorted({img.page for img in retry_images}),
                            "labels_sample": [img.label for img in retry_images[:12]],
                            "raw_path": str(model_dir / f"{stem}.raw.txt"),
                            "image_paths": image_paths,
                        }
                    )
            except Exception:
                pass
        part = _json_loads_loose(raw)
        if not isinstance(part, dict) or part.get("schema_version") != DATA_PACK_SCHEMA_VERSION:
            return merged

        if debug_dir is not None:
            try:
                safe_model = "".join(c if c.isalnum() or c in {"-", "_", "."} else "_" for c in extractor_model_id) or "model"
                model_dir = debug_dir / safe_model
                pages_key = "-".join(str(p) for p in sorted({img.page for img in retry_images}))
                stem = f"retry-p300-{pages_key}"
                (model_dir / f"{stem}.parsed.json").write_text(
                    json.dumps(part, indent=2, sort_keys=True), encoding="utf-8"
                )
            except Exception:
                pass

        merged2 = self._merge_data_pack_parts([merged, part], meta={k: v for k, v in merged.items() if k != "derived"})
        merged2["derived"] = self._derive_data_pack_views(merged2, expected_sessions=expected_sessions)
        return merged2

    async def _targeted_retry_summary_missing(
        self,
        *,
        extractor_model_id: str,
        page_images: list[PageImage],
        merged: dict[str, Any],
        expected_sessions: list[int],
        missing: set[str],
        debug_dir: Path | None = None,
        attempt_log: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        need_summary = any(
            k.startswith("performance:")
            or k.startswith("evoked:")
            or k.startswith("state:")
            or k.startswith("peak_frequency:")
            for k in missing
        )
        if not need_summary:
            return merged

        img_by_page = {img.page: img for img in page_images}
        page1 = img_by_page.get(1)
        if page1 is None:
            return merged

        retry_images = _try_build_summary_table_crops(page1) or [page1]
        focus = (
            "RETRY: Extract PAGE-1 SUMMARY metrics for every session shown:\n"
            "- Performance: Physical Reaction Time (ms), Trail Making Test A (sec), Trail Making Test B (sec)\n"
            "- Evoked: Audio P300 Delay (ms), Audio P300 Voltage (µV)\n"
            "- State: CZ Eyes Closed Theta/Beta (Power), F3/F4 Eyes Closed Alpha (Power)\n"
            "- Peak Frequency: Frontal, Central-Parietal, Occipital (Hz)\n"
            "Use the canonical identifiers and include target ranges when shown."
        )
        prompt_text = _data_pack_prompt(pages=[img.page for img in retry_images], focus=focus)

        raw = await self._call_model_multimodal(
            model_id=extractor_model_id,
            prompt_text=prompt_text,
            images=retry_images,
            temperature=0.0,
            max_tokens=2500,
            allow_text_fallback=False,
        )

        if debug_dir is not None:
            try:
                safe_model = "".join(c if c.isalnum() or c in {"-", "_", "."} else "_" for c in extractor_model_id) or "model"
                model_dir = debug_dir / safe_model
                model_dir.mkdir(parents=True, exist_ok=True)
                stem = "retry-summary-page-1"
                (model_dir / f"{stem}.prompt.txt").write_text(prompt_text, encoding="utf-8")
                (model_dir / f"{stem}.raw.txt").write_text(raw, encoding="utf-8")
                image_paths = _save_debug_images(model_dir=model_dir, stem=stem, images=retry_images)
                if attempt_log is not None:
                    attempt_log.append(
                        {
                            "kind": "retry_summary_page_1",
                            "model_id": extractor_model_id,
                            "pages": sorted({img.page for img in retry_images}),
                            "labels_sample": [img.label for img in retry_images[:12]],
                            "raw_path": str(model_dir / f"{stem}.raw.txt"),
                            "image_paths": image_paths,
                        }
                    )
            except Exception:
                pass

        part = _json_loads_loose(raw)
        if not isinstance(part, dict) or part.get("schema_version") != DATA_PACK_SCHEMA_VERSION:
            return merged

        if debug_dir is not None:
            try:
                safe_model = "".join(c if c.isalnum() or c in {"-", "_", "."} else "_" for c in extractor_model_id) or "model"
                model_dir = debug_dir / safe_model
                (model_dir / "retry-summary-page-1.parsed.json").write_text(
                    json.dumps(part, indent=2, sort_keys=True), encoding="utf-8"
                )
            except Exception:
                pass

        merged2 = self._merge_data_pack_parts([merged, part], meta={k: v for k, v in merged.items() if k != "derived"})
        merged2["derived"] = self._derive_data_pack_views(merged2, expected_sessions=expected_sessions)
        return merged2


