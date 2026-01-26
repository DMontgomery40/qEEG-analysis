from __future__ import annotations

import asyncio
import base64
import json
import os
from pathlib import Path
from typing import Any, Awaitable, Callable

from ...config import DISCOVERED_MODEL_IDS, is_vision_capable
from ...reports import extract_pdf_full
from ...storage import Report, get_report, get_run, set_run_label_map
from ...storage import session_scope
from ..constants import STAGES
from ..db_utils import _aggregate_required_changes, _stage_artifacts, _validate_stage5
from ..json_utils import _json_loads_loose, _strip_to_json
from ..paths import _data_pack_path, _stable_label_map, _vision_transcript_path
from ..prompts import _load_prompt, _workflow_context_block
from ..report_assets import _derive_report_dir, _load_best_report_text, _load_page_images
from ..report_text import _page_count_from_markers
from ..types import PageImage
from ..utils import _chunked, _truthy_env


class _StagesMixin:
    @staticmethod
    def _select_discovered_model_id(preferred: str) -> str | None:
        pref = (preferred or "").strip()
        if not pref:
            return None

        # Exact match first.
        if pref in DISCOVERED_MODEL_IDS:
            return pref

        # Case-insensitive exact match.
        pref_lower = pref.lower()
        for mid in DISCOVERED_MODEL_IDS:
            if mid.lower() == pref_lower:
                return mid

        # Substring match (prefer non-preview variants if both exist).
        matches = [mid for mid in DISCOVERED_MODEL_IDS if pref_lower in mid.lower()]
        if not matches:
            return None

        def rank(mid: str) -> tuple[int, int, str]:
            lower = mid.lower()
            preview_penalty = 1 if "preview" in lower else 0
            return (preview_penalty, len(mid), mid)

        return sorted(matches, key=rank)[0]

    async def _stage1(
        self,
        run_id: str,
        council_model_ids: list[str],
        report: Report,
        emit: Callable[[dict[str, Any]], Awaitable[None]],
    ) -> None:
        stage = STAGES[0]
        prompt = _load_prompt("stage1_analysis.md")
        report_dir = _derive_report_dir(report)
        report_text = _load_best_report_text(report, report_dir)

        # Prefer a single "checker" vision model for multimodal extraction + verification.
        # This is intentionally independent from the selected council model set, so the council can use
        # text-only models while still getting page-grounded structured data + transcript.
        vision_checker_pref = os.getenv("QEEG_VISION_CHECKER_MODEL", "gemini-3-flash")
        vision_checker_id = self._select_discovered_model_id(vision_checker_pref)
        if vision_checker_id and not is_vision_capable(vision_checker_id):
            vision_checker_id = None

        # Load images from the report folder (preferred), then fallback lookup.
        page_images = _load_page_images(report, report_dir)

        needs_images = bool(vision_checker_id) or any(is_vision_capable(m) for m in council_model_ids)
        expected_page_count = _page_count_from_markers(report_text)
        pages_present = {img.page for img in page_images}
        missing_pages: list[int] = []
        if expected_page_count:
            missing_pages = [p for p in range(1, expected_page_count + 1) if p not in pages_present]

        # If a vision-capable model is selected but images are missing/incomplete, generate on the fly.
        if needs_images and Path(report.stored_path).suffix.lower() == ".pdf" and (not page_images or missing_pages):
            try:
                full = extract_pdf_full(Path(report.stored_path))
                enhanced_text = full.enhanced_text
                if enhanced_text and enhanced_text.strip():
                    report_text = enhanced_text
                    try:
                        (report_dir / "extracted_enhanced.txt").write_text(enhanced_text, encoding="utf-8")
                        # Keep extracted.txt aligned so the UI preview and any verification tooling never
                        # shows only a single OCR engine.
                        (report_dir / "extracted.txt").write_text(enhanced_text, encoding="utf-8")
                    except Exception:
                        pass

                page_images = []
                for img in full.page_images:
                    if not isinstance(img, dict):
                        continue
                    page = img.get("page")
                    b64 = img.get("base64_png")
                    if isinstance(page, int) and isinstance(b64, str):
                        page_images.append(PageImage(page=page, base64_png=b64))

                # Best-effort persist generated images + per-page sources/metadata for later stages/debugging.
                if page_images:
                    try:
                        pages_dir = report_dir / "pages"
                        pages_dir.mkdir(parents=True, exist_ok=True)
                        for img in page_images:
                            out = pages_dir / f"page-{img.page}.png"
                            out.write_bytes(base64.b64decode(img.base64_png))
                    except Exception:
                        pass

                try:
                    sources_dir = report_dir / "sources"
                    sources_dir.mkdir(parents=True, exist_ok=True)
                    for p in full.per_page_sources:
                        page_num = p.get("page")
                        if not isinstance(page_num, int):
                            continue
                        (sources_dir / f"page-{page_num}.pypdf.txt").write_text(p.get("pypdf_text", ""), encoding="utf-8")
                        (sources_dir / f"page-{page_num}.pymupdf.txt").write_text(p.get("pymupdf_text", ""), encoding="utf-8")
                        (sources_dir / f"page-{page_num}.apple_vision.txt").write_text(p.get("vision_ocr_text", ""), encoding="utf-8")
                        (sources_dir / f"page-{page_num}.tesseract.txt").write_text(p.get("tesseract_ocr_text", ""), encoding="utf-8")

                    meta = dict(full.metadata)
                    meta.update(
                        {
                            "has_enhanced_ocr": True,
                            "has_page_images": True,
                            "page_images_written": len(page_images),
                            "sources_dir": "sources",
                        }
                    )
                    (report_dir / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
                except Exception:
                    pass
            except Exception:
                page_images = []

        is_pdf = Path(report.stored_path).suffix.lower() == ".pdf"
        strict_data = _truthy_env("QEEG_STRICT_DATA_AVAILABILITY", True)
        # Non-PDF uploads can't be validated via page images.
        if not is_pdf:
            strict_data = False
        # Prevent accidental non-strict PDF runs unless explicitly allowed.
        if is_pdf and not strict_data and not _truthy_env("QEEG_ALLOW_NONSTRICT_DATA_AVAILABILITY", False):
            strict_data = True
        # Tests/mocks: don't hard-fail on missing multimodal extraction.
        if all(mid.startswith("mock-") for mid in council_model_ids):
            strict_data = False

        # In strict mode, enforce multi-source extraction coverage (PDF-native + Apple Vision OCR + Tesseract OCR).
        enforce_all_sources = _truthy_env("QEEG_ENFORCE_ALL_SOURCES", True)
        if strict_data and is_pdf and enforce_all_sources:
            meta_path = report_dir / "metadata.json"
            meta: dict[str, Any] | None = None
            if meta_path.exists():
                try:
                    loaded = json.loads(meta_path.read_text(encoding="utf-8"))
                    if isinstance(loaded, dict):
                        meta = loaded
                except Exception:
                    meta = None

            engines: dict[str, Any] = {}
            if isinstance(meta, dict) and meta.get("schema_version") == 2 and isinstance(meta.get("engines"), dict):
                engines = meta["engines"]

            if not engines:
                # Metadata missing/outdated: regenerate report assets in-place (best effort) so strict runs always have
                # a full audit trail (sources/ + metadata.json).
                try:
                    full = extract_pdf_full(Path(report.stored_path))
                    enhanced_text = full.enhanced_text
                    if enhanced_text and enhanced_text.strip():
                        report_text = enhanced_text
                        (report_dir / "extracted_enhanced.txt").write_text(enhanced_text, encoding="utf-8")
                        (report_dir / "extracted.txt").write_text(enhanced_text, encoding="utf-8")

                    page_images = []
                    for img in full.page_images:
                        page = img.get("page") if isinstance(img, dict) else None
                        b64 = img.get("base64_png") if isinstance(img, dict) else None
                        if isinstance(page, int) and isinstance(b64, str):
                            page_images.append(PageImage(page=page, base64_png=b64))

                    pages_dir = report_dir / "pages"
                    pages_dir.mkdir(parents=True, exist_ok=True)
                    for img in page_images:
                        (pages_dir / f"page-{img.page}.png").write_bytes(base64.b64decode(img.base64_png))

                    sources_dir = report_dir / "sources"
                    sources_dir.mkdir(parents=True, exist_ok=True)
                    for p in full.per_page_sources:
                        page_num = p.get("page")
                        if not isinstance(page_num, int):
                            continue
                        (sources_dir / f"page-{page_num}.pypdf.txt").write_text(p.get("pypdf_text", ""), encoding="utf-8")
                        (sources_dir / f"page-{page_num}.pymupdf.txt").write_text(p.get("pymupdf_text", ""), encoding="utf-8")
                        (sources_dir / f"page-{page_num}.apple_vision.txt").write_text(p.get("vision_ocr_text", ""), encoding="utf-8")
                        (sources_dir / f"page-{page_num}.tesseract.txt").write_text(p.get("tesseract_ocr_text", ""), encoding="utf-8")

                    meta2 = dict(full.metadata)
                    meta2.update(
                        {
                            "has_enhanced_ocr": True,
                            "has_page_images": True,
                            "page_images_written": len(page_images),
                            "sources_dir": "sources",
                        }
                    )
                    meta_path.write_text(json.dumps(meta2, indent=2), encoding="utf-8")
                    engines = meta2.get("engines") if isinstance(meta2.get("engines"), dict) else {}
                except Exception:
                    engines = engines or {}

            required = ["pypdf", "pymupdf", "apple_vision", "tesseract"]
            missing_engines = [k for k in required if not engines.get(k)]
            if missing_engines:
                raise RuntimeError(
                    "Strict data availability requested, but required extraction sources are unavailable.\n"
                    f"Missing sources: {', '.join(missing_engines)}\n"
                    f"Report: {report.filename} ({report.id})\n"
                    f"Metadata: {meta_path}\n"
                    "Fix: ensure Apple Vision OCR + Tesseract are available, then re-run: POST /api/reports/{report_id}/reextract"
                )

        extractor_models = [m for m in council_model_ids if is_vision_capable(m)]
        if vision_checker_id:
            extractor_models = [vision_checker_id] + [m for m in extractor_models if m != vision_checker_id]
        if strict_data and not extractor_models:
            raise RuntimeError(
                "Strict data availability requested, but no vision-capable models were selected.\n"
                "Stage 1 requires at least one vision-capable model to process ALL PDF pages.\n"
                "Select a vision-capable model in the run's council_model_ids OR ensure the vision checker model "
                f"({vision_checker_pref}) is available in /v1/models.\n"
                "See /api/models."
            )

        data_pack = await self._ensure_data_pack(
            run_id=run_id,
            report=report,
            report_text=report_text,
            page_images=page_images,
            candidate_extractor_model_ids=extractor_models,
            strict=strict_data,
        )

        transcript_model_id: str | None = None
        if isinstance(data_pack, dict) and isinstance(data_pack.get("extraction_model_id"), str):
            transcript_model_id = data_pack["extraction_model_id"]
        elif extractor_models:
            transcript_model_id = extractor_models[0]

        vision_transcript_text = await self._ensure_vision_transcript(
            run_id=run_id,
            report=report,
            page_images=page_images,
            transcript_model_id=transcript_model_id,
            strict=strict_data,
        )

        data_pack_block = ""
        if data_pack:
            derived_tables: list[str] = []
            derived = data_pack.get("derived")
            if isinstance(derived, dict):
                for key in (
                    "summary_performance_table_markdown",
                    "summary_evoked_table_markdown",
                    "summary_state_table_markdown",
                    "peak_frequency_table_markdown",
                    "p300_cp_table_markdown",
                    "n100_central_frontal_table_markdown",
                ):
                    val = derived.get(key)
                    if isinstance(val, str) and val.strip():
                        derived_tables.append(val.strip())
            dp_json = json.dumps(data_pack, indent=2, sort_keys=True)
            data_pack_block = (
                "STRUCTURED DATA PACK (authoritative transcription from ALL PDF pages, including graphics):\n\n"
                + ("\n\n".join(derived_tables) + "\n\n" if derived_tables else "")
                + "```json\n"
                + dp_json
                + "\n```\n\n"
            )

        vision_transcript_block = ""
        if isinstance(vision_transcript_text, str) and vision_transcript_text.strip():
            vision_transcript_block = (
                "MULTIMODAL VISION TRANSCRIPT (page-grounded transcription from ALL PDF page images):\n\n"
                f"{vision_transcript_text.strip()}\n\n---\n\n"
            )

        workflow_context = _workflow_context_block(stage_num=stage.num, stage_name=stage.name)

        base_prompt_text = (
            f"{prompt}\n\n---\n\n"
            f"{workflow_context}\n\n---\n\n"
            f"{data_pack_block}"
            f"{vision_transcript_block}"
            "FULL qEEG REPORT OCR TEXT (all pages; may include OCR artifacts):\n\n"
            f"{report_text}\n"
        )

        await emit({"run_id": run_id, "stage_num": stage.num, "stage_name": stage.name, "status": "start"})

        async def one(model_id: str) -> tuple[str, str] | None:
            try:
                # Multi-pass multimodal ingestion for vision models: build page-grounded notes in chunks, then write
                # the final long-form report using the notes + full OCR + data pack.
                notes_text = ""
                per_model_notes = _truthy_env("QEEG_STAGE1_PER_MODEL_VISION_NOTES", False)
                if is_vision_capable(model_id) and page_images and (per_model_notes or not vision_transcript_block):
                    chunk_size = int(os.getenv("QEEG_VISION_PAGES_PER_CALL", "8") or "8")
                    if chunk_size <= 0:
                        chunk_size = 8
                    # Hard requirement: PDFs >10 pages must be ingested in 2+ multimodal passes.
                    if len(page_images) > 10 and chunk_size > 10:
                        chunk_size = 10
                    notes_parts: list[str] = []
                    for chunk in _chunked(page_images, chunk_size):
                        pages = [img.page for img in chunk]
                        ingest_prompt = (
                            "Stage 1 multimodal ingestion pass (do NOT write the final report yet).\n"
                            f"Pages in this pass: {', '.join(str(p) for p in pages)}\n\n"
                            "Task:\n"
                            "- For each provided page image, produce a page-by-page markdown transcript with headings "
                            "\"## Page <n>\".\n"
                            "- Enumerate every table/figure/metric visible on that page and transcribe any clearly "
                            "printed numeric values that are likely clinically relevant.\n"
                            "- Do not interpret or diagnose. Do not invent numbers.\n"
                        )
                        notes = await self._call_model_multimodal(
                            model_id=model_id,
                            prompt_text=ingest_prompt,
                            images=chunk,
                            temperature=0.0,
                            max_tokens=2500,
                            allow_text_fallback=not strict_data,
                        )
                        notes_parts.append(notes)
                    notes_text = "\n\n".join(notes_parts).strip()

                final_prompt = base_prompt_text
                if notes_text:
                    final_prompt = (
                        f"{final_prompt}\n\n---\n\n"
                        "MULTIMODAL INGESTION NOTES (generated from ALL PDF pages in multiple passes):\n\n"
                        f"{notes_text}\n"
                    )
                text = await self._call_model_chat(
                    model_id=model_id,
                    prompt_text=final_prompt,
                    temperature=0.2,
                    max_tokens=8000,
                )
                return model_id, text
            except Exception:
                return None

        results = await asyncio.gather(*(one(m) for m in council_model_ids))
        successes = [r for r in results if r is not None]

        for model_id, text in successes:
            await self._write_artifact(run_id=run_id, stage=stage, model_id=model_id, text=text)

        await emit(
            {
                "run_id": run_id,
                "stage_num": stage.num,
                "stage_name": stage.name,
                "status": "complete",
                "success_count": len(successes),
                "requested_count": len(council_model_ids),
            }
        )

    async def _stage2(
        self,
        run_id: str,
        council_model_ids: list[str],
        emit: Callable[[dict[str, Any]], Awaitable[None]],
    ) -> None:
        stage = STAGES[1]
        prompt = _load_prompt("stage2_peer_review.md")

        with session_scope() as session:
            artifacts = _stage_artifacts(session, run_id, 1, kind="analysis")
            run = get_run(session, run_id)
            report = get_report(session, run.report_id) if run else None

        report_text = ""
        if report and report.extracted_text_path:
            report_dir = _derive_report_dir(report)
            report_text = _load_best_report_text(report, report_dir)

        data_pack_text = ""
        dp_path = _data_pack_path(run_id)
        if dp_path.exists():
            try:
                data_pack_text = dp_path.read_text(encoding="utf-8", errors="replace")
            except Exception:
                data_pack_text = ""

        vision_transcript_text = ""
        vt_path = _vision_transcript_path(run_id)
        if vt_path.exists():
            try:
                vision_transcript_text = vt_path.read_text(encoding="utf-8", errors="replace")
            except Exception:
                vision_transcript_text = ""

        workflow_context = _workflow_context_block(stage_num=stage.num, stage_name=stage.name)
        data_pack_block = ""
        if data_pack_text.strip():
            data_pack_block = (
                "STRUCTURED DATA PACK (authoritative transcription from ALL PDF pages, including graphics):\n\n"
                f"```json\n{data_pack_text.strip()}\n```\n\n---\n\n"
            )

        vision_transcript_block = ""
        if vision_transcript_text.strip():
            vision_transcript_block = (
                "MULTIMODAL VISION TRANSCRIPT (page-grounded transcription from ALL PDF page images):\n\n"
                f"{vision_transcript_text.strip()}\n\n---\n\n"
            )

        analyses_by_model: dict[str, str] = {}
        for a in artifacts:
            analyses_by_model[a.model_id] = Path(a.content_path).read_text(encoding="utf-8", errors="replace")

        available_models = [m for m in council_model_ids if m in analyses_by_model]
        if len(available_models) < 2:
            await emit(
                {
                    "run_id": run_id,
                    "stage_num": stage.num,
                    "stage_name": stage.name,
                    "status": "complete",
                    "skipped": True,
                    "reason": "Not enough Stage 1 analyses for peer review",
                }
            )
            return

        label_map = _stable_label_map(run_id, available_models)
        with session_scope() as session:
            set_run_label_map(session, run_id, label_map)

        analysis_blocks: list[str] = []
        for label, model_id in label_map.items():
            analysis_blocks.append(f"Analysis {label}:\n{analyses_by_model[model_id]}".strip())
        all_analyses_text = "\n\n".join(analysis_blocks)

        await emit({"run_id": run_id, "stage_num": stage.num, "stage_name": stage.name, "status": "start"})

        async def one(reviewer_model_id: str) -> tuple[str, str] | None:
            # Reviewer sees all analyses except its own, still labeled A/B/C...
            reviewer_label = next((lbl for lbl, mid in label_map.items() if mid == reviewer_model_id), None)
            filtered: list[str] = []
            for label, mid in label_map.items():
                if mid == reviewer_model_id:
                    continue
                filtered.append(f"Analysis {label}:\n{analyses_by_model[mid]}".strip())
            if not filtered:
                return None
            filtered_text = "\n\n".join(filtered)
            prompt_text = (
                f"{prompt}\n\n---\n\n"
                f"{workflow_context}\n\n---\n\n"
                f"{data_pack_block}"
                f"{vision_transcript_block}"
                f"ORIGINAL qEEG REPORT (for verification - cross-reference all claims against this):\n\n{report_text}\n\n---\n\n"
                f"Reviewer Model ID: {reviewer_model_id}\n"
                f"Your own analysis label (do not review yourself): {reviewer_label}\n\n"
                f"ANALYSES TO REVIEW:\n\n{filtered_text}\n"
            )
            try:
                text = await self._call_model_chat(
                    model_id=reviewer_model_id,
                    prompt_text=prompt_text,
                    temperature=0.1,
                    max_tokens=4000,
                )
                _json_loads_loose(text)  # sanity
                return reviewer_model_id, _strip_to_json(text)
            except Exception:
                return None

        results = await asyncio.gather(*(one(m) for m in available_models))
        successes = [r for r in results if r is not None]

        for model_id, text in successes:
            await self._write_artifact(run_id=run_id, stage=stage, model_id=model_id, text=text)

        await emit(
            {
                "run_id": run_id,
                "stage_num": stage.num,
                "stage_name": stage.name,
                "status": "complete",
                "success_count": len(successes),
                "requested_count": len(available_models),
                "label_map": label_map,
            }
        )

    async def _stage3(
        self,
        run_id: str,
        council_model_ids: list[str],
        emit: Callable[[dict[str, Any]], Awaitable[None]],
    ) -> None:
        stage = STAGES[2]
        prompt = _load_prompt("stage3_revision.md")

        with session_scope() as session:
            s1 = _stage_artifacts(session, run_id, 1, kind="analysis")
            s2 = _stage_artifacts(session, run_id, 2, kind="peer_review")
            run = get_run(session, run_id)
            label_map = json.loads(run.label_map_json or "{}") if run is not None else {}
            report = get_report(session, run.report_id) if run else None

        report_text = ""
        if report and report.extracted_text_path:
            report_dir = _derive_report_dir(report)
            report_text = _load_best_report_text(report, report_dir)

        data_pack_text = ""
        dp_path = _data_pack_path(run_id)
        if dp_path.exists():
            try:
                data_pack_text = dp_path.read_text(encoding="utf-8", errors="replace")
            except Exception:
                data_pack_text = ""

        vision_transcript_text = ""
        vt_path = _vision_transcript_path(run_id)
        if vt_path.exists():
            try:
                vision_transcript_text = vt_path.read_text(encoding="utf-8", errors="replace")
            except Exception:
                vision_transcript_text = ""

        workflow_context = _workflow_context_block(stage_num=stage.num, stage_name=stage.name)
        data_pack_block = ""
        if data_pack_text.strip():
            data_pack_block = (
                "STRUCTURED DATA PACK (authoritative transcription from ALL PDF pages, including graphics):\n\n"
                f"```json\n{data_pack_text.strip()}\n```\n\n---\n\n"
            )

        vision_transcript_block = ""
        if vision_transcript_text.strip():
            vision_transcript_block = (
                "MULTIMODAL VISION TRANSCRIPT (page-grounded transcription from ALL PDF page images):\n\n"
                f"{vision_transcript_text.strip()}\n\n---\n\n"
            )

        analyses_by_model = {
            a.model_id: Path(a.content_path).read_text(encoding="utf-8", errors="replace") for a in s1
        }
        peer_reviews = [
            (a.model_id, Path(a.content_path).read_text(encoding="utf-8", errors="replace")) for a in s2
        ]

        available_models = [m for m in council_model_ids if m in analyses_by_model]
        if not available_models:
            raise RuntimeError("No Stage 1 analyses available for revision")

        await emit({"run_id": run_id, "stage_num": stage.num, "stage_name": stage.name, "status": "start"})

        async def one(model_id: str) -> tuple[str, str] | None:
            analysis = analyses_by_model.get(model_id)
            if not analysis:
                return None
            my_label = next((lbl for lbl, mid in label_map.items() if mid == model_id), None)
            pr_text = "\n\n".join([f"Peer review by {mid}:\n{txt}" for mid, txt in peer_reviews]).strip()
            prompt_text = (
                f"{prompt}\n\n---\n\n"
                f"{workflow_context}\n\n---\n\n"
                f"{data_pack_block}"
                f"{vision_transcript_block}"
                f"ORIGINAL qEEG REPORT (for fact-checking your revision):\n\n{report_text}\n\n---\n\n"
                f"Your Model ID: {model_id}\n"
                f"Your analysis label (if present): {my_label}\n\n"
                f"Your original analysis:\n\n{analysis}\n\n"
                f"Peer review JSON artifacts:\n\n{pr_text}\n"
            )
            try:
                text = await self._call_model_chat(
                    model_id=model_id, prompt_text=prompt_text, temperature=0.2, max_tokens=8000
                )
                return model_id, text
            except Exception:
                return None

        results = await asyncio.gather(*(one(m) for m in available_models))
        successes = [r for r in results if r is not None]
        if not successes:
            raise RuntimeError("All models failed during Stage 3 revision")

        for model_id, text in successes:
            await self._write_artifact(run_id=run_id, stage=stage, model_id=model_id, text=text)

        await emit(
            {
                "run_id": run_id,
                "stage_num": stage.num,
                "stage_name": stage.name,
                "status": "complete",
                "success_count": len(successes),
                "requested_count": len(available_models),
            }
        )

    async def _stage4(
        self,
        run_id: str,
        emit: Callable[[dict[str, Any]], Awaitable[None]],
    ) -> None:
        stage = STAGES[3]
        prompt = _load_prompt("stage4_consolidation.md")
        required_headings = [
            "# Dataset and Sessions",
            "# Key Empirical Findings",
            "# Performance Assessments",
            "# Auditory ERP: P300 and N100",
            "# Background EEG Metrics",
            "# Speculative Commentary and Interpretive Hypotheses",
            "# Measurement Recommendations",
            "# Uncertainties and Limits",
        ]
        end_sentinel = "<!-- END CONSOLIDATED REPORT -->"

        def has_heading(text: str, heading: str) -> bool:
            import re

            return bool(re.search(rf"(?m)^{re.escape(heading)}\\s*$", text or ""))

        def is_complete(text: str) -> bool:
            if end_sentinel not in (text or ""):
                return False
            return all(has_heading(text, h) for h in required_headings)

        def last_heading_present(text: str) -> tuple[str, int] | tuple[None, None]:
            import re

            positions: list[tuple[str, int]] = []
            for h in required_headings:
                m = re.search(rf"(?m)^{re.escape(h)}\\s*$", text or "")
                if m:
                    positions.append((h, m.start()))
            if not positions:
                return (None, None)
            return max(positions, key=lambda x: x[1])

        with session_scope() as session:
            run = get_run(session, run_id)
            if run is None:
                raise RuntimeError("Run not found")
            consolidator = run.consolidator_model_id
            revisions = _stage_artifacts(session, run_id, 3, kind="revision")
            report = get_report(session, run.report_id) if run else None

        report_text = ""
        if report and report.extracted_text_path:
            report_dir = _derive_report_dir(report)
            report_text = _load_best_report_text(report, report_dir)

        data_pack_text = ""
        dp_path = _data_pack_path(run_id)
        if dp_path.exists():
            try:
                data_pack_text = dp_path.read_text(encoding="utf-8", errors="replace")
            except Exception:
                data_pack_text = ""

        vision_transcript_text = ""
        vt_path = _vision_transcript_path(run_id)
        if vt_path.exists():
            try:
                vision_transcript_text = vt_path.read_text(encoding="utf-8", errors="replace")
            except Exception:
                vision_transcript_text = ""

        workflow_context = _workflow_context_block(stage_num=stage.num, stage_name=stage.name)
        data_pack_block = ""
        if data_pack_text.strip():
            data_pack_block = (
                "STRUCTURED DATA PACK (authoritative transcription from ALL PDF pages, including graphics):\n\n"
                f"```json\n{data_pack_text.strip()}\n```\n\n---\n\n"
            )

        vision_transcript_block = ""
        if vision_transcript_text.strip():
            vision_transcript_block = (
                "MULTIMODAL VISION TRANSCRIPT (page-grounded transcription from ALL PDF page images):\n\n"
                f"{vision_transcript_text.strip()}\n\n---\n\n"
            )

        if not revisions:
            raise RuntimeError("Consolidation requires at least one revision artifact")

        revision_text = "\n\n".join(
            [
                f"Revision by {a.model_id}:\n{Path(a.content_path).read_text(encoding='utf-8', errors='replace')}"
                for a in revisions
            ]
        )
        base_prompt_text = (
            f"{prompt}\n\n"
            "IMPORTANT:\n"
            f"- After finishing the full report, add a final line exactly: {end_sentinel}\n\n"
            "---\n\n"
            f"{workflow_context}\n\n---\n\n"
            f"{data_pack_block}"
            f"{vision_transcript_block}"
            f"ORIGINAL qEEG REPORT (the source of truth - verify all claims against this):\n\n{report_text}\n\n---\n\n"
            f"REVISED ANALYSES TO CONSOLIDATE:\n\n{revision_text}\n"
        )
        try:
            max_tokens = int(os.getenv("QEEG_STAGE4_MAX_TOKENS", "8000") or "8000")
        except Exception:
            max_tokens = 8000
        if max_tokens <= 0:
            max_tokens = 8000

        await emit({"run_id": run_id, "stage_num": stage.num, "stage_name": stage.name, "status": "start"})
        text = await self._call_model_chat(
            model_id=consolidator, prompt_text=base_prompt_text, temperature=0.2, max_tokens=max_tokens
        )

        # Claude-style message APIs frequently clamp output tokens below the requested max, which can truncate
        # long consolidations. Repair by regenerating from the last fully-started required section onward.
        if not is_complete(text):
            repaired = text
            for _ in range(2):  # total up to 3 calls
                start_heading, start_idx = last_heading_present(repaired)
                if start_heading is None or start_idx is None:
                    start_heading = required_headings[0]
                    prefix = ""
                else:
                    prefix = (repaired[:start_idx] or "").rstrip()

                continuation_instruction = (
                    "Your previous output was cut off.\n"
                    "Output ONLY the remaining portion of the consolidated report.\n"
                    f"- Start with this exact heading (no text before it): {start_heading}\n"
                    f"- Continue through: {required_headings[-1]}\n"
                    f"- End with a final line exactly: {end_sentinel}\n"
                )
                cont_prompt = f"{base_prompt_text}\n\n---\n\n{continuation_instruction}"
                cont = await self._call_model_chat(
                    model_id=consolidator, prompt_text=cont_prompt, temperature=0.2, max_tokens=max_tokens
                )

                # Trim any preamble before the requested start heading.
                clean = cont
                try:
                    import re

                    m = re.search(rf"(?m)^{re.escape(start_heading)}\\s*$", cont or "")
                    if m:
                        clean = cont[m.start() :]
                except Exception:
                    clean = cont

                repaired = f"{prefix}\n\n{(clean or '').strip()}\n"
                if is_complete(repaired):
                    break
            text = repaired

        await self._write_artifact(run_id=run_id, stage=stage, model_id=consolidator, text=text)
        await emit({"run_id": run_id, "stage_num": stage.num, "stage_name": stage.name, "status": "complete"})

    async def _stage5(
        self,
        run_id: str,
        council_model_ids: list[str],
        emit: Callable[[dict[str, Any]], Awaitable[None]],
    ) -> None:
        stage = STAGES[4]
        prompt = _load_prompt("stage5_final_review.md")

        with session_scope() as session:
            s4 = _stage_artifacts(session, run_id, 4, kind="consolidation")
            run = get_run(session, run_id)
            report = get_report(session, run.report_id) if run else None

        report_text = ""
        if report and report.extracted_text_path:
            report_dir = _derive_report_dir(report)
            report_text = _load_best_report_text(report, report_dir)

        data_pack_text = ""
        dp_path = _data_pack_path(run_id)
        if dp_path.exists():
            try:
                data_pack_text = dp_path.read_text(encoding="utf-8", errors="replace")
            except Exception:
                data_pack_text = ""

        vision_transcript_text = ""
        vt_path = _vision_transcript_path(run_id)
        if vt_path.exists():
            try:
                vision_transcript_text = vt_path.read_text(encoding="utf-8", errors="replace")
            except Exception:
                vision_transcript_text = ""

        workflow_context = _workflow_context_block(stage_num=stage.num, stage_name=stage.name)
        data_pack_block = ""
        if data_pack_text.strip():
            data_pack_block = (
                "STRUCTURED DATA PACK (authoritative transcription from ALL PDF pages, including graphics):\n\n"
                f"```json\n{data_pack_text.strip()}\n```\n\n---\n\n"
            )

        vision_transcript_block = ""
        if vision_transcript_text.strip():
            vision_transcript_block = (
                "MULTIMODAL VISION TRANSCRIPT (page-grounded transcription from ALL PDF page images):\n\n"
                f"{vision_transcript_text.strip()}\n\n---\n\n"
            )

        if not s4:
            raise RuntimeError("Stage 5 requires Stage 4 consolidation artifact")

        consolidated = Path(s4[0].content_path).read_text(encoding="utf-8", errors="replace")

        await emit({"run_id": run_id, "stage_num": stage.num, "stage_name": stage.name, "status": "start"})

        async def one(model_id: str) -> tuple[str, str] | None:
            prompt_text = (
                f"{prompt}\n\n---\n\n"
                f"{workflow_context}\n\n---\n\n"
                f"{data_pack_block}"
                f"{vision_transcript_block}"
                f"ORIGINAL qEEG REPORT (for verification):\n\n{report_text}\n\n---\n\n"
                f"CONSOLIDATED REPORT TO REVIEW:\n\n{consolidated}\n"
            )
            try:
                text = await self._call_model_chat(
                    model_id=model_id, prompt_text=prompt_text, temperature=0.1, max_tokens=2500
                )
                payload = _json_loads_loose(text)
                _validate_stage5(payload)
                return model_id, json.dumps(payload, indent=2, sort_keys=True)
            except Exception:
                return None

        results = await asyncio.gather(*(one(m) for m in council_model_ids))
        successes = [r for r in results if r is not None]
        if not successes:
            raise RuntimeError("All models failed during Stage 5 final review")

        for model_id, text in successes:
            await self._write_artifact(run_id=run_id, stage=stage, model_id=model_id, text=text)

        await emit(
            {
                "run_id": run_id,
                "stage_num": stage.num,
                "stage_name": stage.name,
                "status": "complete",
                "success_count": len(successes),
                "requested_count": len(council_model_ids),
            }
        )

    async def _stage6(
        self,
        run_id: str,
        council_model_ids: list[str],
        emit: Callable[[dict[str, Any]], Awaitable[None]],
    ) -> None:
        stage = STAGES[5]
        prompt = _load_prompt("stage6_final_draft.md")

        with session_scope() as session:
            s4 = _stage_artifacts(session, run_id, 4, kind="consolidation")
            s5 = _stage_artifacts(session, run_id, 5, kind="final_review")
            run = get_run(session, run_id)
            report = get_report(session, run.report_id) if run else None

        report_text = ""
        if report and report.extracted_text_path:
            report_dir = _derive_report_dir(report)
            report_text = _load_best_report_text(report, report_dir)

        data_pack_text = ""
        dp_path = _data_pack_path(run_id)
        if dp_path.exists():
            try:
                data_pack_text = dp_path.read_text(encoding="utf-8", errors="replace")
            except Exception:
                data_pack_text = ""

        vision_transcript_text = ""
        vt_path = _vision_transcript_path(run_id)
        if vt_path.exists():
            try:
                vision_transcript_text = vt_path.read_text(encoding="utf-8", errors="replace")
            except Exception:
                vision_transcript_text = ""

        workflow_context = _workflow_context_block(stage_num=stage.num, stage_name=stage.name)
        data_pack_block = ""
        if data_pack_text.strip():
            data_pack_block = (
                "STRUCTURED DATA PACK (authoritative transcription from ALL PDF pages, including graphics):\n\n"
                f"```json\n{data_pack_text.strip()}\n```\n\n---\n\n"
            )

        vision_transcript_block = ""
        if vision_transcript_text.strip():
            vision_transcript_block = (
                "MULTIMODAL VISION TRANSCRIPT (page-grounded transcription from ALL PDF page images):\n\n"
                f"{vision_transcript_text.strip()}\n\n---\n\n"
            )

        if not s4:
            raise RuntimeError("Stage 6 requires Stage 4 consolidation artifact")
        consolidated = Path(s4[0].content_path).read_text(encoding="utf-8", errors="replace")

        required_changes = _aggregate_required_changes(s5)

        await emit({"run_id": run_id, "stage_num": stage.num, "stage_name": stage.name, "status": "start"})

        async def one(model_id: str) -> tuple[str, str] | None:
            changes = "\n".join([f"- {c}" for c in required_changes]) if required_changes else "(none)"
            prompt_text = (
                f"{prompt}\n\n---\n\n"
                f"{workflow_context}\n\n---\n\n"
                f"{data_pack_block}"
                f"{vision_transcript_block}"
                f"ORIGINAL qEEG REPORT (for any needed verification):\n\n{report_text}\n\n---\n\n"
                f"Required changes to apply:\n{changes}\n\n"
                f"CONSOLIDATED REPORT:\n\n{consolidated}\n"
            )
            try:
                text = await self._call_model_chat(
                    model_id=model_id, prompt_text=prompt_text, temperature=0.2, max_tokens=8000
                )
                return model_id, text
            except Exception:
                return None

        results = await asyncio.gather(*(one(m) for m in council_model_ids))
        successes = [r for r in results if r is not None]
        if not successes:
            raise RuntimeError("All models failed during Stage 6 final draft")

        for model_id, text in successes:
            await self._write_artifact(run_id=run_id, stage=stage, model_id=model_id, text=text)

        await emit(
            {
                "run_id": run_id,
                "stage_num": stage.num,
                "stage_name": stage.name,
                "status": "complete",
                "success_count": len(successes),
                "requested_count": len(council_model_ids),
            }
        )
