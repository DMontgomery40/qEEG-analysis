from __future__ import annotations

import asyncio
import base64
import json
import os
from pathlib import Path
from typing import Any, Awaitable, Callable

from ...config import DISCOVERED_MODEL_IDS, MODEL_ROLE_DEFAULTS, is_vision_capable
from ...model_selection import resolve_model_preference
from ...reports import extract_pdf_full
from ...storage import Report, get_report, get_run, set_run_label_map
from ...storage import session_scope
from ..ai_review_agents import run_stage2_peer_review_json, run_stage5_final_review_json
from ..constants import STAGES
from ..db_utils import _aggregate_required_changes, _stage_artifacts, _validate_stage5
from ..json_utils import _json_loads_loose
from ..paths import _data_pack_path, _stable_label_map, _vision_transcript_path
from ..prompts import _load_prompt, _workflow_context_block
from ..report_assets import (
    _derive_report_dir,
    _load_best_report_text,
    _load_page_images,
)
from ..report_text import _page_count_from_markers
from ..types import PageImage
from ..utils import _chunked, _truthy_env


class _StagesMixin:
    async def _await_with_heartbeat(
        self,
        awaitable: Awaitable[Any],
        *,
        emit: Callable[[dict[str, Any]], Awaitable[None]] | None,
        payload: dict[str, Any],
        timeout_s: int | None = None,
    ) -> Any:
        if timeout_s is not None and timeout_s <= 0:
            timeout_s = None

        interval_s = self._int_env("QEEG_PROGRESS_HEARTBEAT_S", 30)
        if emit is None or interval_s <= 0:
            if timeout_s is None:
                return await awaitable
            return await asyncio.wait_for(awaitable, timeout=timeout_s)

        task = asyncio.create_task(awaitable)
        loop = asyncio.get_running_loop()
        started_at = loop.time()
        deadline = started_at + timeout_s if timeout_s is not None else None
        heartbeat_count = 0
        while True:
            wait_s = interval_s
            if deadline is not None:
                remaining_s = deadline - loop.time()
                if remaining_s <= 0:
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass
                    raise TimeoutError(
                        f"Timed out after {timeout_s}s waiting for {payload.get('task') or 'task'}"
                    )
                wait_s = min(wait_s, remaining_s)

            try:
                return await asyncio.wait_for(asyncio.shield(task), timeout=wait_s)
            except asyncio.TimeoutError:
                if task.done():
                    return await task

                now = loop.time()
                elapsed_s = int(now - started_at)
                if deadline is not None and now >= deadline:
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass
                    raise TimeoutError(
                        f"Timed out after {timeout_s}s waiting for {payload.get('task') or 'task'}"
                    )

                heartbeat_count += 1
                heartbeat_payload = dict(payload)
                heartbeat_payload.update(
                    {
                        "status": "heartbeat",
                        "elapsed_s": elapsed_s,
                        "heartbeat_count": heartbeat_count,
                    }
                )
                await emit(heartbeat_payload)

    @staticmethod
    def _select_discovered_model_id(preferred: str) -> str | None:
        return resolve_model_preference(preferred, sorted(DISCOVERED_MODEL_IDS))

    @staticmethod
    def _has_heading(text: str, heading: str) -> bool:
        import re

        return bool(re.search(rf"(?m)^{re.escape(heading)}\s*$", text or ""))

    @classmethod
    def _heading_positions(
        cls, text: str, required_headings: list[str]
    ) -> list[tuple[str, int]]:
        import re

        positions: list[tuple[str, int]] = []
        for h in required_headings:
            m = re.search(rf"(?m)^{re.escape(h)}\s*$", text or "")
            if m:
                positions.append((h, m.start()))
        return sorted(positions, key=lambda x: x[1])

    @classmethod
    def _first_missing_heading(
        cls, text: str, required_headings: list[str]
    ) -> str | None:
        for h in required_headings:
            if not cls._has_heading(text, h):
                return h
        return None

    @classmethod
    def _is_longform_complete(
        cls,
        text: str,
        *,
        end_sentinel: str,
        required_headings: list[str] | None,
    ) -> bool:
        if end_sentinel not in (text or ""):
            return False
        if not required_headings:
            return True
        return all(cls._has_heading(text, h) for h in required_headings)

    @staticmethod
    def _int_env(name: str, default: int) -> int:
        try:
            v = int(os.getenv(name, str(default)) or str(default))
        except Exception:
            v = default
        return v

    @staticmethod
    def _compact_middle_text(text: str, *, limit_chars: int, label: str) -> str:
        if not isinstance(text, str) or not text:
            return ""
        if limit_chars <= 0 or len(text) <= limit_chars:
            return text.strip()
        marker_budget = 240
        body_budget = max(0, limit_chars - marker_budget)
        if body_budget < 1000:
            body_budget = max(0, limit_chars)
        head_chars = max(1, body_budget // 2)
        tail_chars = max(1, body_budget - head_chars)
        omitted = max(0, len(text) - head_chars - tail_chars)
        return (
            text[:head_chars].rstrip()
            + "\n\n"
            + f"[STAGE 4 {label} COMPACTED: omitted {omitted} characters to stay within the consolidator context budget.]\n\n"
            + text[-tail_chars:].lstrip()
        ).strip()

    @classmethod
    def _stage4_page_score(cls, block: str, *, page: int, total_pages: int) -> int:
        lower = (block or "").lower()
        score = 0
        if page <= 2 or (total_pages and page > total_pages - 2):
            score += 5
        weighted_terms = {
            "assessment scores": 18,
            "performance assessments": 14,
            "evoked potentials": 14,
            "physical reaction time": 12,
            "trail making": 12,
            "audio p300 delay": 12,
            "audio p300 voltage": 12,
            "moca": 14,
            "montreal cognitive": 14,
            "balance": 10,
            "theta/beta": 9,
            "alpha ratio": 9,
            "peak frequency": 9,
            "percentage change": 9,
            "maximum p300": 8,
            "central-parietal": 8,
            "central-frontal": 8,
            "session number": 8,
        }
        for term, weight in weighted_terms.items():
            if term in lower:
                score += weight
        return score

    @classmethod
    def _stage4_compact_page_marked_text(
        cls,
        text: str,
        *,
        limit_chars: int,
        marker_pattern: str,
        label: str,
    ) -> str:
        if not isinstance(text, str) or not text:
            return ""
        if limit_chars <= 0 or len(text) <= limit_chars:
            return text.strip()

        import re

        markers = list(re.finditer(marker_pattern, text, flags=re.MULTILINE))
        if not markers:
            return cls._compact_middle_text(
                text, limit_chars=limit_chars, label=label
            )

        blocks: list[tuple[int, str]] = []
        for idx, marker in enumerate(markers):
            start = marker.start()
            end = markers[idx + 1].start() if idx + 1 < len(markers) else len(text)
            try:
                page = int(marker.group(1))
            except Exception:
                page = idx + 1
            blocks.append((page, text[start:end].strip()))

        total_pages = max((page for page, _ in blocks), default=len(blocks))
        scored = [
            (
                cls._stage4_page_score(block, page=page, total_pages=total_pages),
                idx,
                page,
                block,
            )
            for idx, (page, block) in enumerate(blocks)
        ]
        scored.sort(key=lambda item: (-item[0], item[2], item[1]))

        selected: dict[int, str] = {}
        used_chars = 0
        intro_budget = 500
        available_chars = max(1000, limit_chars - intro_budget)
        for score, idx, _page, block in scored:
            if score <= 0 and selected:
                continue
            remaining = available_chars - used_chars
            if remaining <= 0:
                break
            if len(block) <= remaining:
                selected[idx] = block
                used_chars += len(block) + 2
                continue
            if score >= 12 and remaining >= 1200:
                selected[idx] = cls._compact_middle_text(
                    block,
                    limit_chars=remaining,
                    label=f"{label} PAGE",
                )
                used_chars = available_chars
                break

        if not selected:
            return cls._compact_middle_text(
                text, limit_chars=limit_chars, label=label
            )

        selected_pages = [blocks[idx][0] for idx in sorted(selected)]
        omitted_pages = [
            page for idx, (page, _block) in enumerate(blocks) if idx not in selected
        ]
        omitted_preview = ", ".join(str(p) for p in omitted_pages[:30])
        if len(omitted_pages) > 30:
            omitted_preview += f", ... ({len(omitted_pages)} total)"
        elif not omitted_preview:
            omitted_preview = "none"

        header = (
            f"[STAGE 4 {label} COMPACTED: retained pages "
            f"{', '.join(str(p) for p in selected_pages)}; omitted pages "
            f"{omitted_preview}. The structured data pack remains authoritative "
            "for extracted numeric facts across all sessions.]\n\n"
        )
        compacted = header + "\n\n".join(
            selected[idx] for idx in sorted(selected)
        )
        if len(compacted) > limit_chars:
            return cls._compact_middle_text(
                compacted, limit_chars=limit_chars, label=label
            )
        return compacted.strip()

    @classmethod
    def _stage4_source_excerpt(cls, text: str, *, limit_chars: int) -> str:
        return cls._stage4_compact_page_marked_text(
            text,
            limit_chars=limit_chars,
            marker_pattern=r"^===\s*PAGE\s+(\d+)\s*/\s*\d+\s*===\s*$",
            label="SOURCE REPORT",
        )

    @classmethod
    def _stage4_vision_excerpt(cls, text: str, *, limit_chars: int) -> str:
        return cls._stage4_compact_page_marked_text(
            text,
            limit_chars=limit_chars,
            marker_pattern=r"^##\s+Page\s+(\d+)\s*$",
            label="VISION TRANSCRIPT",
        )

    async def _call_longform_chat_with_repairs(
        self,
        *,
        model_id: str,
        prompt_text: str,
        temperature: float,
        max_tokens: int,
        end_sentinel: str,
        required_headings: list[str] | None = None,
    ) -> str:
        text = await self._call_model_chat(
            model_id=model_id,
            prompt_text=prompt_text,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        # Keep mock pipeline behavior stable for integration tests.
        require_complete = _truthy_env("QEEG_LONGFORM_REQUIRE_COMPLETE", True)
        if isinstance(model_id, str) and model_id.startswith("mock-"):
            require_complete = False
        if not require_complete:
            return text

        if self._is_longform_complete(
            text, end_sentinel=end_sentinel, required_headings=required_headings
        ):
            return text

        repair_calls = self._int_env("QEEG_LONGFORM_REPAIR_CALLS", 6)
        if repair_calls < 0:
            repair_calls = 0
        continuation_context_chars = self._int_env(
            "QEEG_LONGFORM_CONTINUATION_CONTEXT_CHARS", 12000
        )
        if continuation_context_chars <= 0:
            continuation_context_chars = 12000

        repaired = text
        headings = required_headings or []
        for _ in range(repair_calls):
            start_heading = None
            start_idx = None
            if headings:
                start_heading = self._first_missing_heading(repaired, headings)
                pos = self._heading_positions(repaired, headings)
                pos_map = {h: idx for h, idx in pos}
                if start_heading:
                    start_idx = pos_map.get(start_heading)
                else:
                    start_heading = pos[-1][0] if pos else headings[0]
                    start_idx = pos_map.get(start_heading)

            prefix = (
                repaired[:start_idx] if isinstance(start_idx, int) else repaired
            ).rstrip()
            partial_tail = (repaired or "")[-continuation_context_chars:]
            instruction_lines = [
                "Your previous output was cut off.",
                "Output ONLY the remaining portion of the report.",
            ]
            if start_heading:
                instruction_lines.append(
                    f"- Start with this exact heading (no text before it): {start_heading}"
                )
            else:
                instruction_lines.append(
                    "- Continue directly from where the text ended (do not restart)."
                )
            if headings:
                instruction_lines.append(f"- Continue through: {headings[-1]}")
            instruction_lines.append(f"- End with a final line exactly: {end_sentinel}")
            continuation_instruction = "\n".join(instruction_lines)
            cont_prompt = (
                f"{prompt_text}\n\n---\n\n"
                "CURRENT PARTIAL OUTPUT (tail; continue from this context, do not restart):\n\n"
                f"{partial_tail}\n\n---\n\n"
                f"{continuation_instruction}"
            )

            cont = await self._call_model_chat(
                model_id=model_id,
                prompt_text=cont_prompt,
                temperature=temperature,
                max_tokens=max_tokens,
            )

            clean = (cont or "").strip()
            if start_heading:
                try:
                    import re

                    m = re.search(rf"(?m)^{re.escape(start_heading)}\s*$", clean)
                    if m:
                        clean = clean[m.start() :]
                except Exception:
                    pass
                repaired = f"{prefix}\n\n{clean}\n"
            else:
                repaired = f"{prefix}\n\n{clean}\n"

            if self._is_longform_complete(
                repaired, end_sentinel=end_sentinel, required_headings=required_headings
            ):
                return repaired

        # Return best-effort text; caller can decide whether to hard-fail.
        return repaired

    async def _stage1(
        self,
        run_id: str,
        council_model_ids: list[str],
        report: Report,
        emit: Callable[[dict[str, Any]], Awaitable[None]],
    ) -> None:
        stage = STAGES[0]
        prompt = _load_prompt("stage1_analysis.md")
        end_sentinel = "<!-- END STAGE1 ANALYSIS -->"
        report_dir = _derive_report_dir(report)
        report_text = _load_best_report_text(report, report_dir)

        # Prefer a single "checker" vision model for multimodal extraction + verification.
        # This is intentionally independent from the selected council model set, so the council can use
        # text-only models while still getting page-grounded structured data + transcript.
        vision_checker_pref = os.getenv(
            "QEEG_VISION_CHECKER_MODEL", MODEL_ROLE_DEFAULTS.stage1_vision
        )
        vision_checker_id = self._select_discovered_model_id(vision_checker_pref)
        if vision_checker_id and not is_vision_capable(vision_checker_id):
            vision_checker_id = None

        # Load images from the report folder (preferred), then fallback lookup.
        page_images = _load_page_images(report, report_dir)

        needs_images = bool(vision_checker_id) or any(
            is_vision_capable(m) for m in council_model_ids
        )
        expected_page_count = _page_count_from_markers(report_text)
        pages_present = {img.page for img in page_images}
        missing_pages: list[int] = []
        if expected_page_count:
            missing_pages = [
                p for p in range(1, expected_page_count + 1) if p not in pages_present
            ]

        # If a vision-capable model is selected but images are missing/incomplete, generate on the fly.
        if (
            needs_images
            and Path(report.stored_path).suffix.lower() == ".pdf"
            and (not page_images or missing_pages)
        ):
            try:
                full = extract_pdf_full(Path(report.stored_path))
                enhanced_text = full.enhanced_text
                if enhanced_text and enhanced_text.strip():
                    report_text = enhanced_text
                    try:
                        (report_dir / "extracted_enhanced.txt").write_text(
                            enhanced_text, encoding="utf-8"
                        )
                        # Keep extracted.txt aligned so the UI preview and any verification tooling never
                        # shows only a single OCR engine.
                        (report_dir / "extracted.txt").write_text(
                            enhanced_text, encoding="utf-8"
                        )
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
                        (sources_dir / f"page-{page_num}.pypdf.txt").write_text(
                            p.get("pypdf_text", ""), encoding="utf-8"
                        )
                        (sources_dir / f"page-{page_num}.pymupdf.txt").write_text(
                            p.get("pymupdf_text", ""), encoding="utf-8"
                        )
                        (sources_dir / f"page-{page_num}.apple_vision.txt").write_text(
                            p.get("vision_ocr_text", ""), encoding="utf-8"
                        )
                        (sources_dir / f"page-{page_num}.tesseract.txt").write_text(
                            p.get("tesseract_ocr_text", ""), encoding="utf-8"
                        )

                    meta = dict(full.metadata)
                    meta.update(
                        {
                            "has_enhanced_ocr": True,
                            "has_page_images": True,
                            "page_images_written": len(page_images),
                            "sources_dir": "sources",
                        }
                    )
                    (report_dir / "metadata.json").write_text(
                        json.dumps(meta, indent=2), encoding="utf-8"
                    )
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
        if (
            is_pdf
            and not strict_data
            and not _truthy_env("QEEG_ALLOW_NONSTRICT_DATA_AVAILABILITY", False)
        ):
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
            if (
                isinstance(meta, dict)
                and meta.get("schema_version") == 2
                and isinstance(meta.get("engines"), dict)
            ):
                engines = meta["engines"]

            if not engines:
                # Metadata missing/outdated: regenerate report assets in-place (best effort) so strict runs always have
                # a full audit trail (sources/ + metadata.json).
                try:
                    full = extract_pdf_full(Path(report.stored_path))
                    enhanced_text = full.enhanced_text
                    if enhanced_text and enhanced_text.strip():
                        report_text = enhanced_text
                        (report_dir / "extracted_enhanced.txt").write_text(
                            enhanced_text, encoding="utf-8"
                        )
                        (report_dir / "extracted.txt").write_text(
                            enhanced_text, encoding="utf-8"
                        )

                    page_images = []
                    for img in full.page_images:
                        page = img.get("page") if isinstance(img, dict) else None
                        b64 = img.get("base64_png") if isinstance(img, dict) else None
                        if isinstance(page, int) and isinstance(b64, str):
                            page_images.append(PageImage(page=page, base64_png=b64))

                    pages_dir = report_dir / "pages"
                    pages_dir.mkdir(parents=True, exist_ok=True)
                    for img in page_images:
                        (pages_dir / f"page-{img.page}.png").write_bytes(
                            base64.b64decode(img.base64_png)
                        )

                    sources_dir = report_dir / "sources"
                    sources_dir.mkdir(parents=True, exist_ok=True)
                    for p in full.per_page_sources:
                        page_num = p.get("page")
                        if not isinstance(page_num, int):
                            continue
                        (sources_dir / f"page-{page_num}.pypdf.txt").write_text(
                            p.get("pypdf_text", ""), encoding="utf-8"
                        )
                        (sources_dir / f"page-{page_num}.pymupdf.txt").write_text(
                            p.get("pymupdf_text", ""), encoding="utf-8"
                        )
                        (sources_dir / f"page-{page_num}.apple_vision.txt").write_text(
                            p.get("vision_ocr_text", ""), encoding="utf-8"
                        )
                        (sources_dir / f"page-{page_num}.tesseract.txt").write_text(
                            p.get("tesseract_ocr_text", ""), encoding="utf-8"
                        )

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
                    engines = (
                        meta2.get("engines")
                        if isinstance(meta2.get("engines"), dict)
                        else {}
                    )
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
            extractor_models = [vision_checker_id] + [
                m for m in extractor_models if m != vision_checker_id
            ]
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
            emit=emit,
        )

        transcript_model_id: str | None = None
        if isinstance(data_pack, dict) and isinstance(
            data_pack.get("extraction_model_id"), str
        ):
            transcript_model_id = data_pack["extraction_model_id"]
        elif extractor_models:
            transcript_model_id = extractor_models[0]

        vision_transcript_text = await self._ensure_vision_transcript(
            run_id=run_id,
            report=report,
            page_images=page_images,
            transcript_model_id=transcript_model_id,
            strict=strict_data,
            emit=emit,
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

        workflow_context = _workflow_context_block(
            stage_num=stage.num, stage_name=stage.name
        )

        base_prompt_text = (
            f"{prompt}\n\n"
            "IMPORTANT:\n"
            f"- After finishing the full analysis, add a final line exactly: {end_sentinel}\n\n"
            "---\n\n"
            f"{workflow_context}\n\n---\n\n"
            f"{data_pack_block}"
            f"{vision_transcript_block}"
            "FULL qEEG REPORT OCR TEXT (all pages; may include OCR artifacts):\n\n"
            f"{report_text}\n"
        )

        await emit(
            {
                "run_id": run_id,
                "stage_num": stage.num,
                "stage_name": stage.name,
                "status": "start",
            }
        )
        stage1_max_tokens = self._int_env("QEEG_STAGE1_MAX_TOKENS", 12000)
        if stage1_max_tokens <= 0:
            stage1_max_tokens = 12000
        stage1_retry_max_tokens = self._int_env("QEEG_STAGE1_RETRY_MAX_TOKENS", 6000)
        if stage1_retry_max_tokens <= 0:
            stage1_retry_max_tokens = 6000
        stage1_require_complete = _truthy_env("QEEG_STAGE1_REQUIRE_COMPLETE", True)

        async def one(model_id: str) -> tuple[str, str] | None:
            try:
                await emit(
                    {
                        "run_id": run_id,
                        "stage_num": stage.num,
                        "stage_name": stage.name,
                        "task": "stage1_model",
                        "model_id": model_id,
                        "status": "start",
                    }
                )
                # Multi-pass multimodal ingestion for vision models: build page-grounded notes in chunks, then write
                # the final long-form report using the notes + full OCR + data pack.
                notes_text = ""
                per_model_notes = _truthy_env(
                    "QEEG_STAGE1_PER_MODEL_VISION_NOTES", False
                )
                if (
                    is_vision_capable(model_id)
                    and page_images
                    and (per_model_notes or not vision_transcript_block)
                ):
                    chunk_size = int(
                        os.getenv("QEEG_VISION_PAGES_PER_CALL", "8") or "8"
                    )
                    if chunk_size <= 0:
                        chunk_size = 8
                    # Hard requirement: PDFs >10 pages must be ingested in 2+ multimodal passes.
                    if len(page_images) > 10 and chunk_size > 10:
                        chunk_size = 10
                    notes_parts: list[str] = []
                    note_chunks = list(_chunked(page_images, chunk_size))
                    for chunk_index, chunk in enumerate(note_chunks, start=1):
                        pages = [img.page for img in chunk]
                        ingest_prompt = (
                            "Stage 1 multimodal ingestion pass (do NOT write the final report yet).\n"
                            f"Pages in this pass: {', '.join(str(p) for p in pages)}\n\n"
                            "Task:\n"
                            "- For each provided page image, produce a page-by-page markdown transcript with headings "
                            '"## Page <n>".\n'
                            "- Enumerate every table/figure/metric visible on that page and transcribe any clearly "
                            "printed numeric values that are likely clinically relevant.\n"
                            "- Do not interpret or diagnose. Do not invent numbers.\n"
                        )
                        await emit(
                            {
                                "run_id": run_id,
                                "stage_num": stage.num,
                                "stage_name": stage.name,
                                "task": "stage1_vision_notes_chunk",
                                "model_id": model_id,
                                "status": "start",
                                "chunk_index": chunk_index,
                                "chunk_count": len(note_chunks),
                                "pages": pages,
                            }
                        )
                        notes = await self._await_with_heartbeat(
                            self._call_model_multimodal(
                                model_id=model_id,
                                prompt_text=ingest_prompt,
                                images=chunk,
                                temperature=0.0,
                                max_tokens=2500,
                                allow_text_fallback=not strict_data,
                            ),
                            emit=emit,
                            payload={
                                "run_id": run_id,
                                "stage_num": stage.num,
                                "stage_name": stage.name,
                                "task": "stage1_vision_notes_chunk",
                                "model_id": model_id,
                                "chunk_index": chunk_index,
                                "chunk_count": len(note_chunks),
                                "pages": pages,
                            },
                        )
                        await emit(
                            {
                                "run_id": run_id,
                                "stage_num": stage.num,
                                "stage_name": stage.name,
                                "task": "stage1_vision_notes_chunk",
                                "model_id": model_id,
                                "status": "complete",
                                "chunk_index": chunk_index,
                                "chunk_count": len(note_chunks),
                                "pages": pages,
                            }
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
                stage1_payload = {
                    "run_id": run_id,
                    "stage_num": stage.num,
                    "stage_name": stage.name,
                    "task": "stage1_model",
                    "model_id": model_id,
                }
                try:
                    text = await self._await_with_heartbeat(
                        self._call_longform_chat_with_repairs(
                            model_id=model_id,
                            prompt_text=final_prompt.strip(),
                            temperature=0.2,
                            max_tokens=stage1_max_tokens,
                            end_sentinel=end_sentinel,
                            required_headings=None,
                        ),
                        emit=emit,
                        payload=stage1_payload,
                    )
                except Exception as primary_exc:
                    if isinstance(model_id, str) and model_id.startswith("mock-"):
                        raise
                    await emit(
                        {
                            **stage1_payload,
                            "status": "retry",
                            "error": str(primary_exc)
                            or primary_exc.__class__.__name__,
                            "max_tokens": stage1_retry_max_tokens,
                            "operatorHint": "Stage 1 upstream failed after heartbeat; retrying the same model with a smaller complete-output budget.",
                        }
                    )
                    retry_prompt = (
                        f"{final_prompt.strip()}\n\n---\n\n"
                        "RETRY CONSTRAINT:\n"
                        "The previous upstream call failed before returning usable text. "
                        "Return a concise but complete Stage 1 analysis, preserve all critical numeric findings, "
                        "do not invent missing values, and still end with the required sentinel line.\n"
                    )
                    text = await self._await_with_heartbeat(
                        self._call_longform_chat_with_repairs(
                            model_id=model_id,
                            prompt_text=retry_prompt,
                            temperature=0.2,
                            max_tokens=stage1_retry_max_tokens,
                            end_sentinel=end_sentinel,
                            required_headings=None,
                        ),
                        emit=emit,
                        payload={**stage1_payload, "attempt": "retry"},
                    )
                enforce_complete = stage1_require_complete and not (
                    isinstance(model_id, str) and model_id.startswith("mock-")
                )
                if enforce_complete and not self._is_longform_complete(
                    text,
                    end_sentinel=end_sentinel,
                    required_headings=None,
                ):
                    raise RuntimeError(
                        "Stage 1 analysis remained incomplete after repair attempts. "
                        f"End sentinel present: {end_sentinel in (text or '')}"
                    )
                await emit(
                    {
                        "run_id": run_id,
                        "stage_num": stage.num,
                        "stage_name": stage.name,
                        "task": "stage1_model",
                        "model_id": model_id,
                        "status": "complete",
                    }
                )
                return model_id, text
            except Exception as exc:
                await emit(
                    {
                        "run_id": run_id,
                        "stage_num": stage.num,
                        "stage_name": stage.name,
                        "task": "stage1_model",
                        "model_id": model_id,
                        "status": "failed",
                        "error": str(exc),
                    }
                )
                return None

        results = await asyncio.gather(*(one(m) for m in council_model_ids))
        successes = [r for r in results if r is not None]
        if not successes:
            raise RuntimeError("All models failed during Stage 1 analysis")

        for model_id, text in successes:
            await self._write_artifact(
                run_id=run_id, stage=stage, model_id=model_id, text=text
            )

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
                vision_transcript_text = vt_path.read_text(
                    encoding="utf-8", errors="replace"
                )
            except Exception:
                vision_transcript_text = ""

        workflow_context = _workflow_context_block(
            stage_num=stage.num, stage_name=stage.name
        )
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
            analyses_by_model[a.model_id] = Path(a.content_path).read_text(
                encoding="utf-8", errors="replace"
            )

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

        await emit(
            {
                "run_id": run_id,
                "stage_num": stage.num,
                "stage_name": stage.name,
                "status": "start",
            }
        )

        async def one(reviewer_model_id: str) -> tuple[str, str] | None:
            # Reviewer sees all analyses except its own, still labeled A/B/C...
            reviewer_label = next(
                (lbl for lbl, mid in label_map.items() if mid == reviewer_model_id),
                None,
            )
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
                "ORIGINAL qEEG REPORT OCR TEXT (tertiary verification/context only; if numeric conflicts exist, "
                "STRUCTURED DATA PACK values are authoritative):\n\n"
                f"{report_text}\n\n---\n\n"
                f"Reviewer Model ID: {reviewer_model_id}\n"
                f"Your own analysis label (do not review yourself): {reviewer_label}\n\n"
                f"ANALYSES TO REVIEW:\n\n{filtered_text}\n"
            )
            try:
                expected_labels = [
                    label
                    for label, mid in label_map.items()
                    if mid != reviewer_model_id
                ]
                text = await run_stage2_peer_review_json(
                    llm_client=self._llm,
                    model_id=reviewer_model_id,
                    prompt_text=prompt_text,
                    expected_labels=expected_labels,
                )
                return reviewer_model_id, text
            except Exception:
                return None

        results = await asyncio.gather(*(one(m) for m in available_models))
        successes = [r for r in results if r is not None]

        for model_id, text in successes:
            await self._write_artifact(
                run_id=run_id, stage=stage, model_id=model_id, text=text
            )

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
        required_headings = None
        end_sentinel = "<!-- END STAGE3 REVISION -->"

        with session_scope() as session:
            s1 = _stage_artifacts(session, run_id, 1, kind="analysis")
            s2 = _stage_artifacts(session, run_id, 2, kind="peer_review")
            run = get_run(session, run_id)
            label_map = (
                json.loads(run.label_map_json or "{}") if run is not None else {}
            )
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
                vision_transcript_text = vt_path.read_text(
                    encoding="utf-8", errors="replace"
                )
            except Exception:
                vision_transcript_text = ""

        workflow_context = _workflow_context_block(
            stage_num=stage.num, stage_name=stage.name
        )
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
            a.model_id: Path(a.content_path).read_text(
                encoding="utf-8", errors="replace"
            )
            for a in s1
        }
        peer_reviews = [
            (
                a.model_id,
                Path(a.content_path).read_text(encoding="utf-8", errors="replace"),
            )
            for a in s2
        ]

        available_models = [m for m in council_model_ids if m in analyses_by_model]
        if not available_models:
            raise RuntimeError("No Stage 1 analyses available for revision")

        await emit(
            {
                "run_id": run_id,
                "stage_num": stage.num,
                "stage_name": stage.name,
                "status": "start",
            }
        )
        stage3_max_tokens = self._int_env("QEEG_STAGE3_MAX_TOKENS", 12000)
        if stage3_max_tokens <= 0:
            stage3_max_tokens = 12000
        stage3_require_complete = _truthy_env("QEEG_STAGE3_REQUIRE_COMPLETE", True)
        stage3_timeout_s = self._int_env("QEEG_STAGE3_MODEL_TIMEOUT_S", 0)

        async def one(model_id: str) -> tuple[str, str] | None:
            analysis = analyses_by_model.get(model_id)
            if not analysis:
                return None
            my_label = next(
                (lbl for lbl, mid in label_map.items() if mid == model_id), None
            )
            pr_text = "\n\n".join(
                [f"Peer review by {mid}:\n{txt}" for mid, txt in peer_reviews]
            ).strip()
            prompt_text = (
                f"{prompt}\n\n"
                "IMPORTANT:\n"
                f"- After finishing the full revision, add a final line exactly: {end_sentinel}\n\n"
                "---\n\n"
                f"{workflow_context}\n\n---\n\n"
                f"{data_pack_block}"
                f"{vision_transcript_block}"
                "ORIGINAL qEEG REPORT OCR TEXT (tertiary fact-checking/context only; if numeric conflicts exist, "
                "STRUCTURED DATA PACK values are authoritative):\n\n"
                f"{report_text}\n\n---\n\n"
                f"Your Model ID: {model_id}\n"
                f"Your analysis label (if present): {my_label}\n\n"
                f"Your original analysis:\n\n{analysis}\n\n"
                f"Peer review JSON artifacts:\n\n{pr_text}\n"
            )
            event_payload = {
                "run_id": run_id,
                "stage_num": stage.num,
                "stage_name": stage.name,
                "task": "stage3_model",
                "model_id": model_id,
            }
            await emit({**event_payload, "status": "start"})
            try:
                text = await self._await_with_heartbeat(
                    self._call_longform_chat_with_repairs(
                        model_id=model_id,
                        prompt_text=prompt_text.strip(),
                        temperature=0.2,
                        max_tokens=stage3_max_tokens,
                        end_sentinel=end_sentinel,
                        required_headings=required_headings,
                    ),
                    emit=emit,
                    payload=event_payload,
                    timeout_s=stage3_timeout_s,
                )
                enforce_complete = stage3_require_complete and not (
                    isinstance(model_id, str) and model_id.startswith("mock-")
                )
                if enforce_complete and not self._is_longform_complete(
                    text,
                    end_sentinel=end_sentinel,
                    required_headings=required_headings,
                ):
                    raise RuntimeError(
                        "Stage 3 revision remained incomplete after repair attempts. "
                        f"end sentinel present: {end_sentinel in (text or '')}"
                    )
                await emit({**event_payload, "status": "complete"})
                return model_id, text
            except Exception as exc:
                await emit(
                    {
                        **event_payload,
                        "status": "failed",
                        "error": str(exc) or exc.__class__.__name__,
                        "operatorHint": "Stage 3 model call failed or timed out; retry with a reachable CLIProxy upstream or a smaller model set.",
                    }
                )
                return None

        results = await asyncio.gather(*(one(m) for m in available_models))
        successes = [r for r in results if r is not None]
        if not successes:
            raise RuntimeError("All models failed during Stage 3 revision")

        for model_id, text in successes:
            await self._write_artifact(
                run_id=run_id, stage=stage, model_id=model_id, text=text
            )
        await emit(
            {
                "run_id": run_id,
                "stage_num": stage.num,
                "stage_name": stage.name,
                "status": "complete",
                "success_count": len(successes),
                "requested_count": len(available_models),
                "partial_success": len(successes) < len(available_models),
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
        ]
        repair_headings = (
            required_headings
            if _truthy_env("QEEG_STAGE4_REQUIRE_HEADINGS", False)
            else []
        )
        end_sentinel = "<!-- END CONSOLIDATED REPORT -->"

        def has_heading(text: str, heading: str) -> bool:
            import re

            return bool(re.search(rf"(?m)^{re.escape(heading)}\s*$", text or ""))

        def heading_positions(text: str) -> list[tuple[str, int]]:
            import re

            positions: list[tuple[str, int]] = []
            for h in repair_headings:
                m = re.search(rf"(?m)^{re.escape(h)}\s*$", text or "")
                if m:
                    positions.append((h, m.start()))
            return sorted(positions, key=lambda x: x[1])

        def first_missing_heading(text: str) -> str | None:
            for h in repair_headings:
                if not has_heading(text, h):
                    return h
            return None

        def is_complete(text: str) -> bool:
            if end_sentinel not in (text or ""):
                return False
            if not repair_headings:
                return True
            return all(has_heading(text, h) for h in repair_headings)

        def last_heading_present(text: str) -> tuple[str, int] | tuple[None, None]:
            positions = heading_positions(text)
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
                vision_transcript_text = vt_path.read_text(
                    encoding="utf-8", errors="replace"
                )
            except Exception:
                vision_transcript_text = ""

        workflow_context = _workflow_context_block(
            stage_num=stage.num, stage_name=stage.name
        )
        data_pack_block = ""
        if data_pack_text.strip():
            data_pack_block = (
                "STRUCTURED DATA PACK (authoritative transcription from ALL PDF pages, including graphics):\n\n"
                f"```json\n{data_pack_text.strip()}\n```\n\n---\n\n"
            )

        report_text_for_prompt = self._stage4_source_excerpt(
            report_text,
            limit_chars=self._int_env("QEEG_STAGE4_REPORT_TEXT_CHAR_LIMIT", 70000),
        )
        vision_transcript_for_prompt = self._stage4_vision_excerpt(
            vision_transcript_text,
            limit_chars=self._int_env(
                "QEEG_STAGE4_VISION_TRANSCRIPT_CHAR_LIMIT", 30000
            ),
        )
        revision_limit = self._int_env("QEEG_STAGE4_REVISION_CHAR_LIMIT_PER_MODEL", 0)

        vision_transcript_block = ""
        if vision_transcript_for_prompt.strip():
            vision_transcript_block = (
                "MULTIMODAL VISION TRANSCRIPT (page-grounded transcription from ALL PDF page images):\n\n"
                f"{vision_transcript_for_prompt.strip()}\n\n---\n\n"
            )

        if not revisions:
            raise RuntimeError("Consolidation requires at least one revision artifact")

        revision_text = "\n\n".join(
            [
                "Revision by "
                f"{a.model_id}:\n"
                f"{self._compact_middle_text(Path(a.content_path).read_text(encoding='utf-8', errors='replace'), limit_chars=revision_limit, label='REVISION')}"
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
            "ORIGINAL qEEG REPORT EXCERPT (source text selected for Stage 4 context; "
            "verify numeric claims against the structured data pack above):\n\n"
            f"{report_text_for_prompt}\n\n---\n\n"
            f"REVISED ANALYSES TO CONSOLIDATE:\n\n{revision_text}\n"
        )
        try:
            max_tokens = int(os.getenv("QEEG_STAGE4_MAX_TOKENS", "12000") or "12000")
        except Exception:
            max_tokens = 12000
        if max_tokens <= 0:
            max_tokens = 12000
        try:
            repair_calls = int(os.getenv("QEEG_STAGE4_REPAIR_CALLS", "6") or "6")
        except Exception:
            repair_calls = 6
        if repair_calls < 0:
            repair_calls = 0
        try:
            continuation_context_chars = int(
                os.getenv("QEEG_STAGE4_CONTINUATION_CONTEXT_CHARS", "12000") or "12000"
            )
        except Exception:
            continuation_context_chars = 12000
        if continuation_context_chars <= 0:
            continuation_context_chars = 12000
        require_complete = _truthy_env("QEEG_STAGE4_REQUIRE_COMPLETE", True)
        if isinstance(consolidator, str) and consolidator.startswith("mock-"):
            require_complete = False
        stage4_timeout_s = self._int_env("QEEG_STAGE4_MODEL_TIMEOUT_S", 0)

        await emit(
            {
                "run_id": run_id,
                "stage_num": stage.num,
                "stage_name": stage.name,
                "status": "start",
            }
        )
        task_payload = {
            "run_id": run_id,
            "stage_num": stage.num,
            "stage_name": stage.name,
            "task": "stage4_consolidation",
            "model_id": consolidator,
        }
        await emit({**task_payload, "status": "start"})
        text = await self._await_with_heartbeat(
            self._call_model_chat(
                model_id=consolidator,
                prompt_text=base_prompt_text,
                temperature=0.2,
                max_tokens=max_tokens,
            ),
            emit=emit,
            payload=task_payload,
            timeout_s=stage4_timeout_s,
        )

        # Claude-style message APIs frequently clamp output tokens below the requested max, which can truncate
        # long consolidations. Repair by regenerating from the next missing (or last present) required section onward.
        if not is_complete(text):
            repaired = text
            for _ in range(repair_calls):
                # Prefer resuming from the first missing section. If all required sections are present but the
                # sentinel is missing, resume from the last heading and ask for a clean ending.
                start_heading = (
                    first_missing_heading(repaired) if repair_headings else None
                )
                positions = heading_positions(repaired) if repair_headings else []
                pos_map = {h: idx for h, idx in positions}
                start_idx = pos_map.get(start_heading) if start_heading else None
                if repair_headings and start_heading is None:
                    start_heading, start_idx = last_heading_present(repaired)
                if repair_headings and start_heading is None:
                    start_heading = repair_headings[0]
                    start_idx = None

                prefix = (
                    (repaired[:start_idx] or "").rstrip()
                    if repair_headings and start_idx is not None
                    else (repaired or "").rstrip()
                )

                partial_tail = (repaired or "")[-continuation_context_chars:]
                continuation_instruction = (
                    "Your previous output was cut off.\n"
                    "Output ONLY the remaining portion of the consolidated report.\n"
                )
                if repair_headings and start_heading:
                    continuation_instruction += (
                        f"- Start with this exact heading (no text before it): {start_heading}\n"
                        f"- Continue through: {repair_headings[-1]}\n"
                    )
                continuation_instruction += (
                    f"- End with a final line exactly: {end_sentinel}\n"
                )
                cont_prompt = (
                    f"{base_prompt_text}\n\n---\n\n"
                    "CURRENT PARTIAL CONSOLIDATION (tail; continue from this context, do not restart):\n\n"
                    f"{partial_tail}\n\n---\n\n"
                    f"{continuation_instruction}"
                )
                cont = await self._await_with_heartbeat(
                    self._call_model_chat(
                        model_id=consolidator,
                        prompt_text=cont_prompt,
                        temperature=0.2,
                        max_tokens=max_tokens,
                    ),
                    emit=emit,
                    payload={**task_payload, "task": "stage4_repair"},
                    timeout_s=stage4_timeout_s,
                )

                # Trim any preamble before the requested start heading.
                clean = cont
                if repair_headings and start_heading:
                    try:
                        import re

                        m = re.search(
                            rf"(?m)^{re.escape(start_heading)}\s*$", cont or ""
                        )
                        if m:
                            clean = cont[m.start() :]
                    except Exception:
                        clean = cont

                repaired = f"{prefix}\n\n{(clean or '').strip()}\n"
                if is_complete(repaired):
                    break
            text = repaired

        if not is_complete(text):
            missing = [h for h in repair_headings if not has_heading(text, h)]
            detail = (
                "Stage 4 consolidation remained incomplete after repair attempts.\n"
                f"Missing headings: {missing if missing else '(none)'}\n"
                f"End sentinel present: {end_sentinel in (text or '')}\n"
                "Increase QEEG_STAGE4_MAX_TOKENS and/or QEEG_STAGE4_REPAIR_CALLS, then retry."
            )
            if require_complete:
                raise RuntimeError(detail)

        await self._write_artifact(
            run_id=run_id, stage=stage, model_id=consolidator, text=text
        )
        await emit({**task_payload, "status": "complete"})
        await emit(
            {
                "run_id": run_id,
                "stage_num": stage.num,
                "stage_name": stage.name,
                "status": "complete",
            }
        )

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
                vision_transcript_text = vt_path.read_text(
                    encoding="utf-8", errors="replace"
                )
            except Exception:
                vision_transcript_text = ""

        workflow_context = _workflow_context_block(
            stage_num=stage.num, stage_name=stage.name
        )
        data_pack_block = ""
        if data_pack_text.strip():
            data_pack_block = (
                "STRUCTURED DATA PACK (authoritative transcription from ALL PDF pages, including graphics):\n\n"
                f"```json\n{data_pack_text.strip()}\n```\n\n---\n\n"
            )

        report_text_for_prompt = self._stage4_source_excerpt(
            report_text,
            limit_chars=self._int_env(
                "QEEG_STAGE5_REPORT_TEXT_CHAR_LIMIT",
                self._int_env("QEEG_STAGE4_REPORT_TEXT_CHAR_LIMIT", 70000),
            ),
        )
        vision_transcript_for_prompt = self._stage4_vision_excerpt(
            vision_transcript_text,
            limit_chars=self._int_env(
                "QEEG_STAGE5_VISION_TRANSCRIPT_CHAR_LIMIT",
                self._int_env("QEEG_STAGE4_VISION_TRANSCRIPT_CHAR_LIMIT", 30000),
            ),
        )

        vision_transcript_block = ""
        if vision_transcript_for_prompt.strip():
            vision_transcript_block = (
                "MULTIMODAL VISION TRANSCRIPT (page-grounded transcription from ALL PDF page images):\n\n"
                f"{vision_transcript_for_prompt.strip()}\n\n---\n\n"
            )

        if not s4:
            raise RuntimeError("Stage 5 requires Stage 4 consolidation artifact")

        consolidated = Path(s4[0].content_path).read_text(
            encoding="utf-8", errors="replace"
        )

        await emit(
            {
                "run_id": run_id,
                "stage_num": stage.num,
                "stage_name": stage.name,
                "status": "start",
            }
        )

        async def one(model_id: str) -> tuple[str, str] | None:
            prompt_text = (
                f"{prompt}\n\n---\n\n"
                f"{workflow_context}\n\n---\n\n"
                f"{data_pack_block}"
                f"{vision_transcript_block}"
                "ORIGINAL qEEG REPORT EXCERPT (for verification; verify numeric claims against the structured data pack above):\n\n"
                f"{report_text_for_prompt}\n\n---\n\n"
                f"CONSOLIDATED REPORT TO REVIEW:\n\n{consolidated}\n"
            )
            try:
                text = await run_stage5_final_review_json(
                    llm_client=self._llm,
                    model_id=model_id,
                    prompt_text=prompt_text,
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
            await self._write_artifact(
                run_id=run_id, stage=stage, model_id=model_id, text=text
            )

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
        required_headings = [
            "# Dataset and Sessions",
            "# Key Empirical Findings",
            "# Performance Assessments",
            "# Auditory ERP: P300 and N100",
            "# Background EEG Metrics",
            "# Speculative Commentary and Interpretive Hypotheses",
        ]
        end_sentinel = "<!-- END STAGE6 FINAL DRAFT -->"
        writer_model = (
            os.getenv(
                "QEEG_STAGE6_FINAL_DRAFT_MODEL",
                MODEL_ROLE_DEFAULTS.stage6_final_draft,
            )
            or MODEL_ROLE_DEFAULTS.stage6_final_draft
        ).strip()
        if not writer_model:
            raise RuntimeError("Stage 6 final-draft writer model is not configured")

        with session_scope() as session:
            s4 = _stage_artifacts(session, run_id, 4, kind="consolidation")
            s5 = _stage_artifacts(session, run_id, 5, kind="final_review")
            run = get_run(session, run_id)
            report = get_report(session, run.report_id) if run else None

        writer_candidates: list[str] = []
        if DISCOVERED_MODEL_IDS:
            fallback_writer = (
                os.getenv("QEEG_STAGE6_FINAL_DRAFT_FALLBACK_MODEL", "kimi-k3")
                or "kimi-k3"
            ).strip()
            candidate_preferences = [writer_model, fallback_writer]
            for preference in candidate_preferences:
                resolved = self._select_discovered_model_id(preference)
                if resolved and resolved not in writer_candidates:
                    writer_candidates.append(resolved)
        else:
            # Model discovery is populated by the live application. Retain the
            # configured writer for isolated workflows and test harnesses.
            writer_candidates.append(writer_model)

        if not writer_candidates:
            raise RuntimeError(
                "Stage 6 has no available final-draft model. "
                f"Configured writer: {writer_model}"
            )

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
                vision_transcript_text = vt_path.read_text(
                    encoding="utf-8", errors="replace"
                )
            except Exception:
                vision_transcript_text = ""

        workflow_context = _workflow_context_block(
            stage_num=stage.num, stage_name=stage.name
        )
        data_pack_block = ""
        if data_pack_text.strip():
            data_pack_block = (
                "STRUCTURED DATA PACK (authoritative transcription from ALL PDF pages, including graphics):\n\n"
                f"```json\n{data_pack_text.strip()}\n```\n\n---\n\n"
            )

        report_text_for_prompt = self._stage4_source_excerpt(
            report_text,
            limit_chars=self._int_env(
                "QEEG_STAGE6_REPORT_TEXT_CHAR_LIMIT",
                self._int_env("QEEG_STAGE4_REPORT_TEXT_CHAR_LIMIT", 70000),
            ),
        )
        vision_transcript_for_prompt = self._stage4_vision_excerpt(
            vision_transcript_text,
            limit_chars=self._int_env(
                "QEEG_STAGE6_VISION_TRANSCRIPT_CHAR_LIMIT",
                self._int_env("QEEG_STAGE4_VISION_TRANSCRIPT_CHAR_LIMIT", 30000),
            ),
        )

        vision_transcript_block = ""
        if vision_transcript_for_prompt.strip():
            vision_transcript_block = (
                "MULTIMODAL VISION TRANSCRIPT (page-grounded transcription from ALL PDF page images):\n\n"
                f"{vision_transcript_for_prompt.strip()}\n\n---\n\n"
            )

        if not s4:
            raise RuntimeError("Stage 6 requires Stage 4 consolidation artifact")
        consolidated = Path(s4[0].content_path).read_text(
            encoding="utf-8", errors="replace"
        )

        required_changes = _aggregate_required_changes(s5)

        await emit(
            {
                "run_id": run_id,
                "stage_num": stage.num,
                "stage_name": stage.name,
                "status": "start",
                "model_id": writer_candidates[0],
                "candidate_count": len(writer_candidates),
            }
        )
        stage6_max_tokens = self._int_env("QEEG_STAGE6_MAX_TOKENS", 12000)
        if stage6_max_tokens <= 0:
            stage6_max_tokens = 12000
        stage6_require_complete = _truthy_env("QEEG_STAGE6_REQUIRE_COMPLETE", True)

        failures: list[str] = []

        async def one(model_id: str) -> tuple[str, str] | None:
            changes = (
                "\n".join([f"- {c}" for c in required_changes])
                if required_changes
                else "(none)"
            )
            prompt_text = (
                f"{prompt}\n\n"
                "IMPORTANT:\n"
                f"- After finishing the full final draft, add a final line exactly: {end_sentinel}\n\n"
                "---\n\n"
                f"{workflow_context}\n\n---\n\n"
                f"{data_pack_block}"
                f"{vision_transcript_block}"
                "ORIGINAL qEEG REPORT EXCERPT (for any needed verification; verify numeric claims against the structured data pack above):\n\n"
                f"{report_text_for_prompt}\n\n---\n\n"
                f"Required changes to apply:\n{changes}\n\n"
                f"CONSOLIDATED REPORT:\n\n{consolidated}\n"
            )
            try:
                text = await self._call_longform_chat_with_repairs(
                    model_id=model_id,
                    prompt_text=prompt_text.strip(),
                    temperature=0.2,
                    max_tokens=stage6_max_tokens,
                    end_sentinel=end_sentinel,
                    required_headings=required_headings,
                )
                enforce_complete = stage6_require_complete and not (
                    isinstance(model_id, str) and model_id.startswith("mock-")
                )
                if enforce_complete and not self._is_longform_complete(
                    text,
                    end_sentinel=end_sentinel,
                    required_headings=required_headings,
                ):
                    raise RuntimeError(
                        "Stage 6 final draft remained incomplete after repair attempts. "
                        f"end sentinel present: {end_sentinel in (text or '')}"
                    )
                return model_id, text
            except Exception as exc:
                failures.append(f"{model_id}: {type(exc).__name__}: {exc}")
                return None

        successes: list[tuple[str, str]] = []
        for candidate in writer_candidates:
            result = await one(candidate)
            if result is not None:
                successes.append(result)
                break
        if not successes:
            detail = "; ".join(failures) or "no model call was attempted"
            raise RuntimeError(
                f"All models failed during Stage 6 final draft. {detail}"
            )

        for model_id, text in successes:
            await self._write_artifact(
                run_id=run_id, stage=stage, model_id=model_id, text=text
            )

        # Stage 6 wants exactly one final draft; `writer_candidates` is a
        # fallback chain tried in order until one works, not a fan-out. Counting
        # the chain length here reported 1/2 on every healthy run, which
        # `run_council_completion_gaps` reads as "partial council output" and
        # `run_downstream_delivery_gaps` then treats as a reason to withhold the
        # patient-facing document — so from 2026-08-02 no run could publish at
        # all. The chain length is already reported as `candidate_count` on this
        # stage's start event.
        await emit(
            {
                "run_id": run_id,
                "stage_num": stage.num,
                "stage_name": stage.name,
                "status": "complete",
                "success_count": len(successes),
                "requested_count": 1,
            }
        )
