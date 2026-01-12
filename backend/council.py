from __future__ import annotations

import asyncio
import base64
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Awaitable, Callable

from .config import ARTIFACTS_DIR, is_vision_capable
from .llm_client import AsyncOpenAICompatClient, UpstreamError
from .reports import extract_pdf_with_images, extract_text_from_pdf, get_page_images_base64, get_enhanced_text
from .storage import (
    Artifact,
    Report,
    Run,
    create_artifact,
    get_report,
    get_run,
    set_run_label_map,
    update_run_status,
)
from .storage import session_scope


OnEvent = Callable[[dict[str, Any]], Awaitable[None]] | None


@dataclass(frozen=True)
class StageDef:
    num: int
    name: str
    kind: str
    content_type: str
    ext: str


STAGES: list[StageDef] = [
    StageDef(1, "initial_analysis", "analysis", "text/markdown", ".md"),
    StageDef(2, "peer_review", "peer_review", "application/json", ".json"),
    StageDef(3, "revision", "revision", "text/markdown", ".md"),
    StageDef(4, "consolidation", "consolidation", "text/markdown", ".md"),
    StageDef(5, "final_review", "final_review", "application/json", ".json"),
    StageDef(6, "final_draft", "final_draft", "text/markdown", ".md"),
]


def _prompt_path(name: str) -> Path:
    return Path(__file__).parent / "prompts" / name


def _load_prompt(name: str) -> str:
    path = _prompt_path(name)
    return path.read_text(encoding="utf-8")


def _stage_dir(run_id: str, stage_num: int) -> Path:
    return ARTIFACTS_DIR / run_id / f"stage-{stage_num}"


def _artifact_path(run_id: str, stage_num: int, model_id: str, ext: str) -> Path:
    # model_id is safe enough for filenames in practice; fall back to a stable replacement if needed.
    safe_model = "".join(c if c.isalnum() or c in {"-", "_", "."} else "_" for c in model_id) or "model"
    return _stage_dir(run_id, stage_num) / f"{safe_model}{ext}"


def _stable_label_map(run_id: str, model_ids: list[str]) -> dict[str, str]:
    rng = random.Random(run_id)
    ids = list(model_ids)
    rng.shuffle(ids)
    labels = [chr(ord("A") + i) for i in range(len(ids))]
    return {label: model_id for label, model_id in zip(labels, ids)}


def _strip_to_json(text: str) -> str:
    s = text.strip()
    if not s:
        return s
    if s[0] in "{[" and s[-1] in "}]":
        return s
    # Try to extract the outermost JSON-looking object/array.
    starts = [s.find("{"), s.find("[")]
    first = min((i for i in starts if i != -1), default=-1)
    if first == -1:
        return s
    last_curly = s.rfind("}")
    last_square = s.rfind("]")
    last = max(last_curly, last_square)
    if last <= first:
        return s
    return s[first : last + 1].strip()


def _json_loads_loose(text: str) -> Any:
    return json.loads(_strip_to_json(text))


async def _sleep_backoff(attempt: int) -> None:
    # Exponential backoff with jitter; attempt starts at 0.
    base = min(8.0, 0.5 * (2**attempt))
    jitter = random.random() * 0.2
    await asyncio.sleep(base + jitter)


class QEEGCouncilWorkflow:
    def __init__(self, *, llm: AsyncOpenAICompatClient):
        self._llm = llm

    async def run_pipeline(self, run_id: str, on_event: OnEvent = None) -> None:
        async def emit(payload: dict[str, Any]) -> None:
            if on_event is not None:
                await on_event(payload)

        with session_scope() as session:
            run = get_run(session, run_id)
            if run is None:
                return
            report = get_report(session, run.report_id)
            if report is None:
                update_run_status(session, run_id, status="failed", error_message="Report not found")
                return

            council_model_ids = _loads_json_list(run.council_model_ids_json)
            if not council_model_ids:
                update_run_status(
                    session,
                    run_id,
                    status="failed",
                    error_message="No council models selected for run",
                )
                return

            update_run_status(session, run_id, status="running")

        await emit({"run_id": run_id, "status": "running"})

        try:
            await self._stage1(run_id, council_model_ids, report, emit)
            await self._stage2(run_id, council_model_ids, emit)
            await self._stage3(run_id, council_model_ids, emit)
            await self._stage4(run_id, emit)
            await self._stage5(run_id, council_model_ids, emit)
            await self._stage6(run_id, council_model_ids, emit)
        except _NeedsAuth as e:
            with session_scope() as session:
                update_run_status(session, run_id, status="needs_auth", error_message=str(e))
            await emit({"run_id": run_id, "status": "needs_auth", "error": str(e)})
            return
        except Exception as e:
            with session_scope() as session:
                update_run_status(session, run_id, status="failed", error_message=str(e))
            await emit({"run_id": run_id, "status": "failed", "error": str(e)})
            return

        with session_scope() as session:
            update_run_status(session, run_id, status="complete")
        await emit({"run_id": run_id, "status": "complete"})

    async def _call_model_chat(
        self,
        *,
        model_id: str,
        prompt_text: str,
        temperature: float,
        max_tokens: int,
    ) -> str:
        attempts = 0
        while True:
            try:
                return await self._llm.chat_completions(
                    model_id=model_id,
                    messages=[{"role": "user", "content": prompt_text}],
                    temperature=temperature,
                    max_tokens=max_tokens,
                    stream=False,
                )
            except UpstreamError as e:
                if e.status_code == 401:
                    raise _NeedsAuth(str(e)) from e
                if e.status_code in {429, 502, 503} and attempts < 4:
                    await _sleep_backoff(attempts)
                    attempts += 1
                    continue
                raise

    async def _call_model_multimodal(
        self,
        *,
        model_id: str,
        prompt_text: str,
        images_base64: list[str],
        temperature: float,
        max_tokens: int,
    ) -> str:
        """Call a vision-capable model with text and images."""
        # Build multimodal content array
        content: list[dict] = [{"type": "text", "text": prompt_text}]

        # Add images (limit to first 10 pages to avoid token limits)
        for i, img_base64 in enumerate(images_base64[:10]):
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{img_base64}",
                    "detail": "high"  # Use high detail for clinical data
                }
            })

        messages = [{"role": "user", "content": content}]

        attempts = 0
        while True:
            try:
                return await self._llm.chat_completions(
                    model_id=model_id,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    stream=False,
                )
            except UpstreamError as e:
                if e.status_code == 401:
                    raise _NeedsAuth(str(e)) from e
                if e.status_code in {429, 502, 503} and attempts < 4:
                    await _sleep_backoff(attempts)
                    attempts += 1
                    continue
                # If multimodal fails, fall back to text-only
                if attempts == 0:
                    return await self._call_model_chat(
                        model_id=model_id,
                        prompt_text=prompt_text,
                        temperature=temperature,
                        max_tokens=max_tokens,
                    )
                raise

    async def _write_artifact(
        self,
        *,
        run_id: str,
        stage: StageDef,
        model_id: str,
        text: str,
    ) -> Artifact:
        out_dir = _stage_dir(run_id, stage.num)
        out_dir.mkdir(parents=True, exist_ok=True)
        path = _artifact_path(run_id, stage.num, model_id, stage.ext)
        path.write_text(text, encoding="utf-8")
        with session_scope() as session:
            artifact = create_artifact(
                session,
                run_id=run_id,
                stage_num=stage.num,
                stage_name=stage.name,
                model_id=model_id,
                kind=stage.kind,
                content_path=path,
                content_type=stage.content_type,
            )
        return artifact

    async def _stage1(
        self,
        run_id: str,
        council_model_ids: list[str],
        report: Report,
        emit: Callable[[dict[str, Any]], Awaitable[None]],
    ) -> None:
        stage = STAGES[0]
        prompt = _load_prompt("stage1_analysis.md")
        report_text_path = Path(report.extracted_text_path)
        report_text = report_text_path.read_text(encoding="utf-8", errors="replace")

        # Derive the on-disk report folder from the stored/extracted paths.
        # Important: some older uploads used a different folder id than report.id, so don't assume
        # REPORTS_DIR/<patient_id>/<report.id>/... exists.
        report_dir = report_text_path.parent
        if not report_dir.exists():
            try:
                report_dir = Path(report.stored_path).parent
            except Exception:
                report_dir = report_text_path.parent

        # Prefer enhanced OCR text if available (captures tables/images better).
        # First try the file adjacent to extracted.txt, then fall back to legacy lookup.
        enhanced_text: str | None = None
        enhanced_path = report_dir / "extracted_enhanced.txt"
        if enhanced_path.exists():
            enhanced_text = enhanced_path.read_text(encoding="utf-8", errors="replace")
        else:
            try:
                enhanced_text = get_enhanced_text(report.patient_id, report.id)
            except Exception:
                enhanced_text = None
        if enhanced_text and len(enhanced_text) > len(report_text):
            report_text = enhanced_text

        # Auto-upgrade older extractions (and OCR image-only pages when possible).
        # If the extracted text lacks per-page markers, regenerate from the original PDF/text.
        if "=== PAGE 1 /" not in report_text and Path(report.stored_path).suffix.lower() == ".pdf":
            try:
                regenerated = extract_text_from_pdf(Path(report.stored_path))
                if regenerated and len(regenerated) > len(report_text):
                    report_text_path.write_text(regenerated, encoding="utf-8")
                    report_text = regenerated
            except Exception:
                pass

        # Get page images for multimodal models.
        # Prefer images stored under the report folder; fall back to legacy lookup; finally, generate on the fly.
        page_images_base64: list[str] = []
        pages_dir = report_dir / "pages"
        if pages_dir.exists():
            page_files = sorted(
                pages_dir.glob("page-*.png"),
                key=lambda p: int(p.stem.split("-")[1]) if "-" in p.stem else 0,
            )
            for p in page_files:
                try:
                    page_images_base64.append(base64.b64encode(p.read_bytes()).decode("utf-8"))
                except Exception:
                    continue
        if not page_images_base64:
            try:
                page_images_base64 = get_page_images_base64(report.patient_id, report.id)
            except Exception:
                page_images_base64 = []
        if not page_images_base64 and any(is_vision_capable(m) for m in council_model_ids):
            try:
                _enh_text, page_images = extract_pdf_with_images(Path(report.stored_path))
                page_images_base64 = [
                    img.get("base64_png")
                    for img in page_images
                    if isinstance(img, dict) and isinstance(img.get("base64_png"), str)
                ]
            except Exception:
                page_images_base64 = []

        prompt_text = f"{prompt}\n\n---\n\nqEEG Report Text:\n\n{report_text}\n"

        await emit({"run_id": run_id, "stage_num": stage.num, "stage_name": stage.name, "status": "start"})

        async def one(model_id: str) -> tuple[str, str] | None:
            try:
                # Use multimodal messages for vision-capable models
                if is_vision_capable(model_id) and page_images_base64:
                    text = await self._call_model_multimodal(
                        model_id=model_id,
                        prompt_text=prompt_text,
                        images_base64=page_images_base64,
                        temperature=0.2,
                        max_tokens=8000,
                    )
                else:
                    text = await self._call_model_chat(
                        model_id=model_id, prompt_text=prompt_text, temperature=0.2, max_tokens=8000
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

        # Get original report text for context propagation
        report_text = ""
        if report and report.extracted_text_path:
            report_text = Path(report.extracted_text_path).read_text(encoding="utf-8", errors="replace")

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
                    max_tokens=2000,
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

        # Get original report text for context propagation
        report_text = ""
        if report and report.extracted_text_path:
            report_text = Path(report.extracted_text_path).read_text(encoding="utf-8", errors="replace")

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
                f"ORIGINAL qEEG REPORT (for fact-checking your revision):\n\n{report_text}\n\n---\n\n"
                f"Your Model ID: {model_id}\n"
                f"Your analysis label (if present): {my_label}\n\n"
                f"Your original analysis:\n\n{analysis}\n\n"
                f"Peer review JSON artifacts:\n\n{pr_text}\n"
            )
            try:
                text = await self._call_model_chat(
                    model_id=model_id, prompt_text=prompt_text, temperature=0.2, max_tokens=6000
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

        with session_scope() as session:
            run = get_run(session, run_id)
            if run is None:
                raise RuntimeError("Run not found")
            consolidator = run.consolidator_model_id
            revisions = _stage_artifacts(session, run_id, 3, kind="revision")
            report = get_report(session, run.report_id) if run else None

        # Get original report text for context propagation
        report_text = ""
        if report and report.extracted_text_path:
            report_text = Path(report.extracted_text_path).read_text(encoding="utf-8", errors="replace")

        if not revisions:
            raise RuntimeError("Consolidation requires at least one revision artifact")

        revision_text = "\n\n".join(
            [
                f"Revision by {a.model_id}:\n{Path(a.content_path).read_text(encoding='utf-8', errors='replace')}"
                for a in revisions
            ]
        )
        prompt_text = (
            f"{prompt}\n\n---\n\n"
            f"ORIGINAL qEEG REPORT (the source of truth - verify all claims against this):\n\n{report_text}\n\n---\n\n"
            f"REVISED ANALYSES TO CONSOLIDATE:\n\n{revision_text}\n"
        )

        await emit({"run_id": run_id, "stage_num": stage.num, "stage_name": stage.name, "status": "start"})
        text = await self._call_model_chat(
            model_id=consolidator, prompt_text=prompt_text, temperature=0.2, max_tokens=8000
        )
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

        # Get original report text for context propagation
        report_text = ""
        if report and report.extracted_text_path:
            report_text = Path(report.extracted_text_path).read_text(encoding="utf-8", errors="replace")

        if not s4:
            raise RuntimeError("Stage 5 requires Stage 4 consolidation artifact")

        consolidated = Path(s4[0].content_path).read_text(encoding="utf-8", errors="replace")

        await emit({"run_id": run_id, "stage_num": stage.num, "stage_name": stage.name, "status": "start"})

        async def one(model_id: str) -> tuple[str, str] | None:
            prompt_text = (
                f"{prompt}\n\n---\n\n"
                f"ORIGINAL qEEG REPORT (for verification):\n\n{report_text}\n\n---\n\n"
                f"CONSOLIDATED REPORT TO REVIEW:\n\n{consolidated}\n"
            )
            try:
                text = await self._call_model_chat(
                    model_id=model_id, prompt_text=prompt_text, temperature=0.1, max_tokens=1500
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

        # Get original report text for context propagation
        report_text = ""
        if report and report.extracted_text_path:
            report_text = Path(report.extracted_text_path).read_text(encoding="utf-8", errors="replace")

        if not s4:
            raise RuntimeError("Stage 6 requires Stage 4 consolidation artifact")
        consolidated = Path(s4[0].content_path).read_text(encoding="utf-8", errors="replace")

        required_changes = _aggregate_required_changes(s5)

        await emit({"run_id": run_id, "stage_num": stage.num, "stage_name": stage.name, "status": "start"})

        async def one(model_id: str) -> tuple[str, str] | None:
            changes = "\n".join([f"- {c}" for c in required_changes]) if required_changes else "(none)"
            prompt_text = (
                f"{prompt}\n\n---\n\n"
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


class _NeedsAuth(RuntimeError):
    pass


def _loads_json_list(text: str) -> list[str]:
    try:
        data = json.loads(text)
    except Exception:
        return []
    if not isinstance(data, list):
        return []
    return [x for x in data if isinstance(x, str)]


def _stage_artifacts(session, run_id: str, stage_num: int, *, kind: str) -> list[Artifact]:
    from sqlalchemy import select

    return list(
        session.scalars(
            select(Artifact)
            .where(Artifact.run_id == run_id, Artifact.stage_num == stage_num, Artifact.kind == kind)
            .order_by(Artifact.created_at.asc())
        )
    )


def _validate_stage5(payload: Any) -> None:
    if not isinstance(payload, dict):
        raise ValueError("Stage 5 payload must be an object")
    vote = payload.get("vote")
    if vote not in {"APPROVE", "REVISE"}:
        raise ValueError("Stage 5 vote must be APPROVE or REVISE")
    for key in ("required_changes", "optional_changes"):
        val = payload.get(key)
        if not isinstance(val, list) or not all(isinstance(x, str) for x in val):
            raise ValueError(f"Stage 5 {key} must be a string[]")
    score = payload.get("quality_score_1to10")
    if not isinstance(score, int) or not (1 <= score <= 10):
        raise ValueError("Stage 5 quality_score_1to10 must be int 1..10")


def _aggregate_required_changes(stage5_artifacts: list[Artifact]) -> list[str]:
    changes: list[str] = []
    seen: set[str] = set()
    for art in stage5_artifacts:
        try:
            payload = json.loads(Path(art.content_path).read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(payload, dict):
            continue
        for c in payload.get("required_changes", []) or []:
            if isinstance(c, str):
                c2 = c.strip()
                if c2 and c2 not in seen:
                    seen.add(c2)
                    changes.append(c2)
    return changes
