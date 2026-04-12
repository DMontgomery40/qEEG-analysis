from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ...config import ARTIFACTS_DIR
from ...llm_client import AsyncOpenAICompatClient
from ...logging_utils import get_logger, log_context
from ...storage import get_report, get_run, update_run_status
from ...storage import session_scope
from ..json_utils import _loads_json_list
from ..types import OnEvent
from .data_pack import _DataPackMixin
from .exceptions import _NeedsAuth
from .llm_calls import _LLMCallsMixin
from .stages import _StagesMixin


LOGGER = get_logger(__name__)


def _pipeline_failure_operator_hint() -> str:
    return "run_pipeline advances Stages 1-6 sequentially; inspect the first stage after the last completed emit for a missing artifact, malformed JSON, or strict data guard failure."


def _progress_log_paths(run_id: str) -> tuple[Path, Path]:
    progress_dir = ARTIFACTS_DIR / run_id
    return progress_dir / "progress.log", progress_dir / "progress.jsonl"


def _format_progress_payload(payload: dict[str, Any]) -> str:
    parts: list[str] = []
    status = payload.get("status")
    if isinstance(status, str) and status:
        parts.append(f"status={status}")

    stage_num = payload.get("stage_num")
    stage_name = payload.get("stage_name")
    if stage_num is not None or stage_name:
        parts.append(f"stage={stage_num}:{stage_name}")

    for key in ("task", "model_id"):
        value = payload.get(key)
        if isinstance(value, str) and value:
            parts.append(f"{key}={value}")

    if isinstance(payload.get("chunk_index"), int):
        chunk_total = payload.get("chunk_count")
        if isinstance(chunk_total, int) and chunk_total > 0:
            parts.append(f"chunk={payload['chunk_index']}/{chunk_total}")
        else:
            parts.append(f"chunk={payload['chunk_index']}")

    pages = payload.get("pages")
    if isinstance(pages, list) and pages:
        page_str = ",".join(str(page) for page in pages)
        parts.append(f"pages={page_str}")

    for key in ("elapsed_s", "heartbeat_count", "success_count", "requested_count"):
        value = payload.get(key)
        if value is not None:
            parts.append(f"{key}={value}")

    if payload.get("skipped"):
        parts.append("skipped=true")

    for key in ("reason", "error", "operatorHint"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            compact = " ".join(value.strip().split())
            parts.append(f"{key}={compact}")

    timestamp = payload.get("timestamp")
    if not isinstance(timestamp, str) or not timestamp.strip():
        timestamp = datetime.now(timezone.utc).isoformat()

    return f"{timestamp} {' '.join(parts).strip()}".rstrip()


def _append_progress_event(run_id: str, payload: dict[str, Any]) -> None:
    progress_log_path, progress_jsonl_path = _progress_log_paths(run_id)
    progress_log_path.parent.mkdir(parents=True, exist_ok=True)
    payload_to_write = dict(payload)
    payload_to_write.setdefault("timestamp", datetime.now(timezone.utc).isoformat())

    with progress_log_path.open("a", encoding="utf-8") as handle:
        handle.write(_format_progress_payload(payload_to_write) + "\n")
    with progress_jsonl_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload_to_write, sort_keys=True) + "\n")


class QEEGCouncilWorkflow(_DataPackMixin, _LLMCallsMixin, _StagesMixin):
    def __init__(self, *, llm: AsyncOpenAICompatClient):
        self._llm = llm

    async def run_pipeline(self, run_id: str, on_event: OnEvent = None) -> None:
        async def emit(payload: dict[str, Any]) -> None:
            payload = dict(payload)
            payload.setdefault("run_id", run_id)
            _append_progress_event(run_id, payload)
            if on_event is not None:
                await on_event(payload)

        with session_scope() as session:
            run = get_run(session, run_id)
            if run is None:
                LOGGER.warning("pipeline_run_missing", run_id=run_id)
                return
            report = get_report(session, run.report_id)
            if report is None:
                update_run_status(
                    session, run_id, status="failed", error_message="Report not found"
                )
                LOGGER.error("pipeline_report_missing", run_id=run_id)
                return

            council_model_ids = _loads_json_list(run.council_model_ids_json)
            if not council_model_ids:
                update_run_status(
                    session,
                    run_id,
                    status="failed",
                    error_message="No council models selected for run",
                )
                LOGGER.error("pipeline_models_missing", run_id=run_id)
                return

            patient_id = run.patient_id
            report_id = run.report_id
            update_run_status(session, run_id, status="running")

        with log_context(run_id=run_id, patient_id=patient_id, report_id=report_id):
            LOGGER.info("pipeline_started", council_model_count=len(council_model_ids))
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
                    update_run_status(
                        session, run_id, status="needs_auth", error_message=str(e)
                    )
                operator_hint = (
                    "run_pipeline surfaced _NeedsAuth from a model call; refresh CLIProxy login for the provider used by this run before retrying."
                )
                LOGGER.warning(
                    "pipeline_needs_auth",
                    error=str(e),
                    operatorHint=operator_hint,
                )
                await emit(
                    {
                        "run_id": run_id,
                        "status": "needs_auth",
                        "error": str(e),
                        "operatorHint": operator_hint,
                    }
                )
                return
            except Exception as e:
                with session_scope() as session:
                    update_run_status(
                        session, run_id, status="failed", error_message=str(e)
                    )
                operator_hint = _pipeline_failure_operator_hint()
                LOGGER.exception("pipeline_failed", operatorHint=operator_hint)
                await emit(
                    {
                        "run_id": run_id,
                        "status": "failed",
                        "error": str(e),
                        "operatorHint": operator_hint,
                    }
                )
                return

            with session_scope() as session:
                update_run_status(session, run_id, status="complete")
            LOGGER.info("pipeline_completed")
            await emit({"run_id": run_id, "status": "complete"})
