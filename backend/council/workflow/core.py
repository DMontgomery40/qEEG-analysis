from __future__ import annotations

from typing import Any

from ...llm_client import AsyncOpenAICompatClient
from ...storage import get_report, get_run, update_run_status
from ...storage import session_scope
from ..json_utils import _loads_json_list
from ..types import OnEvent
from .data_pack import _DataPackMixin
from .exceptions import _NeedsAuth
from .llm_calls import _LLMCallsMixin
from .stages import _StagesMixin


class QEEGCouncilWorkflow(_DataPackMixin, _LLMCallsMixin, _StagesMixin):
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


