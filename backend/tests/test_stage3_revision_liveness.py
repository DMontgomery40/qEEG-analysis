"""Tests for Stage 3 revision liveness and configured timeout reporting."""

from __future__ import annotations

import asyncio
import json
import uuid

import pytest


def _create_stage1_run(
    *,
    report_id: str,
    council_model_ids: list[str],
):
    from backend.config import REPORTS_DIR
    from backend.storage import (
        create_patient,
        create_report,
        create_run,
        get_report,
        session_scope,
    )

    tmp_patient_id = str(uuid.uuid4())
    report_dir = REPORTS_DIR / tmp_patient_id / report_id
    report_dir.mkdir(parents=True, exist_ok=True)
    stored_path = report_dir / "original.txt"
    extracted_path = report_dir / "extracted.txt"
    stored_path.write_text("dummy", encoding="utf-8")
    extracted_path.write_text("=== PAGE 1 / 1 ===\nAlpha beta 12.3\n", encoding="utf-8")

    with session_scope() as session:
        patient = create_patient(session, label="12-30-1970-0", notes="")
        report = create_report(
            session,
            report_id=report_id,
            patient_id=patient.id,
            filename="original.txt",
            mime_type="text/plain",
            stored_path=stored_path,
            extracted_text_path=extracted_path,
        )
        run = create_run(
            session,
            patient_id=patient.id,
            report_id=report.id,
            council_model_ids=council_model_ids,
            consolidator_model_id=council_model_ids[0],
        )
        return run.id, get_report(session, report.id)


def _create_stage3_ready_run(
    *,
    report_id: str,
    council_model_ids: list[str],
) -> str:
    from backend.config import ARTIFACTS_DIR, REPORTS_DIR
    from backend.storage import (
        create_artifact,
        create_patient,
        create_report,
        create_run,
        session_scope,
    )

    tmp_patient_id = str(uuid.uuid4())
    report_dir = REPORTS_DIR / tmp_patient_id / report_id
    report_dir.mkdir(parents=True, exist_ok=True)
    stored_path = report_dir / "original.txt"
    extracted_path = report_dir / "extracted.txt"
    stored_path.write_text("dummy", encoding="utf-8")
    extracted_path.write_text("=== PAGE 1 / 1 ===\nHello\n", encoding="utf-8")

    with session_scope() as session:
        patient = create_patient(session, label="12-30-1970-0", notes="")
        create_report(
            session,
            report_id=report_id,
            patient_id=patient.id,
            filename="original.txt",
            mime_type="text/plain",
            stored_path=stored_path,
            extracted_text_path=extracted_path,
        )
        run = create_run(
            session,
            patient_id=patient.id,
            report_id=report_id,
            council_model_ids=council_model_ids,
            consolidator_model_id=council_model_ids[0],
        )
        run_id = run.id

        stage1_dir = ARTIFACTS_DIR / run_id / "stage-1"
        stage2_dir = ARTIFACTS_DIR / run_id / "stage-2"
        stage1_dir.mkdir(parents=True, exist_ok=True)
        stage2_dir.mkdir(parents=True, exist_ok=True)
        for model_id in council_model_ids:
            s1_path = stage1_dir / f"{model_id}.md"
            s1_path.write_text(f"Initial analysis from {model_id}", encoding="utf-8")
            create_artifact(
                session,
                run_id=run_id,
                stage_num=1,
                stage_name="initial_analysis",
                model_id=model_id,
                kind="analysis",
                content_path=s1_path,
                content_type="text/markdown",
            )

            s2_path = stage2_dir / f"{model_id}.json"
            s2_path.write_text(
                json.dumps({"summary": f"Peer review from {model_id}"}),
                encoding="utf-8",
            )
            create_artifact(
                session,
                run_id=run_id,
                stage_num=2,
                stage_name="peer_review",
                model_id=model_id,
                kind="peer_review",
                content_path=s2_path,
                content_type="application/json",
            )

    return run_id


@pytest.mark.asyncio
async def test_await_with_heartbeat_clamps_poll_to_configured_timeout(
    mock_llm_client, monkeypatch
):
    from backend.council import QEEGCouncilWorkflow

    monkeypatch.setenv("QEEG_PROGRESS_HEARTBEAT_S", "30")
    workflow = QEEGCouncilWorkflow(llm=mock_llm_client)
    events: list[dict[str, object]] = []
    loop = asyncio.get_running_loop()
    started_at = loop.time()

    async def emit(payload: dict[str, object]) -> None:
        events.append(payload)

    with pytest.raises(TimeoutError):
        await workflow._await_with_heartbeat(
            asyncio.Event().wait(),
            emit=emit,
            payload={"task": "slow-model"},
            timeout_s=1,
        )

    assert loop.time() - started_at < 2.5
    assert events == []


@pytest.mark.asyncio
async def test_stage3_emits_model_heartbeats_and_honors_configured_timeout(
    temp_data_dir, mock_llm_client, monkeypatch
):
    from backend.config import ARTIFACTS_DIR
    from backend.council import QEEGCouncilWorkflow
    from backend.storage import list_artifacts, session_scope

    model_ids = ["fast-model", "stuck-model"]
    run_id = _create_stage3_ready_run(
        report_id=str(uuid.uuid4()),
        council_model_ids=model_ids,
    )

    async def fake_longform(
        *,
        model_id: str,
        prompt_text: str,
        temperature: float,
        max_tokens: int,
        end_sentinel: str,
        required_headings: list[str] | None = None,
    ) -> str:
        if model_id == "stuck-model":
            await asyncio.sleep(10)
        return f"# Revised\nDone\n{end_sentinel}\n"

    monkeypatch.setenv("QEEG_PROGRESS_HEARTBEAT_S", "1")
    monkeypatch.setenv("QEEG_STAGE3_MODEL_TIMEOUT_S", "2")
    workflow = QEEGCouncilWorkflow(llm=mock_llm_client)
    monkeypatch.setattr(workflow, "_call_longform_chat_with_repairs", fake_longform)

    events: list[dict[str, object]] = []

    async def emit(payload: dict[str, object]) -> None:
        events.append(payload)

    await workflow._stage3(run_id, model_ids, emit)

    stuck_events = [event for event in events if event.get("model_id") == "stuck-model"]
    assert any(event.get("status") == "start" for event in stuck_events)
    assert any(event.get("status") == "heartbeat" for event in stuck_events)
    assert any(event.get("status") == "failed" for event in stuck_events)
    assert any(
        event.get("status") == "complete"
        and event.get("success_count") == 1
        and event.get("requested_count") == 2
        and event.get("partial_success") is True
        for event in events
    )

    with session_scope() as session:
        artifacts = [a for a in list_artifacts(session, run_id) if a.stage_num == 3]
    assert [artifact.model_id for artifact in artifacts] == ["fast-model"]
    assert (ARTIFACTS_DIR / run_id / "stage-3" / "fast-model.md").exists()


@pytest.mark.asyncio
async def test_stage1_retries_same_model_with_smaller_budget_after_upstream_failure(
    temp_data_dir, mock_llm_client, monkeypatch
):
    from backend.council import QEEGCouncilWorkflow
    from backend.storage import list_artifacts, session_scope

    model_id = "gpt-5.5"
    run_id, report = _create_stage1_run(
        report_id=str(uuid.uuid4()),
        council_model_ids=[model_id],
    )
    assert report is not None

    calls: list[int] = []

    async def fake_longform(
        *,
        model_id: str,
        prompt_text: str,
        temperature: float,
        max_tokens: int,
        end_sentinel: str,
        required_headings: list[str] | None = None,
    ) -> str:
        calls.append(max_tokens)
        if len(calls) == 1:
            raise RuntimeError("upstream empty text")
        assert "RETRY CONSTRAINT" in prompt_text
        return f"# Dataset and Sessions\nDone\n{end_sentinel}\n"

    monkeypatch.setenv("QEEG_STAGE1_MAX_TOKENS", "12000")
    monkeypatch.setenv("QEEG_STAGE1_RETRY_MAX_TOKENS", "6000")
    workflow = QEEGCouncilWorkflow(llm=mock_llm_client)
    monkeypatch.setattr(workflow, "_call_longform_chat_with_repairs", fake_longform)

    events: list[dict[str, object]] = []

    async def emit(payload: dict[str, object]) -> None:
        events.append(payload)

    await workflow._stage1(run_id, [model_id], report, emit)

    assert calls == [12000, 6000]
    assert any(
        event.get("status") == "retry" and event.get("model_id") == model_id
        for event in events
    )
    with session_scope() as session:
        artifacts = [a for a in list_artifacts(session, run_id) if a.stage_num == 1]
    assert [artifact.model_id for artifact in artifacts] == [model_id]
