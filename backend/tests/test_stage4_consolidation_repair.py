"""Tests for Stage 4 consolidation truncation repair."""

from __future__ import annotations

import asyncio
import uuid
from pathlib import Path

import pytest


def _create_stage4_ready_run(*, report_id: str, consolidator_model_id: str) -> str:
    from backend.config import ARTIFACTS_DIR, REPORTS_DIR
    from backend.storage import (
        create_artifact,
        create_patient,
        create_report,
        create_run,
        session_scope,
    )

    # Minimal report files (Stage 4 needs extracted text).
    # NOTE: patient_id is assigned by DB; use a placeholder folder first and rename after creation.
    tmp_patient_id = str(uuid.uuid4())
    report_dir = REPORTS_DIR / tmp_patient_id / report_id
    report_dir.mkdir(parents=True, exist_ok=True)
    stored_path = report_dir / "original.txt"
    extracted_path = report_dir / "extracted.txt"
    stored_path.write_text("dummy", encoding="utf-8")
    extracted_path.write_text("=== PAGE 1 / 1 ===\nHello\n", encoding="utf-8")

    with session_scope() as session:
        patient = create_patient(session, label="Test", notes="")
        patient_id = patient.id
        create_report(
            session,
            report_id=report_id,
            patient_id=patient_id,
            filename="original.txt",
            mime_type="text/plain",
            stored_path=stored_path,
            extracted_text_path=extracted_path,
        )
        run = create_run(
            session,
            patient_id=patient_id,
            report_id=report_id,
            council_model_ids=["mock-council-a"],
            consolidator_model_id=consolidator_model_id,
        )
        run_id = run.id

        # Create a Stage 3 revision artifact so Stage 4 can run.
        stage3_dir = ARTIFACTS_DIR / run_id / "stage-3"
        stage3_dir.mkdir(parents=True, exist_ok=True)
        rev_path = stage3_dir / "mock-council-a.md"
        rev_path.write_text("Revision content", encoding="utf-8")
        create_artifact(
            session,
            run_id=run_id,
            stage_num=3,
            stage_name="revision",
            model_id="mock-council-a",
            kind="revision",
            content_path=rev_path,
            content_type="text/markdown",
        )

    return run_id


@pytest.mark.asyncio
async def test_stage4_repairs_truncated_consolidation(
    temp_data_dir, mock_llm_client, monkeypatch
):
    from backend.council import QEEGCouncilWorkflow
    from backend.storage import list_artifacts, session_scope

    report_id = str(uuid.uuid4())
    run_id = _create_stage4_ready_run(
        report_id=report_id,
        consolidator_model_id="claude-sonnet-4-6-20260101",
    )

    truncated = (
        "# Dataset and Sessions\nx\n"
        "# Key Empirical Findings\nx\n"
        "# Performance Assessments\nx\n"
        "# Auditory ERP: P300 and N100\nx\n"
        "# Background EEG Metrics\nx\n"
        "# Speculative Commentary and Interpretive Hypotheses\n"
        "This section is cut off mid-sentence"
    )
    tail = (
        "# Speculative Commentary and Interpretive Hypotheses\nok\n"
        "<!-- END CONSOLIDATED REPORT -->\n"
    )

    call_count = {"n": 0}

    async def fake_call_model_chat(
        *, model_id: str, prompt_text: str, temperature: float, max_tokens: int
    ) -> str:
        call_count["n"] += 1
        return truncated if call_count["n"] == 1 else tail

    workflow = QEEGCouncilWorkflow(llm=mock_llm_client)
    monkeypatch.setattr(workflow, "_call_model_chat", fake_call_model_chat)

    async def emit(_payload):
        return None

    await workflow._stage4(run_id, emit)

    with session_scope() as session:
        artifacts = [a for a in list_artifacts(session, run_id) if a.stage_num == 4]
    assert len(artifacts) == 1
    out_path = Path(artifacts[0].content_path)
    out_text = out_path.read_text(encoding="utf-8", errors="replace")

    # Should include the sentinel and the repaired section.
    assert "<!-- END CONSOLIDATED REPORT -->" in out_text
    assert "# Speculative Commentary and Interpretive Hypotheses" in out_text


@pytest.mark.asyncio
async def test_stage4_emits_heartbeat_and_honors_configured_timeout(
    temp_data_dir, mock_llm_client, monkeypatch
):
    from backend.council import QEEGCouncilWorkflow

    model_id = "claude-sonnet-4-6-20260101"
    run_id = _create_stage4_ready_run(
        report_id=str(uuid.uuid4()),
        consolidator_model_id=model_id,
    )

    async def stuck_call_model_chat(
        *,
        model_id: str,
        prompt_text: str,
        temperature: float,
        max_tokens: int,
    ) -> str:
        await asyncio.sleep(10)
        return "unreachable"

    monkeypatch.setenv("QEEG_PROGRESS_HEARTBEAT_S", "1")
    monkeypatch.setenv("QEEG_STAGE4_MODEL_TIMEOUT_S", "2")
    workflow = QEEGCouncilWorkflow(llm=mock_llm_client)
    monkeypatch.setattr(workflow, "_call_model_chat", stuck_call_model_chat)

    events: list[dict[str, object]] = []

    async def emit(payload: dict[str, object]) -> None:
        events.append(payload)

    with pytest.raises(TimeoutError):
        await workflow._stage4(run_id, emit)

    model_events = [event for event in events if event.get("model_id") == model_id]
    assert any(event.get("status") == "start" for event in model_events)
    assert any(event.get("status") == "heartbeat" for event in model_events)


@pytest.mark.asyncio
async def test_stage4_compacts_oversized_longitudinal_context(
    temp_data_dir, mock_llm_client, monkeypatch
):
    from backend.config import ARTIFACTS_DIR
    from backend.council import QEEGCouncilWorkflow
    from backend.council.paths import _data_pack_path, _vision_transcript_path
    from backend.storage import get_report, get_run, session_scope

    run_id = _create_stage4_ready_run(
        report_id=str(uuid.uuid4()),
        consolidator_model_id="z-ai/glm-5.1",
    )

    with session_scope() as session:
        run = get_run(session, run_id)
        assert run is not None
        report = get_report(session, run.report_id)
        assert report is not None
        extracted_path = Path(report.extracted_text_path)

    extracted_path.write_text(
        "\n\n".join(
            [
                "=== PAGE 1 / 6 ===\nAssessment Scores\nPhysical Reaction Time\nKEEP SUMMARY ONE",
                "=== PAGE 2 / 6 ===\nlow value filler " + ("x" * 5000),
                "=== PAGE 3 / 6 ===\nAssessment Scores\nSession 3\nAudio P300 Delay\nKEEP SESSION THREE",
                "=== PAGE 4 / 6 ===\nlow value filler " + ("y" * 5000),
                "=== PAGE 5 / 6 ===\nMoCA\nAudio P300 Voltage\nKEEP SESSION FOUR",
                "=== PAGE 6 / 6 ===\nAppendix text",
            ]
        ),
        encoding="utf-8",
    )

    dp_path = _data_pack_path(run_id)
    dp_path.parent.mkdir(parents=True, exist_ok=True)
    dp_path.write_text(
        '{"all_sessions":[1,2,3,4],"critical_fact":"do not truncate this data pack"}',
        encoding="utf-8",
    )

    vt_path = _vision_transcript_path(run_id)
    vt_path.parent.mkdir(parents=True, exist_ok=True)
    vt_path.write_text(
        "\n\n".join(
            [
                "# Multimodal Vision Transcript",
                "## Page 1\nAssessment Scores\nKEEP VISION SUMMARY",
                "## Page 2\nvision filler " + ("z" * 4000),
                "## Page 3\nMoCA\nAudio P300 Voltage\nKEEP VISION FINAL",
            ]
        ),
        encoding="utf-8",
    )

    revision_path = ARTIFACTS_DIR / run_id / "stage-3" / "mock-council-a.md"
    revision_path.write_text(
        "Revision start\n" + ("revision filler " * 400) + "\nRevision end",
        encoding="utf-8",
    )

    monkeypatch.setenv("QEEG_STAGE4_REPORT_TEXT_CHAR_LIMIT", "2600")
    monkeypatch.setenv("QEEG_STAGE4_VISION_TRANSCRIPT_CHAR_LIMIT", "1400")
    monkeypatch.setenv("QEEG_STAGE4_REVISION_CHAR_LIMIT_PER_MODEL", "1200")
    seen_prompts: list[str] = []

    async def fake_call_model_chat(
        *, model_id: str, prompt_text: str, temperature: float, max_tokens: int
    ) -> str:
        seen_prompts.append(prompt_text)
        return "# Dataset and Sessions\nok\n<!-- END CONSOLIDATED REPORT -->\n"

    workflow = QEEGCouncilWorkflow(llm=mock_llm_client)
    monkeypatch.setattr(workflow, "_call_model_chat", fake_call_model_chat)

    async def emit(_payload):
        return None

    await workflow._stage4(run_id, emit)

    assert len(seen_prompts) == 1
    prompt = seen_prompts[0]
    assert len(prompt) < 18000
    assert "STAGE 4 SOURCE REPORT COMPACTED" in prompt
    assert "STAGE 4 VISION TRANSCRIPT COMPACTED" in prompt
    assert "do not truncate this data pack" in prompt
    assert "KEEP SUMMARY ONE" in prompt
    assert "KEEP SESSION THREE" in prompt
    assert "KEEP SESSION FOUR" in prompt
    assert "KEEP VISION SUMMARY" in prompt
    assert "KEEP VISION FINAL" in prompt
    assert "Revision start" in prompt
    assert "Revision end" in prompt
    assert "low value filler" not in prompt


@pytest.mark.asyncio
async def test_stage5_compacts_oversized_longitudinal_context(
    temp_data_dir, mock_llm_client, monkeypatch
):
    from backend.config import ARTIFACTS_DIR
    from backend.council import QEEGCouncilWorkflow
    from backend.council.paths import _data_pack_path, _vision_transcript_path
    from backend.storage import create_artifact, get_report, get_run, session_scope

    run_id = _create_stage4_ready_run(
        report_id=str(uuid.uuid4()),
        consolidator_model_id="z-ai/glm-5.1",
    )

    with session_scope() as session:
        run = get_run(session, run_id)
        assert run is not None
        report = get_report(session, run.report_id)
        assert report is not None
        extracted_path = Path(report.extracted_text_path)

        stage4_dir = ARTIFACTS_DIR / run_id / "stage-4"
        stage4_dir.mkdir(parents=True, exist_ok=True)
        s4_path = stage4_dir / "z-ai__glm-5.1.md"
        s4_path.write_text("Consolidated content", encoding="utf-8")
        create_artifact(
            session,
            run_id=run_id,
            stage_num=4,
            stage_name="consolidation",
            model_id="z-ai/glm-5.1",
            kind="consolidation",
            content_path=s4_path,
            content_type="text/markdown",
        )

    extracted_path.write_text(
        "\n\n".join(
            [
                "=== PAGE 1 / 4 ===\nAssessment Scores\nKEEP STAGE5 SUMMARY",
                "=== PAGE 2 / 4 ===\nstage5 filler " + ("x" * 5000),
                "=== PAGE 3 / 4 ===\nMoCA\nAudio P300 Delay\nKEEP STAGE5 FINAL",
                "=== PAGE 4 / 4 ===\nAppendix",
            ]
        ),
        encoding="utf-8",
    )
    dp_path = _data_pack_path(run_id)
    dp_path.parent.mkdir(parents=True, exist_ok=True)
    dp_path.write_text(
        '{"critical_fact":"stage5 keeps data pack"}',
        encoding="utf-8",
    )
    vt_path = _vision_transcript_path(run_id)
    vt_path.parent.mkdir(parents=True, exist_ok=True)
    vt_path.write_text(
        "## Page 1\nAssessment Scores\nKEEP STAGE5 VISION\n\n"
        + "## Page 2\nvision filler "
        + ("z" * 4000),
        encoding="utf-8",
    )

    monkeypatch.setenv("QEEG_STAGE5_REPORT_TEXT_CHAR_LIMIT", "1800")
    monkeypatch.setenv("QEEG_STAGE5_VISION_TRANSCRIPT_CHAR_LIMIT", "1000")
    seen_prompts: list[str] = []

    async def fake_stage5_json(*, llm_client, model_id, prompt_text, model_override=None):
        seen_prompts.append(prompt_text)
        return (
            '{"vote":"APPROVE","required_changes":[],"optional_changes":[],'
            '"quality_score_1to10":8}'
        )

    monkeypatch.setattr(
        "backend.council.workflow.stages.run_stage5_final_review_json",
        fake_stage5_json,
    )

    workflow = QEEGCouncilWorkflow(llm=mock_llm_client)

    async def emit(_payload):
        return None

    await workflow._stage5(run_id, ["deepseek-v4-pro"], emit)

    assert len(seen_prompts) == 1
    prompt = seen_prompts[0]
    assert "STAGE 4 SOURCE REPORT COMPACTED" in prompt
    assert "STAGE 4 VISION TRANSCRIPT COMPACTED" in prompt
    assert "stage5 keeps data pack" in prompt
    assert "KEEP STAGE5 SUMMARY" in prompt
    assert "KEEP STAGE5 FINAL" in prompt
    assert "KEEP STAGE5 VISION" in prompt
    assert "stage5 filler" not in prompt


@pytest.mark.asyncio
async def test_stage4_repairs_after_multiple_continuation_calls(
    temp_data_dir, mock_llm_client, monkeypatch
):
    from backend.council import QEEGCouncilWorkflow
    from backend.storage import list_artifacts, session_scope

    report_id = str(uuid.uuid4())
    run_id = _create_stage4_ready_run(
        report_id=report_id,
        consolidator_model_id="claude-opus-4-6-20260101",
    )

    initial = (
        "# Dataset and Sessions\nok\n"
        "# Key Empirical Findings\nok\n"
        "# Performance Assessments\nok\n"
    )
    cont_1 = (
        "# Auditory ERP: P300 and N100\nok\n"
        "# Background EEG Metrics\nok\n"
        "# Speculative Commentary and Interpretive Hypotheses\nok\n"
    )
    cont_2 = (
        "# Speculative Commentary and Interpretive Hypotheses\nok\n"
        "<!-- END CONSOLIDATED REPORT -->\n"
    )
    responses = [initial, cont_1, cont_2]

    call_count = {"n": 0}

    async def fake_call_model_chat(
        *, model_id: str, prompt_text: str, temperature: float, max_tokens: int
    ) -> str:
        idx = call_count["n"]
        call_count["n"] += 1
        return responses[idx] if idx < len(responses) else responses[-1]

    workflow = QEEGCouncilWorkflow(llm=mock_llm_client)
    monkeypatch.setattr(workflow, "_call_model_chat", fake_call_model_chat)

    async def emit(_payload):
        return None

    await workflow._stage4(run_id, emit)
    assert call_count["n"] == 3

    with session_scope() as session:
        artifacts = [a for a in list_artifacts(session, run_id) if a.stage_num == 4]
    assert len(artifacts) == 1
    out_text = Path(artifacts[0].content_path).read_text(
        encoding="utf-8", errors="replace"
    )
    assert "# Speculative Commentary and Interpretive Hypotheses" in out_text
    assert "<!-- END CONSOLIDATED REPORT -->" in out_text


@pytest.mark.asyncio
async def test_stage4_raises_if_still_incomplete_after_repairs(
    temp_data_dir, mock_llm_client, monkeypatch
):
    from backend.council import QEEGCouncilWorkflow
    from backend.storage import list_artifacts, session_scope

    report_id = str(uuid.uuid4())
    run_id = _create_stage4_ready_run(
        report_id=report_id,
        consolidator_model_id="claude-sonnet-4-6-20260101",
    )

    monkeypatch.setenv("QEEG_STAGE4_REPAIR_CALLS", "1")
    monkeypatch.setenv("QEEG_STAGE4_REQUIRE_COMPLETE", "1")

    async def fake_call_model_chat(
        *, model_id: str, prompt_text: str, temperature: float, max_tokens: int
    ) -> str:
        return "# Dataset and Sessions\ncut off"

    workflow = QEEGCouncilWorkflow(llm=mock_llm_client)
    monkeypatch.setattr(workflow, "_call_model_chat", fake_call_model_chat)

    async def emit(_payload):
        return None

    with pytest.raises(RuntimeError, match="Stage 4 consolidation remained incomplete"):
        await workflow._stage4(run_id, emit)

    with session_scope() as session:
        artifacts = [a for a in list_artifacts(session, run_id) if a.stage_num == 4]
    assert artifacts == []
