"""Tests for Stage 6 final draft truncation repair."""

from __future__ import annotations

import json
import uuid
from pathlib import Path

import pytest


def _complete_stage6(body: str = "ok") -> str:
    return "\n".join(
        [
            "# Dataset and Sessions",
            body,
            "# Key Empirical Findings",
            body,
            "# Performance Assessments",
            body,
            "# Auditory ERP: P300 and N100",
            body,
            "# Background EEG Metrics",
            body,
            "# Speculative Commentary and Interpretive Hypotheses",
            body,
            "<!-- END STAGE6 FINAL DRAFT -->",
            "",
        ]
    )


def _create_stage6_ready_run(*, report_id: str, council_model_id: str) -> str:
    from backend.config import ARTIFACTS_DIR, REPORTS_DIR
    from backend.storage import (
        create_artifact,
        create_patient,
        create_report,
        create_run,
        session_scope,
    )

    # Minimal report files (Stage 6 needs extracted text).
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
            council_model_ids=[council_model_id],
            consolidator_model_id=council_model_id,
        )
        run_id = run.id

        # Create Stage 4 consolidation artifact so Stage 6 can run.
        stage4_dir = ARTIFACTS_DIR / run_id / "stage-4"
        stage4_dir.mkdir(parents=True, exist_ok=True)
        s4_path = stage4_dir / "consolidated.md"
        s4_path.write_text("Consolidated content", encoding="utf-8")
        create_artifact(
            session,
            run_id=run_id,
            stage_num=4,
            stage_name="consolidation",
            model_id=council_model_id,
            kind="consolidation",
            content_path=s4_path,
            content_type="text/markdown",
        )

        # Create Stage 5 final review artifact with required changes.
        stage5_dir = ARTIFACTS_DIR / run_id / "stage-5"
        stage5_dir.mkdir(parents=True, exist_ok=True)
        s5_path = stage5_dir / "review.json"
        s5_path.write_text(
            json.dumps(
                {
                    "vote": "REVISE",
                    "required_changes": ["Clarify Session 3 trend summary."],
                    "optional_changes": [],
                    "quality_score_1to10": 8,
                }
            ),
            encoding="utf-8",
        )
        create_artifact(
            session,
            run_id=run_id,
            stage_num=5,
            stage_name="final_review",
            model_id=council_model_id,
            kind="final_review",
            content_path=s5_path,
            content_type="application/json",
        )

    return run_id


@pytest.mark.asyncio
async def test_stage6_repairs_truncated_final_draft(temp_data_dir, mock_llm_client, monkeypatch):
    from backend.council import QEEGCouncilWorkflow
    from backend.storage import list_artifacts, session_scope

    model_id = "claude-sonnet-4-6-20260101"
    report_id = str(uuid.uuid4())
    run_id = _create_stage6_ready_run(report_id=report_id, council_model_id=model_id)

    truncated = (
        "# Dataset and Sessions\nok\n"
        "# Key Empirical Findings\nok\n"
        "# Performance Assessments\nok\n"
        "# Auditory ERP: P300 and N100\nok\n"
        "# Background EEG Metrics\ncut off"
    )
    tail = (
        "# Background EEG Metrics\nok\n"
        "# Speculative Commentary and Interpretive Hypotheses\nok\n"
        "<!-- END STAGE6 FINAL DRAFT -->\n"
    )

    call_count = {"n": 0}

    async def fake_call_model_chat(*, model_id: str, prompt_text: str, temperature: float, max_tokens: int) -> str:
        call_count["n"] += 1
        return truncated if call_count["n"] == 1 else tail

    workflow = QEEGCouncilWorkflow(llm=mock_llm_client)
    monkeypatch.setattr(workflow, "_call_model_chat", fake_call_model_chat)

    async def emit(_payload):
        return None

    await workflow._stage6(run_id, [model_id], emit)
    assert call_count["n"] == 2

    with session_scope() as session:
        artifacts = [a for a in list_artifacts(session, run_id) if a.stage_num == 6]
    assert len(artifacts) == 1
    out_text = Path(artifacts[0].content_path).read_text(encoding="utf-8", errors="replace")

    assert "# Speculative Commentary and Interpretive Hypotheses" in out_text
    assert "<!-- END STAGE6 FINAL DRAFT -->" in out_text


@pytest.mark.asyncio
async def test_stage6_compacts_oversized_longitudinal_context(
    temp_data_dir, mock_llm_client, monkeypatch
):
    from backend.council import QEEGCouncilWorkflow
    from backend.council.paths import _data_pack_path, _vision_transcript_path
    from backend.storage import get_report, get_run, session_scope

    model_id = "deepseek-v4-pro"
    run_id = _create_stage6_ready_run(
        report_id=str(uuid.uuid4()),
        council_model_id=model_id,
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
                "=== PAGE 1 / 4 ===\nAssessment Scores\nKEEP STAGE6 SUMMARY",
                "=== PAGE 2 / 4 ===\nstage6 filler " + ("x" * 5000),
                "=== PAGE 3 / 4 ===\nMoCA\nAudio P300 Delay\nKEEP STAGE6 FINAL",
                "=== PAGE 4 / 4 ===\nAppendix",
            ]
        ),
        encoding="utf-8",
    )
    dp_path = _data_pack_path(run_id)
    dp_path.parent.mkdir(parents=True, exist_ok=True)
    dp_path.write_text(
        '{"critical_fact":"stage6 keeps data pack"}',
        encoding="utf-8",
    )
    vt_path = _vision_transcript_path(run_id)
    vt_path.parent.mkdir(parents=True, exist_ok=True)
    vt_path.write_text(
        "## Page 1\nAssessment Scores\nKEEP STAGE6 VISION\n\n"
        + "## Page 2\nvision filler "
        + ("z" * 4000),
        encoding="utf-8",
    )

    monkeypatch.setenv("QEEG_STAGE6_REPORT_TEXT_CHAR_LIMIT", "1800")
    monkeypatch.setenv("QEEG_STAGE6_VISION_TRANSCRIPT_CHAR_LIMIT", "1000")
    seen_prompts: list[str] = []

    async def fake_call_model_chat(
        *, model_id: str, prompt_text: str, temperature: float, max_tokens: int
    ) -> str:
        seen_prompts.append(prompt_text)
        return _complete_stage6()

    workflow = QEEGCouncilWorkflow(llm=mock_llm_client)
    monkeypatch.setattr(workflow, "_call_model_chat", fake_call_model_chat)

    async def emit(_payload):
        return None

    await workflow._stage6(run_id, [model_id], emit)

    assert len(seen_prompts) == 1
    prompt = seen_prompts[0]
    assert "STAGE 4 SOURCE REPORT COMPACTED" in prompt
    assert "STAGE 4 VISION TRANSCRIPT COMPACTED" in prompt
    assert "stage6 keeps data pack" in prompt
    assert "KEEP STAGE6 SUMMARY" in prompt
    assert "KEEP STAGE6 FINAL" in prompt
    assert "KEEP STAGE6 VISION" in prompt
    assert "stage6 filler" not in prompt


@pytest.mark.asyncio
async def test_stage6_routes_only_glm52_writer(
    temp_data_dir, mock_llm_client, monkeypatch
):
    from backend.council import QEEGCouncilWorkflow
    from backend.storage import list_artifacts, session_scope

    council_model = "deepseek-v4-pro"
    run_id = _create_stage6_ready_run(
        report_id=str(uuid.uuid4()),
        council_model_id=council_model,
    )
    called_models: list[str] = []

    async def fake_call_model_chat(
        *, model_id: str, prompt_text: str, temperature: float, max_tokens: int
    ) -> str:
        called_models.append(model_id)
        return _complete_stage6()

    workflow = QEEGCouncilWorkflow(llm=mock_llm_client)
    monkeypatch.setattr(workflow, "_call_model_chat", fake_call_model_chat)

    async def emit(_payload):
        return None

    await workflow._stage6(run_id, [council_model], emit)

    assert called_models == ["z-ai/glm-5.2"]
    with session_scope() as session:
        stage6 = [a for a in list_artifacts(session, run_id) if a.stage_num == 6]
    assert [artifact.model_id for artifact in stage6] == ["z-ai/glm-5.2"]


@pytest.mark.asyncio
async def test_stage6_refuses_non_writer_fallback_when_writer_is_undiscovered(
    temp_data_dir, mock_llm_client, monkeypatch
):
    import backend.council.workflow.stages as stages_module
    from backend.council import QEEGCouncilWorkflow

    analytical_model = "gpt-5.6-terra"
    run_id = _create_stage6_ready_run(
        report_id=str(uuid.uuid4()),
        council_model_id=analytical_model,
    )
    called_models: list[str] = []
    monkeypatch.setattr(
        stages_module,
        "DISCOVERED_MODEL_IDS",
        {analytical_model},
    )

    async def fake_call_model_chat(
        *, model_id: str, prompt_text: str, temperature: float, max_tokens: int
    ) -> str:
        called_models.append(model_id)
        return _complete_stage6()

    workflow = QEEGCouncilWorkflow(llm=mock_llm_client)
    monkeypatch.setattr(workflow, "_call_model_chat", fake_call_model_chat)

    async def emit(_payload):
        return None

    with pytest.raises(RuntimeError, match="no available final-draft model"):
        await workflow._stage6(run_id, [analytical_model], emit)

    assert called_models == []


@pytest.mark.asyncio
async def test_stage6_falls_back_when_discovered_writer_fails_at_call_time(
    temp_data_dir, mock_llm_client, monkeypatch
):
    import backend.council.workflow.stages as stages_module
    from backend.council import QEEGCouncilWorkflow
    from backend.storage import list_artifacts, session_scope

    preferred_model = "z-ai/glm-5.2"
    fallback_model = "kimi-k3"
    analytical_model = "gpt-5.6-terra"
    run_id = _create_stage6_ready_run(
        report_id=str(uuid.uuid4()),
        council_model_id=analytical_model,
    )
    called_models: list[str] = []
    monkeypatch.setattr(
        stages_module,
        "DISCOVERED_MODEL_IDS",
        {preferred_model, fallback_model},
    )

    async def fake_call_model_chat(
        *, model_id: str, prompt_text: str, temperature: float, max_tokens: int
    ) -> str:
        called_models.append(model_id)
        if model_id == preferred_model:
            raise RuntimeError("provider authentication unavailable")
        return _complete_stage6()

    workflow = QEEGCouncilWorkflow(llm=mock_llm_client)
    monkeypatch.setattr(workflow, "_call_model_chat", fake_call_model_chat)

    async def emit(_payload):
        return None

    await workflow._stage6(run_id, [analytical_model], emit)

    assert called_models == [preferred_model, fallback_model]
    with session_scope() as session:
        stage6 = [a for a in list_artifacts(session, run_id) if a.stage_num == 6]
    assert [artifact.model_id for artifact in stage6] == [fallback_model]


@pytest.mark.asyncio
async def test_stage6_rejects_final_content_missing_required_sections(
    temp_data_dir, mock_llm_client, monkeypatch
):
    from backend.council import QEEGCouncilWorkflow

    run_id = _create_stage6_ready_run(
        report_id=str(uuid.uuid4()),
        council_model_id="deepseek-v4-pro",
    )
    monkeypatch.setenv("QEEG_LONGFORM_REPAIR_CALLS", "0")

    async def fake_call_model_chat(
        *, model_id: str, prompt_text: str, temperature: float, max_tokens: int
    ) -> str:
        return "# Dataset and Sessions\nIncomplete\n<!-- END STAGE6 FINAL DRAFT -->\n"

    workflow = QEEGCouncilWorkflow(llm=mock_llm_client)
    monkeypatch.setattr(workflow, "_call_model_chat", fake_call_model_chat)

    async def emit(_payload):
        return None

    with pytest.raises(RuntimeError, match="All models failed during Stage 6"):
        await workflow._stage6(run_id, ["deepseek-v4-pro"], emit)


@pytest.mark.asyncio
async def test_stage6_completion_counts_drafts_not_the_fallback_chain(
    temp_data_dir, mock_llm_client, monkeypatch
):
    # 2026-08-05: every healthy run reported success 1 / requested 2, because the
    # completion event counted `writer_candidates` — the ordered fallback chain,
    # which the loop abandons after the first success — instead of the single
    # final draft the stage exists to produce. orchestration reads that as
    # "partial council output 1/2" and withholds the patient-facing document, so
    # from 2026-08-02 no analysis could publish. Michelle Rosen-Camp's and Gianna
    # Rutherford's PDFs both blocked on it with a full, valid draft on disk.
    import backend.council.workflow.stages as stages_module
    from backend.council import QEEGCouncilWorkflow
    from backend.storage import list_artifacts, session_scope

    council_model = "deepseek-v4-pro"
    run_id = _create_stage6_ready_run(
        report_id=str(uuid.uuid4()),
        council_model_id=council_model,
    )
    # Discovery is populated in the live app, which is what grows the fallback
    # chain to writer + kimi-k3. The autouse fixture empties it, so without this
    # the chain is one entry long and the regression cannot reproduce.
    monkeypatch.setattr(
        stages_module,
        "DISCOVERED_MODEL_IDS",
        {"z-ai/glm-5.2", "moonshotai/kimi-k3", council_model},
    )

    async def fake_call_model_chat(
        *, model_id: str, prompt_text: str, temperature: float, max_tokens: int
    ) -> str:
        return _complete_stage6()

    workflow = QEEGCouncilWorkflow(llm=mock_llm_client)
    monkeypatch.setattr(workflow, "_call_model_chat", fake_call_model_chat)

    payloads: list[dict] = []

    async def emit(payload):
        payloads.append(payload)

    await workflow._stage6(run_id, [council_model], emit)

    start = [p for p in payloads if p.get("status") == "start"][0]
    assert start["candidate_count"] > 1, "the fallback chain must be live for this test to bite"

    completion = [p for p in payloads if p.get("status") == "complete"][-1]
    assert completion["success_count"] == completion["requested_count"], (
        "a stage 6 that produced its final draft must not report a shortfall: "
        f"{completion['success_count']}/{completion['requested_count']}"
    )
    assert completion["requested_count"] == 1

    # The draft that count describes really is on disk.
    with session_scope() as session:
        stage6 = [a for a in list_artifacts(session, run_id) if a.stage_num == 6]
    assert len(stage6) == 1


@pytest.mark.asyncio
async def test_stage6_success_survives_orchestration_partial_check(
    temp_data_dir, mock_llm_client, monkeypatch
):
    # The counts above only matter because orchestration turns a shortfall into a
    # delivery gap. Assert the real consumer, not just the numbers: a completed
    # stage 6 must leave no "partial council output" gap behind.
    import backend.council.workflow.stages as stages_module
    from backend.council import QEEGCouncilWorkflow

    council_model = "deepseek-v4-pro"
    run_id = _create_stage6_ready_run(
        report_id=str(uuid.uuid4()),
        council_model_id=council_model,
    )
    monkeypatch.setattr(
        stages_module,
        "DISCOVERED_MODEL_IDS",
        {"z-ai/glm-5.2", "moonshotai/kimi-k3", council_model},
    )

    async def fake_call_model_chat(
        *, model_id: str, prompt_text: str, temperature: float, max_tokens: int
    ) -> str:
        return _complete_stage6()

    workflow = QEEGCouncilWorkflow(llm=mock_llm_client)
    monkeypatch.setattr(workflow, "_call_model_chat", fake_call_model_chat)

    payloads: list[dict] = []

    async def emit(payload):
        payloads.append(payload)

    await workflow._stage6(run_id, [council_model], emit)

    completion = [p for p in payloads if p.get("status") == "complete"][-1]
    success = completion["success_count"]
    requested = completion["requested_count"]
    partial = (
        isinstance(success, int)
        and isinstance(requested, int)
        and requested > 0
        and success < requested
    )
    assert not partial, "a finished stage 6 must not read as partial council output"
