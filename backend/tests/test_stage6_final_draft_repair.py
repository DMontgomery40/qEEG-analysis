"""Tests for Stage 6 final draft truncation repair."""

from __future__ import annotations

import json
import uuid
from pathlib import Path

import pytest


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
        "# Measurement Recommendations\nok\n"
        "# Uncertainties and Limits\nok\n"
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

    assert "# Measurement Recommendations" in out_text
    assert "# Uncertainties and Limits" in out_text
    assert "<!-- END STAGE6 FINAL DRAFT -->" in out_text
