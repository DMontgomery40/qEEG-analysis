"""Tests for Stage 4 consolidation truncation repair."""

from __future__ import annotations

import uuid
from pathlib import Path

import pytest


@pytest.mark.asyncio
async def test_stage4_repairs_truncated_consolidation(temp_data_dir, mock_llm_client, monkeypatch):
    from backend.config import ARTIFACTS_DIR, REPORTS_DIR
    from backend.council import QEEGCouncilWorkflow
    from backend.storage import (
        create_artifact,
        create_patient,
        create_report,
        create_run,
        list_artifacts,
        session_scope,
    )

    patient_id = None
    report_id = str(uuid.uuid4())
    run_id = None

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
            consolidator_model_id="claude-sonnet-4-5-20250929",
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
        "# Measurement Recommendations\nok\n"
        "# Uncertainties and Limits\nok\n"
        "<!-- END CONSOLIDATED REPORT -->\n"
    )

    call_count = {"n": 0}

    async def fake_call_model_chat(*, model_id: str, prompt_text: str, temperature: float, max_tokens: int) -> str:
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

    # Should include the sentinel and the missing sections.
    assert "<!-- END CONSOLIDATED REPORT -->" in out_text
    assert "# Measurement Recommendations" in out_text
    assert "# Uncertainties and Limits" in out_text
