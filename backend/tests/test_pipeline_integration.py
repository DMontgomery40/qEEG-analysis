"""Integration tests for the full 6-stage qEEG Council pipeline.

These tests run the complete workflow with:
- Real SQLite database (temp)
- Real PDF extraction (from example files)
- Real artifact file creation
- Mocked CLIProxyAPI transport (no real LLM calls)

This is THE KEY TEST that validates the pipeline actually works end-to-end.
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path

import pytest


@pytest.mark.asyncio
async def test_full_6_stage_pipeline(temp_data_dir, mock_llm_client, example_pdf_bytes: bytes):
    """Run complete pipeline with mock CLIProxyAPI, verify all stages produce artifacts."""
    from backend.council import QEEGCouncilWorkflow
    from backend.storage import (
        session_scope,
        create_patient,
        create_report,
        create_run,
        get_run,
        list_artifacts,
    )
    from backend.reports import save_report_upload

    # 1. Create patient
    with session_scope() as session:
        patient = create_patient(session, label="Test Patient", notes="Pipeline integration test")
        patient_id = patient.id

    # 2. Upload real PDF
    report_id = str(uuid.uuid4())
    original_path, extracted_path, mime_type, _preview = save_report_upload(
        patient_id=patient_id,
        report_id=report_id,
        filename="test.pdf",
        provided_mime_type="application/pdf",
        file_bytes=example_pdf_bytes,
    )

    # 3. Create Report record
    with session_scope() as session:
        report = create_report(
            session,
            report_id=report_id,
            patient_id=patient_id,
            filename="test.pdf",
            mime_type=mime_type,
            stored_path=original_path,
            extracted_text_path=extracted_path,
        )

    # 4. Create run with mock model IDs (non-vision to avoid multimodal path)
    with session_scope() as session:
        run = create_run(
            session,
            patient_id=patient_id,
            report_id=report_id,
            council_model_ids=["mock-council-a", "mock-council-b"],
            consolidator_model_id="mock-consolidator",
        )
        run_id = run.id

    # 5. Execute pipeline
    workflow = QEEGCouncilWorkflow(llm=mock_llm_client)

    # Collect events for verification
    events = []

    async def on_event(payload):
        events.append(payload)

    await workflow.run_pipeline(run_id, on_event=on_event)

    # 6. Verify run completed successfully
    with session_scope() as session:
        run = get_run(session, run_id)
        assert run is not None, "Run should exist"
        assert run.status == "complete", f"Run should be complete, got: {run.status}. Error: {run.error_message}"

    # 7. Verify all 6 stages have artifacts
    with session_scope() as session:
        artifacts = list_artifacts(session, run_id)
        stages_with_artifacts = {a.stage_num for a in artifacts}
        assert stages_with_artifacts == {1, 2, 3, 4, 5, 6}, (
            f"Missing stages: {set(range(1, 7)) - stages_with_artifacts}"
        )

    # 8. Verify artifact files exist
    for artifact in artifacts:
        content_path = Path(artifact.content_path)
        assert content_path.exists(), f"Artifact file should exist: {content_path}"
        content = content_path.read_text()
        assert len(content) > 100, f"Artifact should have content: {artifact.stage_name}"


@pytest.mark.asyncio
async def test_stage2_produces_valid_json(temp_data_dir, mock_llm_client, example_pdf_bytes: bytes):
    """Stage 2 artifacts should contain valid JSON with required keys."""
    from backend.council import QEEGCouncilWorkflow
    from backend.storage import (
        session_scope,
        create_patient,
        create_report,
        create_run,
        list_artifacts,
    )
    from backend.reports import save_report_upload

    # Setup
    with session_scope() as session:
        patient = create_patient(session, label="Test", notes="")
        patient_id = patient.id

    report_id = str(uuid.uuid4())
    original_path, extracted_path, mime_type, _ = save_report_upload(
        patient_id=patient_id,
        report_id=report_id,
        filename="test.pdf",
        provided_mime_type="application/pdf",
        file_bytes=example_pdf_bytes,
    )

    with session_scope() as session:
        create_report(
            session,
            report_id=report_id,
            patient_id=patient_id,
            filename="test.pdf",
            mime_type=mime_type,
            stored_path=original_path,
            extracted_text_path=extracted_path,
        )
        run = create_run(
            session,
            patient_id=patient_id,
            report_id=report_id,
            council_model_ids=["mock-council-a", "mock-council-b"],
            consolidator_model_id="mock-consolidator",
        )
        run_id = run.id

    # Execute
    workflow = QEEGCouncilWorkflow(llm=mock_llm_client)
    await workflow.run_pipeline(run_id)

    # Verify Stage 2 JSON structure
    with session_scope() as session:
        artifacts = list_artifacts(session, run_id)
        stage2_artifacts = [a for a in artifacts if a.stage_num == 2]

        assert len(stage2_artifacts) >= 1, "Should have Stage 2 artifacts"

        for artifact in stage2_artifacts:
            content_path = Path(artifact.content_path)
            content = json.loads(content_path.read_text())

            # Verify required keys (from stage2_peer_review.md prompt)
            assert "reviews" in content, "Stage 2 should have 'reviews' key"
            assert "ranking_best_to_worst" in content, "Stage 2 should have 'ranking_best_to_worst'"
            assert "overall_notes" in content, "Stage 2 should have 'overall_notes'"

            # Verify reviews structure
            assert isinstance(content["reviews"], list), "reviews should be a list"
            for review in content["reviews"]:
                assert "analysis_label" in review, "Each review should have 'analysis_label'"


@pytest.mark.asyncio
async def test_stage5_produces_valid_json(temp_data_dir, mock_llm_client, example_pdf_bytes: bytes):
    """Stage 5 artifacts should contain valid JSON matching _validate_stage5 schema."""
    from backend.council import QEEGCouncilWorkflow
    from backend.storage import (
        session_scope,
        create_patient,
        create_report,
        create_run,
        list_artifacts,
    )
    from backend.reports import save_report_upload

    # Setup
    with session_scope() as session:
        patient = create_patient(session, label="Test", notes="")
        patient_id = patient.id

    report_id = str(uuid.uuid4())
    original_path, extracted_path, mime_type, _ = save_report_upload(
        patient_id=patient_id,
        report_id=report_id,
        filename="test.pdf",
        provided_mime_type="application/pdf",
        file_bytes=example_pdf_bytes,
    )

    with session_scope() as session:
        create_report(
            session,
            report_id=report_id,
            patient_id=patient_id,
            filename="test.pdf",
            mime_type=mime_type,
            stored_path=original_path,
            extracted_text_path=extracted_path,
        )
        run = create_run(
            session,
            patient_id=patient_id,
            report_id=report_id,
            council_model_ids=["mock-council-a", "mock-council-b"],
            consolidator_model_id="mock-consolidator",
        )
        run_id = run.id

    # Execute
    workflow = QEEGCouncilWorkflow(llm=mock_llm_client)
    await workflow.run_pipeline(run_id)

    # Verify Stage 5 JSON structure (this is validated by _validate_stage5)
    with session_scope() as session:
        artifacts = list_artifacts(session, run_id)
        stage5_artifacts = [a for a in artifacts if a.stage_num == 5]

        assert len(stage5_artifacts) >= 1, "Should have Stage 5 artifacts"

        for artifact in stage5_artifacts:
            content_path = Path(artifact.content_path)
            content = json.loads(content_path.read_text())

            # Verify required keys (from _validate_stage5)
            assert content["vote"] in ("APPROVE", "REVISE"), f"vote should be APPROVE or REVISE, got {content['vote']}"
            assert isinstance(content["required_changes"], list), "required_changes should be a list"
            assert isinstance(content["optional_changes"], list), "optional_changes should be a list"
            assert isinstance(content["quality_score_1to10"], int), "quality_score should be int"
            assert 1 <= content["quality_score_1to10"] <= 10, "quality_score should be 1-10"


@pytest.mark.asyncio
async def test_pipeline_emits_completion_event(temp_data_dir, mock_llm_client, example_pdf_bytes: bytes):
    """Pipeline should emit a completion event when finished."""
    from backend.council import QEEGCouncilWorkflow
    from backend.storage import (
        session_scope,
        create_patient,
        create_report,
        create_run,
    )
    from backend.reports import save_report_upload

    # Setup
    with session_scope() as session:
        patient = create_patient(session, label="Test", notes="")
        patient_id = patient.id

    report_id = str(uuid.uuid4())
    original_path, extracted_path, mime_type, _ = save_report_upload(
        patient_id=patient_id,
        report_id=report_id,
        filename="test.pdf",
        provided_mime_type="application/pdf",
        file_bytes=example_pdf_bytes,
    )

    with session_scope() as session:
        create_report(
            session,
            report_id=report_id,
            patient_id=patient_id,
            filename="test.pdf",
            mime_type=mime_type,
            stored_path=original_path,
            extracted_text_path=extracted_path,
        )
        run = create_run(
            session,
            patient_id=patient_id,
            report_id=report_id,
            council_model_ids=["mock-council-a", "mock-council-b"],
            consolidator_model_id="mock-consolidator",
        )
        run_id = run.id

    # Collect events
    events = []

    async def on_event(payload):
        events.append(payload)

    # Execute
    workflow = QEEGCouncilWorkflow(llm=mock_llm_client)
    await workflow.run_pipeline(run_id, on_event=on_event)

    # Verify events were emitted
    assert len(events) > 0, "Should have emitted events"

    # Check for final completion event (run status complete)
    # Events can have status at run level or stage level
    run_complete_events = [
        e for e in events
        if e.get("run_id") == run_id and e.get("status") == "complete" and "stage_num" not in e
    ]
    assert len(run_complete_events) >= 1, f"Should have run completion event. Events: {events}"


@pytest.mark.asyncio
async def test_pipeline_creates_artifact_files_in_correct_locations(
    temp_data_dir, mock_llm_client, example_pdf_bytes: bytes
):
    """Artifacts should be created in the correct directory structure."""
    from backend.council import QEEGCouncilWorkflow
    from backend.config import ARTIFACTS_DIR
    from backend.storage import (
        session_scope,
        create_patient,
        create_report,
        create_run,
        list_artifacts,
    )
    from backend.reports import save_report_upload

    # Setup
    with session_scope() as session:
        patient = create_patient(session, label="Test", notes="")
        patient_id = patient.id

    report_id = str(uuid.uuid4())
    original_path, extracted_path, mime_type, _ = save_report_upload(
        patient_id=patient_id,
        report_id=report_id,
        filename="test.pdf",
        provided_mime_type="application/pdf",
        file_bytes=example_pdf_bytes,
    )

    with session_scope() as session:
        create_report(
            session,
            report_id=report_id,
            patient_id=patient_id,
            filename="test.pdf",
            mime_type=mime_type,
            stored_path=original_path,
            extracted_text_path=extracted_path,
        )
        run = create_run(
            session,
            patient_id=patient_id,
            report_id=report_id,
            council_model_ids=["mock-council-a", "mock-council-b"],
            consolidator_model_id="mock-consolidator",
        )
        run_id = run.id

    # Execute
    workflow = QEEGCouncilWorkflow(llm=mock_llm_client)
    await workflow.run_pipeline(run_id)

    # Verify artifact directory structure
    run_dir = ARTIFACTS_DIR / run_id
    assert run_dir.exists(), f"Run artifact directory should exist: {run_dir}"

    # Check stage directories
    for stage_num in range(1, 7):
        stage_dir = run_dir / f"stage-{stage_num}"
        assert stage_dir.exists(), f"Stage {stage_num} directory should exist"

    # Verify all artifact paths point to files inside the run directory
    with session_scope() as session:
        artifacts = list_artifacts(session, run_id)
        for artifact in artifacts:
            content_path = Path(artifact.content_path)
            assert str(ARTIFACTS_DIR) in str(content_path), "Artifact should be in ARTIFACTS_DIR"
            assert run_id in str(content_path), "Artifact path should include run_id"


@pytest.mark.asyncio
async def test_pipeline_with_minimal_council(temp_data_dir, mock_llm_client, example_pdf_bytes: bytes):
    """Pipeline should work with minimum 2 council models."""
    from backend.council import QEEGCouncilWorkflow
    from backend.storage import (
        session_scope,
        create_patient,
        create_report,
        create_run,
        get_run,
    )
    from backend.reports import save_report_upload

    # Setup with only 2 council models
    with session_scope() as session:
        patient = create_patient(session, label="Test", notes="")
        patient_id = patient.id

    report_id = str(uuid.uuid4())
    original_path, extracted_path, mime_type, _ = save_report_upload(
        patient_id=patient_id,
        report_id=report_id,
        filename="test.pdf",
        provided_mime_type="application/pdf",
        file_bytes=example_pdf_bytes,
    )

    with session_scope() as session:
        create_report(
            session,
            report_id=report_id,
            patient_id=patient_id,
            filename="test.pdf",
            mime_type=mime_type,
            stored_path=original_path,
            extracted_text_path=extracted_path,
        )
        run = create_run(
            session,
            patient_id=patient_id,
            report_id=report_id,
            # Minimal council: exactly 2 models
            council_model_ids=["mock-council-a", "mock-council-b"],
            consolidator_model_id="mock-council-a",  # Consolidator can be one of the council
        )
        run_id = run.id

    # Execute
    workflow = QEEGCouncilWorkflow(llm=mock_llm_client)
    await workflow.run_pipeline(run_id)

    # Verify completion
    with session_scope() as session:
        run = get_run(session, run_id)
        assert run.status == "complete", f"Run should complete with 2 models, got: {run.status}"
