from __future__ import annotations

from datetime import timedelta
import json
from pathlib import Path

from fastapi.testclient import TestClient


def _test_app(temp_data_dir, monkeypatch):
    monkeypatch.setenv("QEEG_MOCK_LLM", "1")
    from backend import main

    monkeypatch.setattr(
        main,
        "_ensure_project_clipr_config",
        lambda: Path(temp_data_dir) / "cliproxyapi.conf",
    )
    monkeypatch.setattr(main, "_sync_home_auth_to_project", lambda: 0)
    return main.app


def test_summarize_run_progress_uses_real_chunk_progress(temp_data_dir):
    from backend import storage
    from backend.orchestration import progress_jsonl_path, summarize_run_progress

    with storage.session_scope() as session:
        patient = storage.create_patient(session, label="AB_03-05-2010", notes="")
        report_dir = Path(temp_data_dir) / "reports" / patient.id / "report-1"
        report_dir.mkdir(parents=True, exist_ok=True)
        stored_path = report_dir / "original.txt"
        extracted_path = report_dir / "extracted.txt"
        stored_path.write_text("dummy", encoding="utf-8")
        extracted_path.write_text("dummy", encoding="utf-8")
        report = storage.create_report(
            session,
            report_id="report-1",
            patient_id=patient.id,
            filename="source.pdf",
            mime_type="application/pdf",
            stored_path=stored_path,
            extracted_text_path=extracted_path,
        )
        run = storage.create_run(
            session,
            patient_id=patient.id,
            report_id=report.id,
            council_model_ids=["gemini-3.1-pro-preview"],
            consolidator_model_id="claude-sonnet-4-6",
        )
        storage.update_run_status(session, run.id, status="running")

    progress_path = progress_jsonl_path(run.id)
    progress_path.parent.mkdir(parents=True, exist_ok=True)
    progress_path.write_text(
        json.dumps(
            {
                "run_id": run.id,
                "stage_num": 1,
                "stage_name": "initial_analysis",
                "task": "data_pack_chunk",
                "model_id": "gemini-3.1-pro-preview",
                "chunk_index": 2,
                "chunk_count": 4,
                "status": "heartbeat",
                "elapsed_s": 90,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    summary = summarize_run_progress(run)

    assert summary["determinate"] is True
    assert summary["percent"] == 8.3
    assert summary["stage_num"] == 1
    assert "chunk 2/4" in summary["phase_label"]


def test_summarize_run_progress_keeps_partial_success_visible_on_complete(
    temp_data_dir,
):
    from backend import storage
    from backend.orchestration import progress_jsonl_path, summarize_run_progress

    with storage.session_scope() as session:
        patient = storage.create_patient(session, label="AB_03-05-2010", notes="")
        report_dir = Path(temp_data_dir) / "reports" / patient.id / "report-partial"
        report_dir.mkdir(parents=True, exist_ok=True)
        stored_path = report_dir / "original.txt"
        extracted_path = report_dir / "extracted.txt"
        stored_path.write_text("dummy", encoding="utf-8")
        extracted_path.write_text("dummy", encoding="utf-8")
        report = storage.create_report(
            session,
            report_id="report-partial",
            patient_id=patient.id,
            filename="source.pdf",
            mime_type="application/pdf",
            stored_path=stored_path,
            extracted_text_path=extracted_path,
        )
        run = storage.create_run(
            session,
            patient_id=patient.id,
            report_id=report.id,
            council_model_ids=["gpt-5.4", "claude-sonnet-4-6"],
            consolidator_model_id="claude-sonnet-4-6",
        )
        storage.update_run_status(session, run.id, status="complete")

    progress_path = progress_jsonl_path(run.id)
    progress_path.parent.mkdir(parents=True, exist_ok=True)
    progress_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "run_id": run.id,
                        "stage_num": 2,
                        "stage_name": "peer_review",
                        "status": "complete",
                        "success_count": 1,
                        "requested_count": 2,
                    }
                ),
                json.dumps({"run_id": run.id, "status": "complete"}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    summary = summarize_run_progress(run)

    assert summary["partial_success"] is True
    assert summary["success_count"] == 1
    assert summary["requested_count"] == 2
    assert "partial 1/2" in summary["phase_label"]


def test_summarize_run_progress_keeps_peer_review_skip_visible_on_later_complete(
    temp_data_dir,
):
    from backend import storage
    from backend.orchestration import progress_jsonl_path, summarize_run_progress

    with storage.session_scope() as session:
        patient = storage.create_patient(session, label="AB_03-05-2010", notes="")
        report_dir = Path(temp_data_dir) / "reports" / patient.id / "report-skipped"
        report_dir.mkdir(parents=True, exist_ok=True)
        stored_path = report_dir / "original.txt"
        extracted_path = report_dir / "extracted.txt"
        stored_path.write_text("dummy", encoding="utf-8")
        extracted_path.write_text("dummy", encoding="utf-8")
        report = storage.create_report(
            session,
            report_id="report-skipped",
            patient_id=patient.id,
            filename="source.pdf",
            mime_type="application/pdf",
            stored_path=stored_path,
            extracted_text_path=extracted_path,
        )
        run = storage.create_run(
            session,
            patient_id=patient.id,
            report_id=report.id,
            council_model_ids=["deepseek-v4-flash", "gpt-5.5", "claude-sonnet-4-6"],
            consolidator_model_id="gpt-5.5",
        )
        storage.update_run_status(session, run.id, status="complete")

    progress_path = progress_jsonl_path(run.id)
    progress_path.parent.mkdir(parents=True, exist_ok=True)
    progress_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "run_id": run.id,
                        "stage_num": 2,
                        "stage_name": "peer_review",
                        "status": "complete",
                        "skipped": True,
                        "reason": "Not enough Stage 1 analyses for peer review",
                    }
                ),
                json.dumps(
                    {
                        "run_id": run.id,
                        "stage_num": 6,
                        "stage_name": "final_draft",
                        "status": "complete",
                        "success_count": 3,
                        "requested_count": 3,
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    summary = summarize_run_progress(run)

    assert summary["peer_review_skipped"] is True
    assert "peer review skipped" in summary["phase_label"]


def test_partial_complete_run_is_not_delivery_complete(temp_data_dir):
    from backend import storage
    from backend.orchestration import (
        build_patient_orchestration_summary,
        progress_jsonl_path,
    )

    with storage.session_scope() as session:
        patient = storage.create_patient(session, label="AB_03-05-2010", notes="")
        report_dir = (
            Path(temp_data_dir) / "reports" / patient.id / "report-partial-delivery"
        )
        report_dir.mkdir(parents=True, exist_ok=True)
        stored_path = report_dir / "original.txt"
        extracted_path = report_dir / "extracted.txt"
        stored_path.write_text("dummy", encoding="utf-8")
        extracted_path.write_text("dummy", encoding="utf-8")
        report = storage.create_report(
            session,
            report_id="report-partial-delivery",
            patient_id=patient.id,
            filename="source.pdf",
            mime_type="application/pdf",
            stored_path=stored_path,
            extracted_text_path=extracted_path,
        )
        run = storage.create_run(
            session,
            patient_id=patient.id,
            report_id=report.id,
            council_model_ids=["deepseek-v4-flash", "gpt-5.5", "claude-sonnet-4-6"],
            consolidator_model_id="gpt-5.5",
        )
        storage.update_run_status(session, run.id, status="complete")

        progress_path = progress_jsonl_path(run.id)
        progress_path.parent.mkdir(parents=True, exist_ok=True)
        progress_path.write_text(
            "\n".join(
                [
                    json.dumps(
                        {
                            "run_id": run.id,
                            "stage_num": 2,
                            "stage_name": "peer_review",
                            "status": "complete",
                            "skipped": True,
                            "reason": "Not enough Stage 1 analyses for peer review",
                        }
                    ),
                    json.dumps(
                        {
                            "run_id": run.id,
                            "status": "complete",
                            "success_count": 2,
                            "requested_count": 3,
                        }
                    ),
                ]
            )
            + "\n",
            encoding="utf-8",
        )

        summary = build_patient_orchestration_summary(session, patient)

    assert summary["state"] == "attention"
    assert summary["status"] == "incomplete"
    assert summary["label"].startswith("Not clinic-ready")


def test_liveness_uses_artifacts_to_mark_incomplete_delivery_run(temp_data_dir):
    from backend import storage
    from backend.orchestration import derive_run_liveness

    with storage.session_scope() as session:
        patient = storage.create_patient(session, label="AB_03-05-2010", notes="")
        report = storage.create_report(
            session,
            report_id="report-no-review-artifacts",
            patient_id=patient.id,
            filename="source.pdf",
            mime_type="application/pdf",
            stored_path=temp_data_dir / "source.pdf",
            extracted_text_path=temp_data_dir / "extracted.txt",
        )
        run = storage.create_run(
            session,
            patient_id=patient.id,
            report_id=report.id,
            council_model_ids=["mock-council-a"],
            consolidator_model_id="mock-consolidator",
        )
        storage.update_run_status(session, run.id, status="complete")
        liveness = derive_run_liveness(run, artifacts=[])

    assert liveness["display_status"] == "incomplete"
    assert "peer review did not complete" in liveness["council_completion_gaps"]
    assert "no revised council artifact" in liveness["council_completion_gaps"]


def test_majority_peer_review_complete_run_is_delivery_eligible(temp_data_dir):
    from backend import storage
    from backend.orchestration import (
        build_patient_orchestration_summary,
        progress_jsonl_path,
        run_council_completion_gaps,
        summarize_run_progress,
    )

    with storage.session_scope() as session:
        patient = storage.create_patient(session, label="AB_03-05-2010", notes="")
        report_dir = Path(temp_data_dir) / "reports" / patient.id / "report-partial-pr"
        report_dir.mkdir(parents=True, exist_ok=True)
        stored_path = report_dir / "original.txt"
        extracted_path = report_dir / "extracted.txt"
        stored_path.write_text("dummy", encoding="utf-8")
        extracted_path.write_text("dummy", encoding="utf-8")
        report = storage.create_report(
            session,
            report_id="report-partial-pr",
            patient_id=patient.id,
            filename="source.pdf",
            mime_type="application/pdf",
            stored_path=stored_path,
            extracted_text_path=extracted_path,
        )
        run = storage.create_run(
            session,
            patient_id=patient.id,
            report_id=report.id,
            council_model_ids=["deepseek-v4-flash", "gpt-5.5", "claude-sonnet-4-6"],
            consolidator_model_id="gpt-5.5",
        )
        storage.update_run_status(session, run.id, status="complete")
        stage2_dir = Path(temp_data_dir) / "artifacts" / run.id / "stage-2"
        stage2_dir.mkdir(parents=True, exist_ok=True)
        for model_id in ["gpt-5.5", "claude-sonnet-4-6"]:
            artifact_path = stage2_dir / f"{model_id}.json"
            artifact_path.write_text('{"summary": "ok"}', encoding="utf-8")
            storage.create_artifact(
                session,
                run_id=run.id,
                stage_num=2,
                stage_name="peer_review",
                model_id=model_id,
                kind="peer_review",
                content_path=artifact_path,
                content_type="application/json",
            )
        stage3_dir = Path(temp_data_dir) / "artifacts" / run.id / "stage-3"
        stage3_dir.mkdir(parents=True, exist_ok=True)
        revision_path = stage3_dir / "gpt-5.5.md"
        revision_path.write_text("# Revision", encoding="utf-8")
        storage.create_artifact(
            session,
            run_id=run.id,
            stage_num=3,
            stage_name="revision",
            model_id="gpt-5.5",
            kind="revision",
            content_path=revision_path,
            content_type="text/markdown",
        )

        progress_path = progress_jsonl_path(run.id)
        progress_path.parent.mkdir(parents=True, exist_ok=True)
        progress_path.write_text(
            "\n".join(
                [
                    json.dumps(
                        {
                            "run_id": run.id,
                            "stage_num": 2,
                            "stage_name": "peer_review",
                            "status": "complete",
                            "success_count": 2,
                            "requested_count": 3,
                        }
                    ),
                    json.dumps(
                        {
                            "run_id": run.id,
                            "status": "complete",
                            "success_count": 3,
                            "requested_count": 3,
                        }
                    ),
                ]
            )
            + "\n",
            encoding="utf-8",
        )

        gaps = run_council_completion_gaps(
            run,
            progress=summarize_run_progress(run),
            artifacts=storage.list_artifacts(session, run.id),
        )
        summary = build_patient_orchestration_summary(session, patient)

    assert gaps == []
    assert summary["liveness"]["display_status"] == "complete"


def test_below_majority_peer_review_complete_run_is_not_delivery_complete(
    temp_data_dir,
):
    from backend import storage
    from backend.orchestration import (
        progress_jsonl_path,
        run_council_completion_gaps,
        summarize_run_progress,
    )

    with storage.session_scope() as session:
        patient = storage.create_patient(session, label="AB_03-05-2010", notes="")
        report_dir = (
            Path(temp_data_dir) / "reports" / patient.id / "report-below-majority-pr"
        )
        report_dir.mkdir(parents=True, exist_ok=True)
        stored_path = report_dir / "original.txt"
        extracted_path = report_dir / "extracted.txt"
        stored_path.write_text("dummy", encoding="utf-8")
        extracted_path.write_text("dummy", encoding="utf-8")
        report = storage.create_report(
            session,
            report_id="report-below-majority-pr",
            patient_id=patient.id,
            filename="source.pdf",
            mime_type="application/pdf",
            stored_path=stored_path,
            extracted_text_path=extracted_path,
        )
        run = storage.create_run(
            session,
            patient_id=patient.id,
            report_id=report.id,
            council_model_ids=["deepseek-v4-flash", "gpt-5.5", "claude-sonnet-4-6"],
            consolidator_model_id="gpt-5.5",
        )
        storage.update_run_status(session, run.id, status="complete")
        stage2_dir = Path(temp_data_dir) / "artifacts" / run.id / "stage-2"
        stage2_dir.mkdir(parents=True, exist_ok=True)
        artifact_path = stage2_dir / "gpt-5.5.json"
        artifact_path.write_text('{"summary": "ok"}', encoding="utf-8")
        storage.create_artifact(
            session,
            run_id=run.id,
            stage_num=2,
            stage_name="peer_review",
            model_id="gpt-5.5",
            kind="peer_review",
            content_path=artifact_path,
            content_type="application/json",
        )
        stage3_dir = Path(temp_data_dir) / "artifacts" / run.id / "stage-3"
        stage3_dir.mkdir(parents=True, exist_ok=True)
        revision_path = stage3_dir / "gpt-5.5.md"
        revision_path.write_text("# Revision", encoding="utf-8")
        storage.create_artifact(
            session,
            run_id=run.id,
            stage_num=3,
            stage_name="revision",
            model_id="gpt-5.5",
            kind="revision",
            content_path=revision_path,
            content_type="text/markdown",
        )

        progress_path = progress_jsonl_path(run.id)
        progress_path.parent.mkdir(parents=True, exist_ok=True)
        progress_path.write_text(
            "\n".join(
                [
                    json.dumps(
                        {
                            "run_id": run.id,
                            "stage_num": 2,
                            "stage_name": "peer_review",
                            "status": "complete",
                            "success_count": 1,
                            "requested_count": 3,
                        }
                    ),
                    json.dumps(
                        {
                            "run_id": run.id,
                            "status": "complete",
                            "success_count": 3,
                            "requested_count": 3,
                        }
                    ),
                ]
            )
            + "\n",
            encoding="utf-8",
        )

        gaps = run_council_completion_gaps(
            run,
            progress=summarize_run_progress(run),
            artifacts=storage.list_artifacts(session, run.id),
        )

    assert "peer review below majority 1/3" in gaps


def test_patient_facing_regeneration_action_requires_stage6_final_draft(
    temp_data_dir,
):
    from backend import storage
    from backend.orchestration import build_patient_orchestration_detail

    with storage.session_scope() as session:
        patient = storage.create_patient(session, label="AB_03-05-2010", notes="")
        report = storage.create_report(
            session,
            report_id="report-patient-facing-action",
            patient_id=patient.id,
            filename="source.pdf",
            mime_type="application/pdf",
            stored_path=temp_data_dir / "source.pdf",
            extracted_text_path=temp_data_dir / "extracted.txt",
        )
        run = storage.create_run(
            session,
            patient_id=patient.id,
            report_id=report.id,
            council_model_ids=["mock-council-a"],
            consolidator_model_id="mock-consolidator",
        )
        storage.update_run_status(session, run.id, status="complete")
        artifact_root = temp_data_dir / "artifacts" / run.id
        peer_path = artifact_root / "stage-2" / "mock-council-a.json"
        revision_path = artifact_root / "stage-3" / "mock-council-a.md"
        for path, text in (
            (peer_path, "{}"),
            (revision_path, "# Revision"),
        ):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="utf-8")
        storage.create_artifact(
            session,
            run_id=run.id,
            stage_num=2,
            stage_name="peer_review",
            model_id="mock-council-a",
            kind="peer_review",
            content_path=peer_path,
            content_type="application/json",
        )
        storage.create_artifact(
            session,
            run_id=run.id,
            stage_num=3,
            stage_name="revision",
            model_id="mock-council-a",
            kind="revision",
            content_path=revision_path,
            content_type="text/markdown",
        )

        missing_final = build_patient_orchestration_detail(session, patient)

        final_path = artifact_root / "stage-6" / "mock-council-a.md"
        final_path.parent.mkdir(parents=True, exist_ok=True)
        final_path.write_text("# Final", encoding="utf-8")
        storage.create_artifact(
            session,
            run_id=run.id,
            stage_num=6,
            stage_name="final_draft",
            model_id="mock-council-a",
            kind="final_draft",
            content_path=final_path,
            content_type="text/markdown",
        )

        delivery_ready = build_patient_orchestration_detail(session, patient)

    missing_action = missing_final["actions"]["regenerate_patient_facing"]
    ready_action = delivery_ready["actions"]["regenerate_patient_facing"]

    assert missing_action["enabled"] is False
    assert "Stage 6 final draft" in missing_action["reason"]
    assert ready_action["enabled"] is True


def test_patient_facing_regeneration_uses_latest_delivery_ready_run(
    temp_data_dir,
    monkeypatch,
):
    from backend import storage

    with storage.session_scope() as session:
        patient = storage.create_patient(session, label="AB_03-05-2010", notes="")
        report = storage.create_report(
            session,
            report_id="report-patient-facing-ready-fallback",
            patient_id=patient.id,
            filename="source.pdf",
            mime_type="application/pdf",
            stored_path=temp_data_dir / "source.pdf",
            extracted_text_path=temp_data_dir / "extracted.txt",
        )
        ready_run = storage.create_run(
            session,
            patient_id=patient.id,
            report_id=report.id,
            council_model_ids=["mock-council-a"],
            consolidator_model_id="mock-consolidator",
        )
        storage.update_run_status(session, ready_run.id, status="complete")
        ready_root = temp_data_dir / "artifacts" / ready_run.id
        for stage_num, stage_name, kind, suffix, text, content_type in (
            (
                2,
                "peer_review",
                "peer_review",
                "peer-review.json",
                "{}",
                "application/json",
            ),
            (
                3,
                "revision",
                "revision",
                "revision.md",
                "# Revision",
                "text/markdown",
            ),
            (
                6,
                "final_draft",
                "final_draft",
                "final-draft.md",
                "# Final",
                "text/markdown",
            ),
        ):
            artifact_path = ready_root / f"stage-{stage_num}" / suffix
            artifact_path.parent.mkdir(parents=True, exist_ok=True)
            artifact_path.write_text(text, encoding="utf-8")
            storage.create_artifact(
                session,
                run_id=ready_run.id,
                stage_num=stage_num,
                stage_name=stage_name,
                model_id="mock-council-a",
                kind=kind,
                content_path=artifact_path,
                content_type=content_type,
            )

        newer_run = storage.create_run(
            session,
            patient_id=patient.id,
            report_id=report.id,
            council_model_ids=["mock-council-a"],
            consolidator_model_id="mock-consolidator",
        )
        storage.update_run_status(session, newer_run.id, status="complete")
        newer_root = temp_data_dir / "artifacts" / newer_run.id
        for stage_num, stage_name, kind, suffix, text, content_type in (
            (
                2,
                "peer_review",
                "peer_review",
                "peer-review.json",
                "{}",
                "application/json",
            ),
            (
                3,
                "revision",
                "revision",
                "revision.md",
                "# Revision",
                "text/markdown",
            ),
        ):
            artifact_path = newer_root / f"stage-{stage_num}" / suffix
            artifact_path.parent.mkdir(parents=True, exist_ok=True)
            artifact_path.write_text(text, encoding="utf-8")
            storage.create_artifact(
                session,
                run_id=newer_run.id,
                stage_num=stage_num,
                stage_name=stage_name,
                model_id="mock-council-a",
                kind=kind,
                content_path=artifact_path,
                content_type=content_type,
            )
        patient_id = patient.id
        ready_run_id = ready_run.id
        newer_run_id = newer_run.id

    from backend.run_runtime import RunRuntime
    from unittest.mock import AsyncMock

    monkeypatch.setattr(RunRuntime, "start", AsyncMock())
    monkeypatch.setenv("QEEG_PORTAL_RAW_SYNC_WATCHER", "0")
    scheduled = {}
    app = _test_app(temp_data_dir, monkeypatch)
    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.post(
            f"/api/patients/{patient_id}/actions/regenerate_patient_facing",
            json={},
        )
        first_scheduled = dict(scheduled)
        scheduled.clear()
        requested_response = client.post(
            f"/api/patients/{patient_id}/actions/regenerate_patient_facing",
            json={"run_id": newer_run_id},
        )

    assert response.status_code == 200
    assert response.json()["run_id"] == ready_run_id
    assert response.json()["run_id"] != newer_run_id
    assert requested_response.status_code == 409
    assert "No delivery-ready complete run" in requested_response.text
    assert first_scheduled == scheduled == {}
    assert response.json()["postprocessing"]["state"] == "pending"
    with storage.session_scope() as session:
        assert (
            session.get(storage.PostObligation, (ready_run_id, "patient_facing"))
            is not None
        )
        assert session.get(storage.Run, ready_run_id).analysis_input_fingerprint == ""


def test_peer_review_artifact_majority_is_checked_without_progress_counts(
    temp_data_dir,
):
    from backend import storage
    from backend.orchestration import run_council_completion_gaps

    with storage.session_scope() as session:
        patient = storage.create_patient(session, label="AB_03-05-2010", notes="")
        report = storage.create_report(
            session,
            report_id="report-artifact-majority-pr",
            patient_id=patient.id,
            filename="source.pdf",
            mime_type="application/pdf",
            stored_path=temp_data_dir / "source.pdf",
            extracted_text_path=temp_data_dir / "extracted.txt",
        )
        run = storage.create_run(
            session,
            patient_id=patient.id,
            report_id=report.id,
            council_model_ids=["m1", "m2", "m3"],
            consolidator_model_id="m1",
        )
        storage.update_run_status(session, run.id, status="complete")
        stage2_dir = Path(temp_data_dir) / "artifacts" / run.id / "stage-2"
        stage2_dir.mkdir(parents=True, exist_ok=True)
        peer_path = stage2_dir / "m1.json"
        peer_path.write_text("{}", encoding="utf-8")
        storage.create_artifact(
            session,
            run_id=run.id,
            stage_num=2,
            stage_name="peer_review",
            model_id="m1",
            kind="peer_review",
            content_path=peer_path,
            content_type="application/json",
        )
        stage3_dir = Path(temp_data_dir) / "artifacts" / run.id / "stage-3"
        stage3_dir.mkdir(parents=True, exist_ok=True)
        revision_path = stage3_dir / "m1.md"
        revision_path.write_text("# Revision", encoding="utf-8")
        storage.create_artifact(
            session,
            run_id=run.id,
            stage_num=3,
            stage_name="revision",
            model_id="m1",
            kind="revision",
            content_path=revision_path,
            content_type="text/markdown",
        )

        gaps = run_council_completion_gaps(
            run,
            progress={},
            artifacts=storage.list_artifacts(session, run.id),
        )
        second_peer_path = stage2_dir / "m2.json"
        second_peer_path.write_text("{}", encoding="utf-8")
        storage.create_artifact(
            session,
            run_id=run.id,
            stage_num=2,
            stage_name="peer_review",
            model_id="m2",
            kind="peer_review",
            content_path=second_peer_path,
            content_type="application/json",
        )
        majority_gaps = run_council_completion_gaps(
            run,
            progress={},
            artifacts=storage.list_artifacts(session, run.id),
        )

    assert "peer review below majority 1/3" in gaps
    assert majority_gaps == []


def test_derive_run_liveness_marks_old_running_heartbeat_stale(
    temp_data_dir, monkeypatch
):
    from backend import storage
    from backend.orchestration import (
        derive_run_liveness,
        progress_jsonl_path,
        summarize_run_progress,
    )

    monkeypatch.setenv("QEEG_RUN_STALE_AFTER_S", "300")

    with storage.session_scope() as session:
        patient = storage.create_patient(session, label="AB_03-05-2010", notes="")
        report_dir = Path(temp_data_dir) / "reports" / patient.id / "report-stale"
        report_dir.mkdir(parents=True, exist_ok=True)
        stored_path = report_dir / "original.txt"
        extracted_path = report_dir / "extracted.txt"
        stored_path.write_text("dummy", encoding="utf-8")
        extracted_path.write_text("dummy", encoding="utf-8")
        report = storage.create_report(
            session,
            report_id="report-stale",
            patient_id=patient.id,
            filename="source.pdf",
            mime_type="application/pdf",
            stored_path=stored_path,
            extracted_text_path=extracted_path,
        )
        run = storage.create_run(
            session,
            patient_id=patient.id,
            report_id=report.id,
            council_model_ids=["gpt-5.4"],
            consolidator_model_id="claude-sonnet-4-6",
        )
        storage.update_run_status(session, run.id, status="running")

    progress_path = progress_jsonl_path(run.id)
    progress_path.parent.mkdir(parents=True, exist_ok=True)
    progress_path.write_text(
        json.dumps(
            {
                "run_id": run.id,
                "stage_num": 1,
                "stage_name": "initial_analysis",
                "task": "data_pack_chunk",
                "status": "heartbeat",
                "timestamp": "2026-04-12T10:00:00Z",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    progress = summarize_run_progress(run)
    liveness = derive_run_liveness(run, progress=progress)

    assert liveness["is_stale"] is True
    assert liveness["is_live"] is False
    assert liveness["display_status"] == "stale"
    assert "last update" in liveness["display_label"]


def test_derive_run_liveness_keeps_fresh_created_runs_blocking_duplicate_work(
    temp_data_dir, monkeypatch
):
    from backend import storage
    from backend.orchestration import derive_run_liveness

    monkeypatch.setenv("QEEG_RUN_STALE_AFTER_S", "300")

    with storage.session_scope() as session:
        patient = storage.create_patient(session, label="AB_03-05-2010", notes="")
        report_dir = Path(temp_data_dir) / "reports" / patient.id / "report-created"
        report_dir.mkdir(parents=True, exist_ok=True)
        stored_path = report_dir / "original.txt"
        extracted_path = report_dir / "extracted.txt"
        stored_path.write_text("dummy", encoding="utf-8")
        extracted_path.write_text("dummy", encoding="utf-8")
        report = storage.create_report(
            session,
            report_id="report-created",
            patient_id=patient.id,
            filename="source.pdf",
            mime_type="application/pdf",
            stored_path=stored_path,
            extracted_text_path=extracted_path,
        )
        run = storage.create_run(
            session,
            patient_id=patient.id,
            report_id=report.id,
            council_model_ids=["gpt-5.4"],
            consolidator_model_id="claude-sonnet-4-6",
        )

    fresh = derive_run_liveness(run, now=run.created_at + timedelta(seconds=60))
    stale = derive_run_liveness(run, now=run.created_at + timedelta(seconds=301))

    assert fresh["is_live"] is False
    assert fresh["blocks_duplicate_work"] is True
    assert fresh["display_status"] == "created"
    assert stale["is_stale"] is True
    assert stale["blocks_duplicate_work"] is False


def test_patient_orchestration_endpoint_reports_pipeline_and_cathode_state(
    temp_data_dir, monkeypatch
):
    from backend import storage

    cathode_root = Path(temp_data_dir) / "cathode_projects"
    monkeypatch.setenv("QEEG_CATHODE_PROJECTS_DIR", str(cathode_root))

    with storage.session_scope() as session:
        patient = storage.create_patient(session, label="AB_03-05-2010", notes="")
        report_dir = Path(temp_data_dir) / "reports" / patient.id / "report-2"
        report_dir.mkdir(parents=True, exist_ok=True)
        stored_path = report_dir / "original.txt"
        extracted_path = report_dir / "extracted.txt"
        stored_path.write_text("dummy", encoding="utf-8")
        extracted_path.write_text("dummy", encoding="utf-8")
        report = storage.create_report(
            session,
            report_id="report-2",
            patient_id=patient.id,
            filename="source.pdf",
            mime_type="application/pdf",
            stored_path=stored_path,
            extracted_text_path=extracted_path,
        )
        run = storage.create_run(
            session,
            patient_id=patient.id,
            report_id=report.id,
            council_model_ids=["claude-sonnet-4-6"],
            consolidator_model_id="claude-sonnet-4-6",
        )
        storage.update_run_status(session, run.id, status="complete")
        peer_path = (
            Path(temp_data_dir)
            / "artifacts"
            / run.id
            / "stage-2"
            / "claude-sonnet-4-6.json"
        )
        peer_path.parent.mkdir(parents=True, exist_ok=True)
        peer_path.write_text("{}", encoding="utf-8")
        storage.create_artifact(
            session,
            run_id=run.id,
            stage_num=2,
            stage_name="peer_review",
            model_id="claude-sonnet-4-6",
            kind="peer_review",
            content_path=peer_path,
            content_type="application/json",
        )
        revision_path = (
            Path(temp_data_dir)
            / "artifacts"
            / run.id
            / "stage-3"
            / "claude-sonnet-4-6.md"
        )
        revision_path.parent.mkdir(parents=True, exist_ok=True)
        revision_path.write_text("# Revision", encoding="utf-8")
        storage.create_artifact(
            session,
            run_id=run.id,
            stage_num=3,
            stage_name="revision",
            model_id="claude-sonnet-4-6",
            kind="revision",
            content_path=revision_path,
            content_type="text/markdown",
        )
        artifact_dir = Path(temp_data_dir) / "artifacts" / run.id / "stage-4"
        artifact_dir.mkdir(parents=True, exist_ok=True)
        artifact_path = artifact_dir / "claude-sonnet-4-6.md"
        artifact_path.write_text("# Consolidation\nok", encoding="utf-8")
        artifact = storage.create_artifact(
            session,
            run_id=run.id,
            stage_num=4,
            stage_name="consolidation",
            model_id="claude-sonnet-4-6",
            kind="consolidation",
            content_path=artifact_path,
            content_type="text/markdown",
        )
        storage.select_artifact(session, run.id, artifact.id)

    portal_dir = Path(temp_data_dir) / "portal_patients" / "AB_03-05-2010"
    portal_dir.mkdir(parents=True, exist_ok=True)
    (
        portal_dir / "AB_03-05-2010__patient-facing__auto-test__2026-04-12.pdf"
    ).write_bytes(b"%PDF-1.4")
    status_dir = Path(temp_data_dir) / "pipeline_jobs"
    status_dir.mkdir(parents=True, exist_ok=True)
    (status_dir / "AB_03-05-2010.json").write_text(
        json.dumps(
            {"patient_id": "AB_03-05-2010", "status": "complete", "note": "all good"}
        ),
        encoding="utf-8",
    )
    cathode_project = cathode_root / "AB_03-05-2010"
    cathode_project.mkdir(parents=True, exist_ok=True)
    (cathode_project / "qeeg_handoff_payload.json").write_text(
        json.dumps({"ready_for_handoff": True}),
        encoding="utf-8",
    )

    app = _test_app(temp_data_dir, monkeypatch)
    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.get(f"/api/patients/{patient.id}/orchestration")

    assert response.status_code == 200
    payload = response.json()
    assert payload["summary"]["portal"]["patient_facing_count"] == 1
    assert payload["pipeline_job"]["status"] == "complete"
    assert payload["cathode"]["handoff_payload_exists"] is True
    assert payload["recommended_cathode_source"]["artifact"]["stage_num"] == 4


def test_patient_orchestration_endpoint_surfaces_stale_running_rows(
    temp_data_dir, monkeypatch
):
    from backend import storage
    from backend.orchestration import progress_jsonl_path

    monkeypatch.setenv("QEEG_RUN_STALE_AFTER_S", "300")

    with storage.session_scope() as session:
        patient = storage.create_patient(session, label="YZ_12-02-1985", notes="")
        report_dir = Path(temp_data_dir) / "reports" / patient.id / "report-stale-ui"
        report_dir.mkdir(parents=True, exist_ok=True)
        stored_path = report_dir / "original.txt"
        extracted_path = report_dir / "extracted.txt"
        stored_path.write_text("dummy", encoding="utf-8")
        extracted_path.write_text("dummy", encoding="utf-8")
        report = storage.create_report(
            session,
            report_id="report-stale-ui",
            patient_id=patient.id,
            filename="LM_autism-TBI_depressn_20tx_Redacted.pdf",
            mime_type="application/pdf",
            stored_path=stored_path,
            extracted_text_path=extracted_path,
        )
        run = storage.create_run(
            session,
            patient_id=patient.id,
            report_id=report.id,
            council_model_ids=["gpt-5.4"],
            consolidator_model_id="claude-sonnet-4-6",
        )
        storage.update_run_status(session, run.id, status="running")

    progress_path = progress_jsonl_path(run.id)
    progress_path.parent.mkdir(parents=True, exist_ok=True)
    progress_path.write_text(
        json.dumps(
            {
                "run_id": run.id,
                "stage_num": 1,
                "stage_name": "initial_analysis",
                "task": "stage1_model",
                "model_id": "gpt-5.4",
                "status": "heartbeat",
                "timestamp": "2026-04-12T10:00:00Z",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    app = _test_app(temp_data_dir, monkeypatch)
    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.get(f"/api/patients/{patient.id}/orchestration")

    assert response.status_code == 200
    payload = response.json()
    assert payload["summary"]["status"] == "stale"
    assert payload["summary"]["state"] == "attention"
    assert payload["reports"][0]["lifecycle"]["council_status"] == "stale"
    assert payload["active_runs"] == []
    assert payload["stale_runs"][0]["display_status"] == "stale"


def test_patient_orchestration_summary_liveness_tracks_the_run_being_summarized(
    temp_data_dir, monkeypatch
):
    from backend import storage

    with storage.session_scope() as session:
        patient = storage.create_patient(session, label="TV_01-18-1991", notes="")
        report_dir = Path(temp_data_dir) / "reports" / patient.id / "report-mixed"
        report_dir.mkdir(parents=True, exist_ok=True)
        stored_path = report_dir / "original.txt"
        extracted_path = report_dir / "extracted.txt"
        stored_path.write_text("dummy", encoding="utf-8")
        extracted_path.write_text("dummy", encoding="utf-8")
        report = storage.create_report(
            session,
            report_id="report-mixed",
            patient_id=patient.id,
            filename="source.pdf",
            mime_type="application/pdf",
            stored_path=stored_path,
            extracted_text_path=extracted_path,
        )
        running_run = storage.create_run(
            session,
            patient_id=patient.id,
            report_id=report.id,
            council_model_ids=["gpt-5.4"],
            consolidator_model_id="claude-sonnet-4-6",
        )
        storage.update_run_status(session, running_run.id, status="running")
        failed_run = storage.create_run(
            session,
            patient_id=patient.id,
            report_id=report.id,
            council_model_ids=["gpt-5.4"],
            consolidator_model_id="claude-sonnet-4-6",
        )
        storage.update_run_status(
            session, failed_run.id, status="failed", error_message="boom"
        )

    app = _test_app(temp_data_dir, monkeypatch)
    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.get("/api/patients")

    assert response.status_code == 200
    payload = response.json()
    match = next(item for item in payload if item["label"] == "TV_01-18-1991")
    summary = match["orchestration_summary"]
    assert summary["state"] == "running"
    assert summary["status"] == "running"
    assert summary["liveness"]["raw_status"] == "running"
    assert summary["liveness"]["display_status"] == "running"


def test_patient_orchestration_detail_exposes_current_run_over_newer_failed_run(
    temp_data_dir, monkeypatch
):
    from backend import storage

    with storage.session_scope() as session:
        patient = storage.create_patient(session, label="TV_01-18-1991_2", notes="")
        report_dir = (
            Path(temp_data_dir) / "reports" / patient.id / "report-mixed-detail"
        )
        report_dir.mkdir(parents=True, exist_ok=True)
        stored_path = report_dir / "original.txt"
        extracted_path = report_dir / "extracted.txt"
        stored_path.write_text("dummy", encoding="utf-8")
        extracted_path.write_text("dummy", encoding="utf-8")
        report = storage.create_report(
            session,
            report_id="report-mixed-detail",
            patient_id=patient.id,
            filename="source.pdf",
            mime_type="application/pdf",
            stored_path=stored_path,
            extracted_text_path=extracted_path,
        )
        running_run = storage.create_run(
            session,
            patient_id=patient.id,
            report_id=report.id,
            council_model_ids=["gpt-5.4"],
            consolidator_model_id="claude-sonnet-4-6",
        )
        storage.update_run_status(session, running_run.id, status="running")
        failed_run = storage.create_run(
            session,
            patient_id=patient.id,
            report_id=report.id,
            council_model_ids=["gpt-5.4"],
            consolidator_model_id="claude-sonnet-4-6",
        )
        storage.update_run_status(
            session, failed_run.id, status="failed", error_message="boom"
        )

    app = _test_app(temp_data_dir, monkeypatch)
    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.get(f"/api/patients/{patient.id}/orchestration")

    assert response.status_code == 200
    payload = response.json()
    assert payload["latest_run"]["display_status"] == "failed"
    assert payload["current_run"]["display_status"] == "running"
    assert payload["current_run"]["id"] == payload["active_runs"][0]["id"]


def test_patient_orchestration_summary_prefers_complete_over_pipeline_failure(
    temp_data_dir, monkeypatch
):
    from backend import storage

    with storage.session_scope() as session:
        patient = storage.create_patient(session, label="LW_02-28-1978", notes="")
        report_dir = Path(temp_data_dir) / "reports" / patient.id / "report-complete"
        report_dir.mkdir(parents=True, exist_ok=True)
        stored_path = report_dir / "original.txt"
        extracted_path = report_dir / "extracted.txt"
        stored_path.write_text("dummy", encoding="utf-8")
        extracted_path.write_text("dummy", encoding="utf-8")
        report = storage.create_report(
            session,
            report_id="report-complete",
            patient_id=patient.id,
            filename="source.pdf",
            mime_type="application/pdf",
            stored_path=stored_path,
            extracted_text_path=extracted_path,
        )
        run = storage.create_run(
            session,
            patient_id=patient.id,
            report_id=report.id,
            council_model_ids=["gpt-5.4"],
            consolidator_model_id="claude-sonnet-4-6",
        )
        storage.update_run_status(session, run.id, status="complete")
        peer_path = (
            Path(temp_data_dir) / "artifacts" / run.id / "stage-2" / "gpt-5.4.json"
        )
        peer_path.parent.mkdir(parents=True, exist_ok=True)
        peer_path.write_text("{}", encoding="utf-8")
        storage.create_artifact(
            session,
            run_id=run.id,
            stage_num=2,
            stage_name="peer_review",
            model_id="gpt-5.4",
            kind="peer_review",
            content_path=peer_path,
            content_type="application/json",
        )
        revision_path = (
            Path(temp_data_dir) / "artifacts" / run.id / "stage-3" / "gpt-5.4.md"
        )
        revision_path.parent.mkdir(parents=True, exist_ok=True)
        revision_path.write_text("# Revision", encoding="utf-8")
        storage.create_artifact(
            session,
            run_id=run.id,
            stage_num=3,
            stage_name="revision",
            model_id="gpt-5.4",
            kind="revision",
            content_path=revision_path,
            content_type="text/markdown",
        )

    portal_dir = Path(temp_data_dir) / "portal_patients" / "LW_02-28-1978"
    portal_dir.mkdir(parents=True, exist_ok=True)
    (
        portal_dir / "LW_02-28-1978__patient-facing__auto-test__2026-05-09.pdf"
    ).write_bytes(b"%PDF-1.4\n")
    cathode_dir = Path(temp_data_dir) / "cathode_projects" / "LW_02-28-1978"
    cathode_dir.mkdir(parents=True, exist_ok=True)
    (cathode_dir / "qeeg_handoff_payload.json").write_text("{}", encoding="utf-8")
    monkeypatch.setenv(
        "QEEG_CATHODE_PROJECTS_DIR", str(Path(temp_data_dir) / "cathode_projects")
    )

    status_dir = Path(temp_data_dir) / "pipeline_jobs"
    status_dir.mkdir(parents=True, exist_ok=True)
    (status_dir / "LW_02-28-1978.json").write_text(
        json.dumps(
            {
                "patient_id": "LW_02-28-1978",
                "status": "failed",
                "note": "worker saw duplicate legacy PDFs",
            }
        ),
        encoding="utf-8",
    )

    app = _test_app(temp_data_dir, monkeypatch)
    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.get("/api/patients")

    assert response.status_code == 200
    payload = response.json()
    match = next(item for item in payload if item["label"] == "LW_02-28-1978")
    summary = match["orchestration_summary"]
    assert summary["state"] == "ready"
    assert summary["status"] == "complete"


def test_prepare_cathode_handoff_action_writes_payload_and_source(
    temp_data_dir, monkeypatch
):
    from backend import storage

    cathode_root = Path(temp_data_dir) / "cathode_projects"
    monkeypatch.setenv("QEEG_CATHODE_PROJECTS_DIR", str(cathode_root))

    with storage.session_scope() as session:
        patient = storage.create_patient(session, label="AB_03-05-2010", notes="")
        report_dir = Path(temp_data_dir) / "reports" / patient.id / "report-3"
        report_dir.mkdir(parents=True, exist_ok=True)
        stored_path = report_dir / "original.txt"
        extracted_path = report_dir / "extracted.txt"
        stored_path.write_text("dummy", encoding="utf-8")
        extracted_path.write_text("dummy", encoding="utf-8")
        report = storage.create_report(
            session,
            report_id="report-3",
            patient_id=patient.id,
            filename="source.pdf",
            mime_type="application/pdf",
            stored_path=stored_path,
            extracted_text_path=extracted_path,
        )
        run = storage.create_run(
            session,
            patient_id=patient.id,
            report_id=report.id,
            council_model_ids=["claude-sonnet-4-6"],
            consolidator_model_id="claude-sonnet-4-6",
        )
        storage.update_run_status(session, run.id, status="complete")
        peer_path = (
            Path(temp_data_dir)
            / "artifacts"
            / run.id
            / "stage-2"
            / "claude-sonnet-4-6.json"
        )
        peer_path.parent.mkdir(parents=True, exist_ok=True)
        peer_path.write_text("{}", encoding="utf-8")
        storage.create_artifact(
            session,
            run_id=run.id,
            stage_num=2,
            stage_name="peer_review",
            model_id="claude-sonnet-4-6",
            kind="peer_review",
            content_path=peer_path,
            content_type="application/json",
        )
        revision_path = (
            Path(temp_data_dir)
            / "artifacts"
            / run.id
            / "stage-3"
            / "claude-sonnet-4-6.md"
        )
        revision_path.parent.mkdir(parents=True, exist_ok=True)
        revision_path.write_text("# Revision", encoding="utf-8")
        storage.create_artifact(
            session,
            run_id=run.id,
            stage_num=3,
            stage_name="revision",
            model_id="claude-sonnet-4-6",
            kind="revision",
            content_path=revision_path,
            content_type="text/markdown",
        )
        artifact_dir = Path(temp_data_dir) / "artifacts" / run.id / "stage-4"
        artifact_dir.mkdir(parents=True, exist_ok=True)
        artifact_path = artifact_dir / "claude-sonnet-4-6.md"
        artifact_path.write_text(
            "# Consolidation\nusable for cathode", encoding="utf-8"
        )
        artifact = storage.create_artifact(
            session,
            run_id=run.id,
            stage_num=4,
            stage_name="consolidation",
            model_id="claude-sonnet-4-6",
            kind="consolidation",
            content_path=artifact_path,
            content_type="text/markdown",
        )
        storage.select_artifact(session, run.id, artifact.id)

    app = _test_app(temp_data_dir, monkeypatch)
    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.post(
            f"/api/patients/{patient.id}/actions/prepare_cathode_handoff",
            json={},
        )

    assert response.status_code == 200
    payload = response.json()
    source_path = Path(payload["source_markdown_path"])
    handoff_payload_path = Path(payload["payload_path"])
    assert source_path.exists()
    assert handoff_payload_path.exists()
    assert "usable for cathode" in source_path.read_text(encoding="utf-8")
    handoff_payload = json.loads(handoff_payload_path.read_text(encoding="utf-8"))
    assert handoff_payload["ready_for_handoff"] is True
    assert handoff_payload["qeeg_source"]["run_id"] == payload["run_id"]


def test_prepare_cathode_handoff_action_falls_back_to_peer_reviewed_source(
    temp_data_dir, monkeypatch
):
    from backend import storage

    cathode_root = Path(temp_data_dir) / "cathode_projects"
    monkeypatch.setenv("QEEG_CATHODE_PROJECTS_DIR", str(cathode_root))

    with storage.session_scope() as session:
        patient = storage.create_patient(session, label="AB_03-05-2010", notes="")
        report_dir = Path(temp_data_dir) / "reports" / patient.id / "report-cathode"
        report_dir.mkdir(parents=True, exist_ok=True)
        stored_path = report_dir / "original.txt"
        extracted_path = report_dir / "extracted.txt"
        stored_path.write_text("dummy", encoding="utf-8")
        extracted_path.write_text("dummy", encoding="utf-8")
        report = storage.create_report(
            session,
            report_id="report-cathode",
            patient_id=patient.id,
            filename="source.pdf",
            mime_type="application/pdf",
            stored_path=stored_path,
            extracted_text_path=extracted_path,
        )

        older_run = storage.create_run(
            session,
            patient_id=patient.id,
            report_id=report.id,
            council_model_ids=["mock-council-a"],
            consolidator_model_id="mock-consolidator",
        )
        storage.update_run_status(session, older_run.id, status="complete")
        older_root = Path(temp_data_dir) / "artifacts" / older_run.id
        for stage_num, stage_name, kind, suffix, text, content_type in (
            (
                2,
                "peer_review",
                "peer_review",
                "peer-review.json",
                "{}",
                "application/json",
            ),
            (
                3,
                "revision",
                "revision",
                "revision.md",
                "# Revision",
                "text/markdown",
            ),
            (
                4,
                "consolidation",
                "consolidation",
                "consolidation.md",
                "# Consolidation\nolder valid source",
                "text/markdown",
            ),
        ):
            artifact_path = older_root / f"stage-{stage_num}" / suffix
            artifact_path.parent.mkdir(parents=True, exist_ok=True)
            artifact_path.write_text(text, encoding="utf-8")
            storage.create_artifact(
                session,
                run_id=older_run.id,
                stage_num=stage_num,
                stage_name=stage_name,
                model_id="mock-council-a",
                kind=kind,
                content_path=artifact_path,
                content_type=content_type,
            )

        newer_run = storage.create_run(
            session,
            patient_id=patient.id,
            report_id=report.id,
            council_model_ids=["mock-council-a"],
            consolidator_model_id="mock-consolidator",
        )
        storage.update_run_status(session, newer_run.id, status="complete")
        newer_path = (
            Path(temp_data_dir)
            / "artifacts"
            / newer_run.id
            / "stage-4"
            / "consolidation.md"
        )
        newer_path.parent.mkdir(parents=True, exist_ok=True)
        newer_path.write_text(
            "# Consolidation\nnewer unreviewed source", encoding="utf-8"
        )
        storage.create_artifact(
            session,
            run_id=newer_run.id,
            stage_num=4,
            stage_name="consolidation",
            model_id="mock-council-a",
            kind="consolidation",
            content_path=newer_path,
            content_type="text/markdown",
        )
        patient_id = patient.id
        older_run_id = older_run.id
        newer_run_id = newer_run.id

    app = _test_app(temp_data_dir, monkeypatch)
    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.post(
            f"/api/patients/{patient_id}/actions/prepare_cathode_handoff",
            json={},
        )
        requested_response = client.post(
            f"/api/patients/{patient_id}/actions/prepare_cathode_handoff",
            json={"run_id": newer_run_id},
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["run_id"] == older_run_id
    assert payload["run_id"] != newer_run_id
    assert requested_response.status_code == 409
    assert "No peer-reviewed council markdown artifact" in requested_response.text
    source_text = Path(payload["source_markdown_path"]).read_text(encoding="utf-8")
    assert "older valid source" in source_text
    assert "newer unreviewed source" not in source_text


def test_choose_cathode_source_artifact_prefers_stage4_then_stage3(
    temp_data_dir, monkeypatch
):
    from backend import storage
    from backend.orchestration import choose_cathode_source_artifact

    with storage.session_scope() as session:
        patient = storage.create_patient(session, label="AB_03-05-2010", notes="")
        report_dir = Path(temp_data_dir) / "reports" / patient.id / "report-4"
        report_dir.mkdir(parents=True, exist_ok=True)
        stored_path = report_dir / "original.txt"
        extracted_path = report_dir / "extracted.txt"
        stored_path.write_text("dummy", encoding="utf-8")
        extracted_path.write_text("dummy", encoding="utf-8")
        report = storage.create_report(
            session,
            report_id="report-4",
            patient_id=patient.id,
            filename="source.pdf",
            mime_type="application/pdf",
            stored_path=stored_path,
            extracted_text_path=extracted_path,
        )

        newer_run = storage.create_run(
            session,
            patient_id=patient.id,
            report_id=report.id,
            council_model_ids=["claude-sonnet-4-6"],
            consolidator_model_id="claude-sonnet-4-6",
        )
        storage.update_run_status(session, newer_run.id, status="failed")
        newer_artifact_dir = (
            Path(temp_data_dir) / "artifacts" / newer_run.id / "stage-3"
        )
        newer_artifact_dir.mkdir(parents=True, exist_ok=True)
        newer_artifact_path = newer_artifact_dir / "revision.md"
        newer_artifact_path.write_text("# Revision\nusable fallback", encoding="utf-8")
        storage.create_artifact(
            session,
            run_id=newer_run.id,
            stage_num=3,
            stage_name="revision",
            model_id="claude-sonnet-4-6",
            kind="revision",
            content_path=newer_artifact_path,
            content_type="text/markdown",
        )

        older_run = storage.create_run(
            session,
            patient_id=patient.id,
            report_id=report.id,
            council_model_ids=["claude-sonnet-4-6"],
            consolidator_model_id="claude-sonnet-4-6",
        )
        storage.update_run_status(session, older_run.id, status="complete")
        older_artifact_dir = (
            Path(temp_data_dir) / "artifacts" / older_run.id / "stage-4"
        )
        older_artifact_dir.mkdir(parents=True, exist_ok=True)
        older_artifact_path = older_artifact_dir / "consolidation.md"
        older_artifact_path.write_text("# Consolidation\npreferred", encoding="utf-8")
        storage.create_artifact(
            session,
            run_id=older_run.id,
            stage_num=4,
            stage_name="consolidation",
            model_id="claude-sonnet-4-6",
            kind="consolidation",
            content_path=older_artifact_path,
            content_type="text/markdown",
        )

        chosen = choose_cathode_source_artifact(session, patient_id=patient.id)

    assert chosen is not None
    chosen_run, chosen_artifact = chosen
    assert chosen_run.id == older_run.id
    assert chosen_artifact.stage_num == 4


def test_export_council_artifacts_action_exports_selected_final_draft(
    temp_data_dir, monkeypatch
):
    from backend import storage
    from backend import main

    with storage.session_scope() as session:
        patient = storage.create_patient(session, label="AB_03-05-2010", notes="")
        report_dir = Path(temp_data_dir) / "reports" / patient.id / "report-5"
        report_dir.mkdir(parents=True, exist_ok=True)
        stored_path = report_dir / "original.txt"
        extracted_path = report_dir / "extracted.txt"
        stored_path.write_text("dummy", encoding="utf-8")
        extracted_path.write_text("dummy", encoding="utf-8")
        report = storage.create_report(
            session,
            report_id="report-5",
            patient_id=patient.id,
            filename="source.pdf",
            mime_type="application/pdf",
            stored_path=stored_path,
            extracted_text_path=extracted_path,
        )
        run = storage.create_run(
            session,
            patient_id=patient.id,
            report_id=report.id,
            council_model_ids=["claude-sonnet-4-6"],
            consolidator_model_id="claude-sonnet-4-6",
        )
        storage.update_run_status(session, run.id, status="complete")
        artifact_root = Path(temp_data_dir) / "artifacts" / run.id
        stage2_path = artifact_root / "stage-2" / "peer-review.json"
        stage3_path = artifact_root / "stage-3" / "revision.md"
        artifact_path = artifact_root / "stage-6" / "final-draft.md"
        for path, text in (
            (stage2_path, "{}"),
            (stage3_path, "# Revision\nready"),
            (artifact_path, "# Final Draft\nexportable"),
        ):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="utf-8")
        storage.create_artifact(
            session,
            run_id=run.id,
            stage_num=2,
            stage_name="peer_review",
            model_id="claude-sonnet-4-6",
            kind="peer_review",
            content_path=stage2_path,
            content_type="application/json",
        )
        storage.create_artifact(
            session,
            run_id=run.id,
            stage_num=3,
            stage_name="revision",
            model_id="claude-sonnet-4-6",
            kind="revision",
            content_path=stage3_path,
            content_type="text/markdown",
        )
        artifact = storage.create_artifact(
            session,
            run_id=run.id,
            stage_num=6,
            stage_name="final_draft",
            model_id="claude-sonnet-4-6",
            kind="final_draft",
            content_path=artifact_path,
            content_type="text/markdown",
        )
        storage.select_artifact(session, run.id, artifact.id)

    monkeypatch.setattr(
        main, "render_markdown_to_pdf", lambda md, path: path.write_bytes(b"%PDF-1.4")
    )
    monkeypatch.setattr(main, "_publish_file_to_portal_folder", lambda **kwargs: None)
    monkeypatch.setattr(main, "_schedule_portal_sync", lambda *args, **kwargs: None)

    app = _test_app(temp_data_dir, monkeypatch)
    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.post(
            f"/api/patients/{patient.id}/actions/export_council_artifacts",
            json={},
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["run_id"] == run.id
    assert Path(payload["final_md"]).exists()
    assert Path(payload["final_pdf"]).exists()


def test_export_council_artifacts_action_falls_back_to_export_ready_run(
    temp_data_dir, monkeypatch
):
    from backend import main, storage

    with storage.session_scope() as session:
        patient = storage.create_patient(session, label="AB_03-05-2010", notes="")
        report_dir = Path(temp_data_dir) / "reports" / patient.id / "report-export"
        report_dir.mkdir(parents=True, exist_ok=True)
        stored_path = report_dir / "original.txt"
        extracted_path = report_dir / "extracted.txt"
        stored_path.write_text("dummy", encoding="utf-8")
        extracted_path.write_text("dummy", encoding="utf-8")
        report = storage.create_report(
            session,
            report_id="report-export",
            patient_id=patient.id,
            filename="source.pdf",
            mime_type="application/pdf",
            stored_path=stored_path,
            extracted_text_path=extracted_path,
        )

        older_run = storage.create_run(
            session,
            patient_id=patient.id,
            report_id=report.id,
            council_model_ids=["mock-council-a"],
            consolidator_model_id="mock-consolidator",
        )
        storage.update_run_status(session, older_run.id, status="complete")
        older_root = Path(temp_data_dir) / "artifacts" / older_run.id
        for stage_num, stage_name, kind, suffix, text, content_type in (
            (
                2,
                "peer_review",
                "peer_review",
                "peer-review.json",
                "{}",
                "application/json",
            ),
            (
                3,
                "revision",
                "revision",
                "revision.md",
                "# Revision",
                "text/markdown",
            ),
            (
                6,
                "final_draft",
                "final_draft",
                "final-draft.md",
                "# Final Draft\nexportable",
                "text/markdown",
            ),
        ):
            artifact_path = older_root / f"stage-{stage_num}" / suffix
            artifact_path.parent.mkdir(parents=True, exist_ok=True)
            artifact_path.write_text(text, encoding="utf-8")
            artifact = storage.create_artifact(
                session,
                run_id=older_run.id,
                stage_num=stage_num,
                stage_name=stage_name,
                model_id="mock-council-a",
                kind=kind,
                content_path=artifact_path,
                content_type=content_type,
            )
            if stage_num == 6:
                storage.select_artifact(session, older_run.id, artifact.id)

        newer_run = storage.create_run(
            session,
            patient_id=patient.id,
            report_id=report.id,
            council_model_ids=["mock-council-a"],
            consolidator_model_id="mock-consolidator",
        )
        storage.update_run_status(session, newer_run.id, status="complete")
        newer_root = Path(temp_data_dir) / "artifacts" / newer_run.id
        for stage_num, stage_name, kind, suffix, text, content_type in (
            (
                2,
                "peer_review",
                "peer_review",
                "peer-review.json",
                "{}",
                "application/json",
            ),
            (
                3,
                "revision",
                "revision",
                "revision.md",
                "# Revision",
                "text/markdown",
            ),
        ):
            artifact_path = newer_root / f"stage-{stage_num}" / suffix
            artifact_path.parent.mkdir(parents=True, exist_ok=True)
            artifact_path.write_text(text, encoding="utf-8")
            storage.create_artifact(
                session,
                run_id=newer_run.id,
                stage_num=stage_num,
                stage_name=stage_name,
                model_id="mock-council-a",
                kind=kind,
                content_path=artifact_path,
                content_type=content_type,
            )
        patient_id = patient.id
        older_run_id = older_run.id
        newer_run_id = newer_run.id

    monkeypatch.setattr(
        main, "render_markdown_to_pdf", lambda md, path: path.write_bytes(b"%PDF-1.4")
    )
    monkeypatch.setattr(main, "_publish_file_to_portal_folder", lambda **kwargs: None)
    monkeypatch.setattr(main, "_schedule_portal_sync", lambda *args, **kwargs: None)

    app = _test_app(temp_data_dir, monkeypatch)
    with TestClient(app, raise_server_exceptions=False) as client:
        detail_response = client.get(f"/api/patients/{patient_id}/orchestration")
        response = client.post(
            f"/api/patients/{patient_id}/actions/export_council_artifacts",
            json={},
        )
        requested_response = client.post(
            f"/api/patients/{patient_id}/actions/export_council_artifacts",
            json={"run_id": newer_run_id},
        )

    assert detail_response.status_code == 200
    assert (
        detail_response.json()["actions"]["export_council_artifacts"]["enabled"] is True
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["run_id"] == older_run_id
    assert payload["run_id"] != newer_run_id
    assert requested_response.status_code == 409
    assert "No export-ready complete run" in requested_response.text
    assert Path(payload["final_md"]).exists()
    assert Path(payload["final_pdf"]).exists()
