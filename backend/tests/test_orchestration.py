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
        patient = storage.create_patient(session, label="03-05-2010-0", notes="")
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


def test_summarize_run_progress_keeps_partial_success_visible_on_complete(temp_data_dir):
    from backend import storage
    from backend.orchestration import progress_jsonl_path, summarize_run_progress

    with storage.session_scope() as session:
        patient = storage.create_patient(session, label="03-05-2010-0", notes="")
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


def test_derive_run_liveness_marks_old_running_heartbeat_stale(temp_data_dir, monkeypatch):
    from backend import storage
    from backend.orchestration import (
        derive_run_liveness,
        progress_jsonl_path,
        summarize_run_progress,
    )

    monkeypatch.setenv("QEEG_RUN_STALE_AFTER_S", "300")

    with storage.session_scope() as session:
        patient = storage.create_patient(session, label="03-05-2010-0", notes="")
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
        patient = storage.create_patient(session, label="03-05-2010-0", notes="")
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
        patient = storage.create_patient(session, label="03-05-2010-0", notes="")
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

    portal_dir = Path(temp_data_dir) / "portal_patients" / "03-05-2010-0"
    portal_dir.mkdir(parents=True, exist_ok=True)
    (portal_dir / "03-05-2010-0__patient-facing__auto-test__2026-04-12.pdf").write_bytes(
        b"%PDF-1.4"
    )
    status_dir = Path(temp_data_dir) / "pipeline_jobs"
    status_dir.mkdir(parents=True, exist_ok=True)
    (status_dir / "03-05-2010-0.json").write_text(
        json.dumps({"patient_id": "03-05-2010-0", "status": "complete", "note": "all good"}),
        encoding="utf-8",
    )
    cathode_project = cathode_root / "03-05-2010-0"
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
        patient = storage.create_patient(session, label="12-02-1985-0", notes="")
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
        patient = storage.create_patient(session, label="01-18-1991-0", notes="")
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
        storage.update_run_status(session, failed_run.id, status="failed", error_message="boom")

    app = _test_app(temp_data_dir, monkeypatch)
    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.get("/api/patients")

    assert response.status_code == 200
    payload = response.json()
    match = next(item for item in payload if item["label"] == "01-18-1991-0")
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
        patient = storage.create_patient(session, label="01-18-1991-1", notes="")
        report_dir = Path(temp_data_dir) / "reports" / patient.id / "report-mixed-detail"
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
        storage.update_run_status(session, failed_run.id, status="failed", error_message="boom")

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
        patient = storage.create_patient(session, label="02-28-1978-0", notes="")
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

    status_dir = Path(temp_data_dir) / "pipeline_jobs"
    status_dir.mkdir(parents=True, exist_ok=True)
    (status_dir / "02-28-1978-0.json").write_text(
        json.dumps(
            {
                "patient_id": "02-28-1978-0",
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
    match = next(item for item in payload if item["label"] == "02-28-1978-0")
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
        patient = storage.create_patient(session, label="03-05-2010-0", notes="")
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
        artifact_dir = Path(temp_data_dir) / "artifacts" / run.id / "stage-4"
        artifact_dir.mkdir(parents=True, exist_ok=True)
        artifact_path = artifact_dir / "claude-sonnet-4-6.md"
        artifact_path.write_text("# Consolidation\nusable for cathode", encoding="utf-8")
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


def test_choose_cathode_source_artifact_prefers_stage4_then_stage3(
    temp_data_dir, monkeypatch
):
    from backend import storage
    from backend.orchestration import choose_cathode_source_artifact

    with storage.session_scope() as session:
        patient = storage.create_patient(session, label="03-05-2010-0", notes="")
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
        newer_artifact_dir = Path(temp_data_dir) / "artifacts" / newer_run.id / "stage-3"
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
        older_artifact_dir = Path(temp_data_dir) / "artifacts" / older_run.id / "stage-4"
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
        patient = storage.create_patient(session, label="03-05-2010-0", notes="")
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
        artifact_dir = Path(temp_data_dir) / "artifacts" / run.id / "stage-6"
        artifact_dir.mkdir(parents=True, exist_ok=True)
        artifact_path = artifact_dir / "final-draft.md"
        artifact_path.write_text("# Final Draft\nexportable", encoding="utf-8")
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

    monkeypatch.setattr(main, "render_markdown_to_pdf", lambda md, path: path.write_bytes(b"%PDF-1.4"))
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
