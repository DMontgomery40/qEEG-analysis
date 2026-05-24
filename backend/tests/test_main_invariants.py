from __future__ import annotations

import json
import uuid
from pathlib import Path

from fastapi.testclient import TestClient


def _test_app(temp_data_dir, monkeypatch):
    monkeypatch.setenv("QEEG_MOCK_LLM", "1")
    monkeypatch.setenv("QEEG_PORTAL_RAW_SYNC_WATCHER", "0")
    from backend import main

    monkeypatch.setattr(
        main,
        "_ensure_project_clipr_config",
        lambda: Path(temp_data_dir) / "cliproxyapi.conf",
    )
    monkeypatch.setattr(main, "_sync_home_auth_to_project", lambda: 0)
    monkeypatch.setattr(main, "EXPORTS_DIR", Path(temp_data_dir) / "exports")
    return main.app, main


def _create_report(
    storage, temp_data_dir, *, patient_id: str, filename: str = "report.txt"
):
    report_id = str(uuid.uuid4())
    report_dir = Path(temp_data_dir) / "reports" / patient_id / report_id
    report_dir.mkdir(parents=True, exist_ok=True)
    stored_path = report_dir / filename
    extracted_path = report_dir / "extracted.txt"
    stored_path.write_text("dummy report", encoding="utf-8")
    extracted_path.write_text("dummy extracted", encoding="utf-8")
    with storage.session_scope() as session:
        report = storage.create_report(
            session,
            report_id=report_id,
            patient_id=patient_id,
            filename=filename,
            mime_type="text/plain",
            stored_path=stored_path,
            extracted_text_path=extracted_path,
        )
    return report


def test_create_run_rejects_report_from_different_patient(temp_data_dir, monkeypatch):
    app, _main = _test_app(temp_data_dir, monkeypatch)
    from backend import storage

    with storage.session_scope() as session:
        patient_a = storage.create_patient(session, label="A", notes="")
        patient_b = storage.create_patient(session, label="B", notes="")

    report = _create_report(storage, temp_data_dir, patient_id=patient_b.id)

    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.post(
            "/api/runs",
            json={
                "patient_id": patient_a.id,
                "report_id": report.id,
                "council_model_ids": ["mock-council-a"],
                "consolidator_model_id": "mock-consolidator",
            },
        )

    assert response.status_code == 400
    assert response.json()["detail"] == "Report does not belong to patient"


def test_start_run_is_idempotent_once_claimed(temp_data_dir, monkeypatch):
    app, main = _test_app(temp_data_dir, monkeypatch)
    from backend import storage

    with storage.session_scope() as session:
        patient = storage.create_patient(session, label="P", notes="")
    report = _create_report(storage, temp_data_dir, patient_id=patient.id)
    with storage.session_scope() as session:
        run = storage.create_run(
            session,
            patient_id=patient.id,
            report_id=report.id,
            council_model_ids=["mock-council-a"],
            consolidator_model_id="mock-consolidator",
        )

    scheduled: list[str | None] = []

    def fake_create_task(coro, *, name=None):
        scheduled.append(name)
        coro.close()

        class _Task:
            pass

        return _Task()

    monkeypatch.setattr(main, "_spawn_task", fake_create_task)

    with TestClient(app, raise_server_exceptions=False) as client:
        first = client.post(f"/api/runs/{run.id}/start")
        second = client.post(f"/api/runs/{run.id}/start")

    assert first.status_code == 200
    assert first.json() == {"ok": True}
    assert second.status_code == 200
    assert second.json()["status"] == "running"
    assert scheduled == [f"qeeg-run-{run.id}"]


def test_select_rejects_artifact_from_different_run(temp_data_dir, monkeypatch):
    app, _main = _test_app(temp_data_dir, monkeypatch)
    from backend import storage

    with storage.session_scope() as session:
        patient = storage.create_patient(session, label="P", notes="")
    report = _create_report(storage, temp_data_dir, patient_id=patient.id)
    artifact_path = Path(temp_data_dir) / "artifacts" / "foreign.md"
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text("# final", encoding="utf-8")

    with storage.session_scope() as session:
        run_a = storage.create_run(
            session,
            patient_id=patient.id,
            report_id=report.id,
            council_model_ids=["mock-council-a"],
            consolidator_model_id="mock-consolidator",
        )
        run_b = storage.create_run(
            session,
            patient_id=patient.id,
            report_id=report.id,
            council_model_ids=["mock-council-a"],
            consolidator_model_id="mock-consolidator",
        )
        artifact = storage.create_artifact(
            session,
            run_id=run_b.id,
            stage_num=6,
            stage_name="final_draft",
            model_id="mock-council-a",
            kind="final_draft",
            content_path=artifact_path,
            content_type="text/markdown",
        )

    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.post(
            f"/api/runs/{run_a.id}/select", json={"artifact_id": artifact.id}
        )

    assert response.status_code == 404
    assert response.json()["detail"] == "Artifact not found for run"


def test_export_rejects_selected_artifact_that_is_not_final_markdown(
    temp_data_dir, monkeypatch
):
    app, _main = _test_app(temp_data_dir, monkeypatch)
    from backend import storage

    with storage.session_scope() as session:
        patient = storage.create_patient(session, label="09-05-1954-0", notes="")
    report = _create_report(storage, temp_data_dir, patient_id=patient.id)
    artifact_path = Path(temp_data_dir) / "artifacts" / "stage5.json"
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text('{"vote":"APPROVE"}', encoding="utf-8")

    with storage.session_scope() as session:
        run = storage.create_run(
            session,
            patient_id=patient.id,
            report_id=report.id,
            council_model_ids=["mock-council-a"],
            consolidator_model_id="mock-consolidator",
        )
        artifact = storage.create_artifact(
            session,
            run_id=run.id,
            stage_num=5,
            stage_name="final_review",
            model_id="mock-council-a",
            kind="final_review",
            content_path=artifact_path,
            content_type="application/json",
        )
        storage.select_artifact(session, run.id, artifact.id)

    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.post(f"/api/runs/{run.id}/export")

    assert response.status_code == 400
    assert (
        response.json()["detail"] == "Selected artifact is not a final markdown draft"
    )


def test_export_rejects_selected_artifact_from_different_run(
    temp_data_dir, monkeypatch
):
    app, _main = _test_app(temp_data_dir, monkeypatch)
    from backend import storage

    with storage.session_scope() as session:
        patient = storage.create_patient(session, label="09-05-1954-0", notes="")
    report = _create_report(storage, temp_data_dir, patient_id=patient.id)
    artifact_path = Path(temp_data_dir) / "artifacts" / "final.md"
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text("# Final Draft", encoding="utf-8")

    with storage.session_scope() as session:
        run_a = storage.create_run(
            session,
            patient_id=patient.id,
            report_id=report.id,
            council_model_ids=["mock-council-a"],
            consolidator_model_id="mock-consolidator",
        )
        run_b = storage.create_run(
            session,
            patient_id=patient.id,
            report_id=report.id,
            council_model_ids=["mock-council-a"],
            consolidator_model_id="mock-consolidator",
        )
        artifact = storage.create_artifact(
            session,
            run_id=run_b.id,
            stage_num=6,
            stage_name="final_draft",
            model_id="mock-council-a",
            kind="final_draft",
            content_path=artifact_path,
            content_type="text/markdown",
        )
        run_a_row = storage.get_run(session, run_a.id)
        run_a_row.selected_artifact_id = artifact.id
        session.commit()

    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.post(f"/api/runs/{run_a.id}/export")

    assert response.status_code == 400
    assert response.json()["detail"] == "Selected artifact does not belong to run"


def test_create_and_update_patient_reject_duplicate_labels(temp_data_dir, monkeypatch):
    app, _main = _test_app(temp_data_dir, monkeypatch)

    with TestClient(app, raise_server_exceptions=False) as client:
        first = client.post("/api/patients", json={"label": "Same Label", "notes": ""})
        second = client.post("/api/patients", json={"label": "same label", "notes": ""})
        other = client.post("/api/patients", json={"label": "Other Label", "notes": ""})
        update = client.put(
            f"/api/patients/{other.json()['id']}",
            json={"label": "Same Label", "notes": ""},
        )

    assert first.status_code == 200
    assert second.status_code == 409
    assert second.json()["detail"] == "Patient label already exists"
    assert update.status_code == 409
    assert update.json()["detail"] == "Patient label already exists"


def test_delete_patient_file_removes_portal_copy(temp_data_dir, monkeypatch):
    app, _main = _test_app(temp_data_dir, monkeypatch)
    from backend import storage

    with storage.session_scope() as session:
        patient = storage.create_patient(session, label="09-05-1954-0", notes="")

    with TestClient(app, raise_server_exceptions=False) as client:
        upload = client.post(
            f"/api/patients/{patient.id}/files",
            files={"file": ("guide.pdf", b"%PDF-1.4\n", "application/pdf")},
        )
        file_id = upload.json()["file"]["id"]

        portal_path = (
            Path(temp_data_dir) / "portal_patients" / "09-05-1954-0" / "guide.pdf"
        )
        assert portal_path.exists()

        response = client.delete(f"/api/patient_files/{file_id}")

    assert response.status_code == 200
    assert response.json()["ok"] is True
    assert not portal_path.exists()


def test_upload_patient_file_schedules_portal_sync(temp_data_dir, monkeypatch):
    app, main = _test_app(temp_data_dir, monkeypatch)
    from backend import storage

    scheduled: list[tuple[str, str]] = []
    monkeypatch.setattr(
        main,
        "_schedule_portal_sync",
        lambda patient_label, *, source: scheduled.append((patient_label, source)),
    )

    with storage.session_scope() as session:
        patient = storage.create_patient(session, label="09-05-1954-0", notes="")

    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.post(
            f"/api/patients/{patient.id}/files",
            files={"file": ("guide.pdf", b"%PDF-1.4\n", "application/pdf")},
        )

    assert response.status_code == 200
    assert scheduled == [("09-05-1954-0", "upload_patient_file")]


def test_export_rejects_unreviewed_selected_final_draft(temp_data_dir, monkeypatch):
    app, main = _test_app(temp_data_dir, monkeypatch)
    from backend import storage

    with storage.session_scope() as session:
        patient = storage.create_patient(session, label="09-05-1954-0", notes="")
    report = _create_report(storage, temp_data_dir, patient_id=patient.id)
    artifact_path = Path(temp_data_dir) / "artifacts" / "final.md"
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text("# Final Draft", encoding="utf-8")

    with storage.session_scope() as session:
        run = storage.create_run(
            session,
            patient_id=patient.id,
            report_id=report.id,
            council_model_ids=["mock-council-a"],
            consolidator_model_id="mock-consolidator",
        )
        artifact = storage.create_artifact(
            session,
            run_id=run.id,
            stage_num=6,
            stage_name="final_draft",
            model_id="mock-council-a",
            kind="final_draft",
            content_path=artifact_path,
            content_type="text/markdown",
        )
        storage.select_artifact(session, run.id, artifact.id)
        storage.update_run_status(session, run.id, status="complete")

    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.post(f"/api/runs/{run.id}/export")

    assert response.status_code == 409
    assert "peer review did not complete" in response.json()["detail"]


def test_cached_export_download_rejects_unreviewed_run(temp_data_dir, monkeypatch):
    app, _main = _test_app(temp_data_dir, monkeypatch)
    from backend import storage

    with storage.session_scope() as session:
        patient = storage.create_patient(session, label="09-05-1954-0", notes="")
    report = _create_report(storage, temp_data_dir, patient_id=patient.id)
    artifact_path = Path(temp_data_dir) / "artifacts" / "final.md"
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text("# Final Draft", encoding="utf-8")

    with storage.session_scope() as session:
        run = storage.create_run(
            session,
            patient_id=patient.id,
            report_id=report.id,
            council_model_ids=["mock-council-a"],
            consolidator_model_id="mock-consolidator",
        )
        artifact = storage.create_artifact(
            session,
            run_id=run.id,
            stage_num=6,
            stage_name="final_draft",
            model_id="mock-council-a",
            kind="final_draft",
            content_path=artifact_path,
            content_type="text/markdown",
        )
        storage.select_artifact(session, run.id, artifact.id)
        storage.update_run_status(session, run.id, status="complete")

    cached_export = Path(temp_data_dir) / "exports" / run.id / "final.md"
    cached_export.parent.mkdir(parents=True, exist_ok=True)
    cached_export.write_text("# Final Draft", encoding="utf-8")

    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.get(f"/api/runs/{run.id}/export/final.md")

    assert response.status_code == 409
    assert "peer review did not complete" in response.json()["detail"]


def test_export_schedules_portal_sync_for_delivery_ready_run(temp_data_dir, monkeypatch):
    app, main = _test_app(temp_data_dir, monkeypatch)
    from backend import storage

    scheduled: list[tuple[str, str]] = []
    monkeypatch.setattr(
        main,
        "_schedule_portal_sync",
        lambda patient_label, *, source: scheduled.append((patient_label, source)),
    )

    with storage.session_scope() as session:
        patient = storage.create_patient(session, label="09-05-1954-0", notes="")
    report = _create_report(storage, temp_data_dir, patient_id=patient.id)
    artifact_dir = Path(temp_data_dir) / "artifacts"
    stage2_path = artifact_dir / "stage-2" / "peer-review.json"
    stage3_path = artifact_dir / "stage-3" / "revision.md"
    stage6_path = artifact_dir / "stage-6" / "final.md"
    for path, text in (
        (stage2_path, "{}"),
        (stage3_path, "# Revision"),
        (stage6_path, "# Final Draft"),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")

    with storage.session_scope() as session:
        run = storage.create_run(
            session,
            patient_id=patient.id,
            report_id=report.id,
            council_model_ids=["mock-council-a"],
            consolidator_model_id="mock-consolidator",
        )
        storage.update_run_status(session, run.id, status="complete")
        storage.create_artifact(
            session,
            run_id=run.id,
            stage_num=2,
            stage_name="peer_review",
            model_id="mock-council-a",
            kind="peer_review",
            content_path=stage2_path,
            content_type="application/json",
        )
        storage.create_artifact(
            session,
            run_id=run.id,
            stage_num=3,
            stage_name="revision",
            model_id="mock-council-a",
            kind="revision",
            content_path=stage3_path,
            content_type="text/markdown",
        )
        artifact = storage.create_artifact(
            session,
            run_id=run.id,
            stage_num=6,
            stage_name="final_draft",
            model_id="mock-council-a",
            kind="final_draft",
            content_path=stage6_path,
            content_type="text/markdown",
        )
        storage.select_artifact(session, run.id, artifact.id)

    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.post(f"/api/runs/{run.id}/export")
        download = client.get(f"/api/runs/{run.id}/export/final.md")
        Path(response.json()["final_md"]).write_text("# stale export", encoding="utf-8")
        stale_download = client.get(f"/api/runs/{run.id}/export/final.md")

    assert response.status_code == 200
    payload = response.json()
    meta_path = Path(payload["portal_export_meta"])
    assert meta_path.exists()
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    assert meta["run_id"] == run.id
    assert meta["selected_artifact_id"] == artifact.id
    assert download.status_code == 200
    assert download.text == "# Final Draft"
    assert stale_download.status_code == 409
    assert "no longer matches" in stale_download.json()["detail"]
    assert scheduled == [("09-05-1954-0", "export_run")]
