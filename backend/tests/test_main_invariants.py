from __future__ import annotations

import json
import re
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
        patient = storage.create_patient(session, label="HT_09-05-1954", notes="")
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
        patient = storage.create_patient(session, label="HT_09-05-1954", notes="")
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
        first = client.post(
            "/api/patients", json={"label": "BT_12-11-1963", "notes": ""}
        )
        second = client.post(
            "/api/patients", json={"label": "BT_12-11-1963", "notes": ""}
        )
        wrong_case = client.post(
            "/api/patients", json={"label": "bt_12-11-1963", "notes": ""}
        )
        other = client.post(
            "/api/patients", json={"label": "HT_09-05-1954", "notes": ""}
        )
        update = client.put(
            f"/api/patients/{other.json()['id']}",
            json={"label": "BT_12-11-1963", "notes": ""},
        )

    assert first.status_code == 200
    assert second.status_code == 409
    assert second.json()["detail"] == "Patient label already exists"
    assert update.status_code == 409
    assert update.json()["detail"] == "Patient label already exists"
    # A clinic id is case-strict, so a lowercase variant never reaches the
    # duplicate check — it is not an id at all. Keeps a case-divergent folder
    # from existing beside the real one on a case-preserving filesystem.
    assert wrong_case.status_code == 400


def test_delete_patient_file_removes_portal_copy(temp_data_dir, monkeypatch):
    app, _main = _test_app(temp_data_dir, monkeypatch)
    from backend import storage

    with storage.session_scope() as session:
        patient = storage.create_patient(session, label="HT_09-05-1954", notes="")

    with TestClient(app, raise_server_exceptions=False) as client:
        upload = client.post(
            f"/api/patients/{patient.id}/files",
            files={"file": ("guide.pdf", b"%PDF-1.4\n", "application/pdf")},
        )
        file_id = upload.json()["file"]["id"]

        portal_path = (
            Path(temp_data_dir) / "portal_patients" / "HT_09-05-1954" / "guide.pdf"
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
        patient = storage.create_patient(session, label="HT_09-05-1954", notes="")

    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.post(
            f"/api/patients/{patient.id}/files",
            files={"file": ("guide.pdf", b"%PDF-1.4\n", "application/pdf")},
        )

    assert response.status_code == 200
    assert scheduled == [("HT_09-05-1954", "upload_patient_file")]


def test_export_rejects_unreviewed_selected_final_draft(temp_data_dir, monkeypatch):
    app, main = _test_app(temp_data_dir, monkeypatch)
    from backend import storage

    with storage.session_scope() as session:
        patient = storage.create_patient(session, label="HT_09-05-1954", notes="")
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
        patient = storage.create_patient(session, label="HT_09-05-1954", notes="")
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
        patient = storage.create_patient(session, label="HT_09-05-1954", notes="")
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
    assert scheduled == [("HT_09-05-1954", "export_run")]


def test_bulk_upload_registers_a_report_under_an_allocated_canonical_id(
    temp_data_dir, monkeypatch
):
    """Intake order: identity first, canonical patient second, report third.

    Nothing downstream may carry the date of birth on its own. This asserts the
    database label, the portal folder, and the API payload all land on
    ``BT_12-11-1963`` even though the uploaded file is named after the DOB.
    """
    app, _main = _test_app(temp_data_dir, monkeypatch)
    from backend import storage

    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.post(
            "/api/patients/bulk_upload",
            files=[("files", ("12-11-1963-0.txt", b"qEEG report text", "text/plain"))],
            data={
                "identities": json.dumps(
                    [
                        {
                            "filename": "12-11-1963-0.txt",
                            "first_name": "Barto",
                            "last_name": "Tinker",
                            "birthdate": "12-11-1963",
                        }
                    ]
                )
            },
        )

    assert response.status_code == 200
    body = response.json()
    assert body["counts"] == {"created": 1, "skipped": 0, "errors": 0}
    created = body["created"][0]
    assert created["patient"]["patient_id"] == "BT_12-11-1963"
    assert created["patient"]["label"] == "BT_12-11-1963"
    assert created["patient"]["first_name"] == "Barto"

    with storage.session_scope() as session:
        labels = [patient.label for patient in storage.list_patients(session)]
        reports = storage.list_reports(session, created["patient"]["id"])
    assert labels == ["BT_12-11-1963"]
    assert [report.filename for report in reports] == ["12-11-1963-0.txt"]

    portal_root = Path(temp_data_dir) / "portal_patients"
    assert sorted(path.name for path in portal_root.iterdir()) == ["BT_12-11-1963"]


def test_bulk_upload_without_identity_creates_no_patient_and_no_folder(
    temp_data_dir, monkeypatch
):
    """A file the operator has not identified is an error, never a fallback label."""
    app, _main = _test_app(temp_data_dir, monkeypatch)
    from backend import storage

    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.post(
            "/api/patients/bulk_upload",
            files=[("files", ("12-11-1963-0.txt", b"qEEG report text", "text/plain"))],
        )

    assert response.status_code == 200
    body = response.json()
    assert body["counts"] == {"created": 0, "skipped": 0, "errors": 1}
    assert body["errors"][0]["filename"] == "12-11-1963-0.txt"

    with storage.session_scope() as session:
        assert storage.list_patients(session) == []

    portal_root = Path(temp_data_dir) / "portal_patients"
    assert not portal_root.exists() or list(portal_root.iterdir()) == []


def test_bulk_upload_reuses_the_patient_already_wearing_that_canonical_id(
    temp_data_dir, monkeypatch
):
    """The same person twice is one patient, never a ``_2`` duplicate family."""
    app, _main = _test_app(temp_data_dir, monkeypatch)
    from backend import storage

    identity = {
        "first_name": "Barto",
        "last_name": "Tinker",
        "birthdate": "12-11-1963",
    }
    with TestClient(app, raise_server_exceptions=False) as client:
        first = client.post(
            "/api/patients/bulk_upload",
            files=[("files", ("session-one.txt", b"qEEG report text", "text/plain"))],
            data={"identities": json.dumps([{"filename": "session-one.txt", **identity}])},
        )
        second = client.post(
            "/api/patients/bulk_upload",
            files=[("files", ("session-two.txt", b"qEEG report text", "text/plain"))],
            data={"identities": json.dumps([{"filename": "session-two.txt", **identity}])},
        )

    assert first.json()["counts"]["created"] == 1
    assert second.json()["counts"]["created"] == 1
    patient_ids = {
        first.json()["created"][0]["patient"]["id"],
        second.json()["created"][0]["patient"]["id"],
    }
    assert len(patient_ids) == 1

    with storage.session_scope() as session:
        labels = [patient.label for patient in storage.list_patients(session)]
        reports = storage.list_reports(session, patient_ids.pop())
    assert labels == ["BT_12-11-1963"]
    assert sorted(report.filename for report in reports) == [
        "session-one.txt",
        "session-two.txt",
    ]


def test_portal_publishing_refuses_a_legacy_dob_label(temp_data_dir, monkeypatch):
    """``MM-DD-YYYY-N`` is not a patient id any more, so nothing routes on it."""
    app, main = _test_app(temp_data_dir, monkeypatch)
    from backend import storage

    with storage.session_scope() as session:
        patient = storage.create_patient(session, label="09-05-1954-0", notes="")

    scheduled: list[tuple[str, str]] = []
    monkeypatch.setattr(
        main,
        "_schedule_portal_sync",
        lambda patient_label, *, source: scheduled.append((patient_label, source)),
    )

    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.post(
            f"/api/patients/{patient.id}/files",
            files={"file": ("guide.pdf", b"%PDF-1.4\n", "application/pdf")},
        )

    assert response.status_code == 200
    assert response.json()["portal_published_path"] is None
    assert scheduled == []
    assert not (Path(temp_data_dir) / "portal_patients" / "09-05-1954-0").exists()


def test_bulk_upload_names_the_ambiguity_instead_of_adding_a_third_chart(
    temp_data_dir, monkeypatch
):
    """Two charts already matching one identity is the operator's call, not a guess."""
    app, _main = _test_app(temp_data_dir, monkeypatch)
    from backend import storage

    for label in ("BT_12-11-1963", "BT_12-11-1963_2"):
        with storage.session_scope() as session:
            storage.create_patient(
                session,
                label=label,
                notes="",
                first_name="Barto",
                last_name="Tinker",
                birthdate="12-11-1963",
                first_initial="B",
                last_initial="T",
            )

    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.post(
            "/api/patients/bulk_upload",
            files=[("files", ("session-three.txt", b"qEEG report text", "text/plain"))],
            data={
                "identities": json.dumps(
                    [
                        {
                            "filename": "session-three.txt",
                            "first_name": "Barto",
                            "last_name": "Tinker",
                            "birthdate": "12-11-1963",
                        }
                    ]
                )
            },
        )

    assert response.status_code == 200
    body = response.json()
    assert body["counts"]["created"] == 0
    assert "BT_12-11-1963, BT_12-11-1963_2" in body["errors"][0]["error"]

    with storage.session_scope() as session:
        labels = sorted(patient.label for patient in storage.list_patients(session))
    assert labels == ["BT_12-11-1963", "BT_12-11-1963_2"]


def test_bulk_upload_lands_on_the_chart_created_from_a_bare_canonical_label(
    temp_data_dir, monkeypatch
):
    """A chart with no names on it is still that patient's chart.

    Creating a patient from a bare canonical label stores the initials and date
    of birth but no names. The first report to arrive with a full name must land
    on that chart and fill the names in — not allocate `_2` and split one person
    into two families.
    """
    app, _main = _test_app(temp_data_dir, monkeypatch)
    from backend import storage

    with TestClient(app, raise_server_exceptions=False) as client:
        created = client.post(
            "/api/patients", json={"label": "BT_12-11-1963", "notes": ""}
        )
        assert created.status_code == 200, created.text

        uploaded = client.post(
            "/api/patients/bulk_upload",
            files=[("files", ("session-one.txt", b"qEEG report text", "text/plain"))],
            data={
                "identities": json.dumps(
                    [
                        {
                            "filename": "session-one.txt",
                            "first_name": "Barto",
                            "last_name": "Tinker",
                            "birthdate": "12-11-1963",
                        }
                    ]
                )
            },
        )

    assert uploaded.status_code == 200
    body = uploaded.json()
    assert body["counts"]["created"] == 1
    assert body["created"][0]["patient"]["id"] == created.json()["id"]

    with storage.session_scope() as session:
        patients = storage.list_patients(session)
    assert [patient.label for patient in patients] == ["BT_12-11-1963"]
    assert (patients[0].first_name, patients[0].last_name) == ("Barto", "Tinker")


def test_bulk_upload_failure_leaves_no_half_created_patient(
    temp_data_dir, monkeypatch
):
    """A report that cannot be read files nothing at all.

    Identity resolution commits the patient before the upload is extracted, so a
    failure after that point has to be compensated: no patient row, no empty
    portal folder that batch discovery would go on to enumerate.
    """
    app, _main = _test_app(temp_data_dir, monkeypatch)
    from backend import storage

    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.post(
            "/api/patients/bulk_upload",
            files=[("files", ("broken.pdf", b"%PDF-1.4\n", "application/pdf"))],
            data={
                "identities": json.dumps(
                    [
                        {
                            "filename": "broken.pdf",
                            "first_name": "Barto",
                            "last_name": "Tinker",
                            "birthdate": "12-11-1963",
                        }
                    ]
                )
            },
        )

    assert response.status_code == 200
    assert response.json()["counts"] == {"created": 0, "skipped": 0, "errors": 1}

    with storage.session_scope() as session:
        assert storage.list_patients(session) == []

    portal_root = Path(temp_data_dir) / "portal_patients"
    assert not portal_root.exists() or list(portal_root.iterdir()) == []


def test_generated_portal_filenames_never_carry_a_legacy_patient_key(
    temp_data_dir, monkeypatch
):
    """The filename leg of the invariant: nothing the engine writes is DOB-keyed.

    Exercises the generator rather than the classifier — every file export puts
    into the portal tree has to be named off the canonical id.
    """
    app, main = _test_app(temp_data_dir, monkeypatch)
    from backend import storage

    monkeypatch.setattr(
        main, "_schedule_portal_sync", lambda patient_label, *, source: None
    )

    with storage.session_scope() as session:
        patient = storage.create_patient(session, label="BT_12-11-1963", notes="")
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
        for stage_num, stage_name, kind, path, content_type in (
            (2, "peer_review", "peer_review", stage2_path, "application/json"),
            (3, "revision", "revision", stage3_path, "text/markdown"),
        ):
            storage.create_artifact(
                session,
                run_id=run.id,
                stage_num=stage_num,
                stage_name=stage_name,
                model_id="mock-council-a",
                kind=kind,
                content_path=path,
                content_type=content_type,
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

    assert response.status_code == 200

    portal_root = Path(temp_data_dir) / "portal_patients"
    written = [path for path in portal_root.rglob("*") if path.is_file()]
    assert written, "export published nothing to the portal tree"
    legacy_key = re.compile(r"\d{2}-\d{2}-\d{4}-\d")
    for path in written:
        assert legacy_key.search(path.name) is None, path.name
        assert path.name.startswith("BT_12-11-1963"), path.name


def test_saving_a_patient_without_notes_keeps_the_notes_on_file(
    temp_data_dir, monkeypatch
):
    """Notes are agent-managed memory, so silence is not an instruction to erase."""
    app, _main = _test_app(temp_data_dir, monkeypatch)

    with TestClient(app, raise_server_exceptions=False) as client:
        created = client.post(
            "/api/patients",
            json={
                "first_name": "Barto",
                "last_name": "Tinker",
                "birthdate": "12-11-1963",
                "notes": "sleeps badly before sessions",
            },
        ).json()

        untouched = client.put(
            f"/api/patients/{created['id']}",
            json={
                "first_name": "Barto",
                "last_name": "Tinker",
                "birthdate": "12-11-1963",
            },
        ).json()

        replaced = client.put(
            f"/api/patients/{created['id']}",
            json={
                "first_name": "Barto",
                "last_name": "Tinker",
                "birthdate": "12-11-1963",
                "notes": "sleep improved after week three",
            },
        ).json()

        cleared = client.put(
            f"/api/patients/{created['id']}",
            json={
                "first_name": "Barto",
                "last_name": "Tinker",
                "birthdate": "12-11-1963",
                "notes": "",
            },
        ).json()

    assert untouched["notes"] == "sleeps badly before sessions"
    assert replaced["notes"] == "sleep improved after week three"
    assert cleared["notes"] == ""
