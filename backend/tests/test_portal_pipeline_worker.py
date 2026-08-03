from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest


def test_sync_remote_patient_identity_writes_versioned_local_metadata(tmp_path: Path):
    from scripts import portal_pipeline_worker as worker

    patient_id = "AB_03-05-2010"
    patient_dir = tmp_path / patient_id
    patient_dir.mkdir()

    result = worker.sync_remote_patient_identity(
        portal_dir=tmp_path,
        patient_id=patient_id,
        remote_meta={
            "identity": {
                "schemaVersion": 1,
                "firstInitial": "a",
                "lastInitial": "b",
            }
        },
    )

    stored = json.loads((patient_dir / "$meta.json").read_text(encoding="utf-8"))
    assert result == {"schemaVersion": 1, "firstInitial": "A", "lastInitial": "B"}
    assert stored["patientId"] == patient_id
    assert stored["birthdate"] == "03-05-2010"
    # The ordinal is the canonical one, so an unsuffixed id writes 1. The legacy
    # MM-DD-YYYY-N world started this field at 0; that contract is retired.
    assert stored["index"] == 1
    assert stored["identity"] == result
    assert "name" not in json.dumps(stored).lower()


def test_sync_remote_patient_identity_rejects_conflicting_local_initials(tmp_path: Path):
    from scripts import portal_pipeline_worker as worker

    patient_id = "AB_03-05-2010"
    patient_dir = tmp_path / patient_id
    patient_dir.mkdir()
    (patient_dir / "$meta.json").write_text(
        json.dumps(
            {
                "identity": {
                    "schemaVersion": 1,
                    "firstInitial": "A",
                    "lastInitial": "B",
                }
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="conflict"):
        worker.sync_remote_patient_identity(
            portal_dir=tmp_path,
            patient_id=patient_id,
            remote_meta={
                "identity": {
                    "schemaVersion": 1,
                    "firstInitial": "C",
                    "lastInitial": "D",
                }
            },
        )


def test_reports_from_index_prefers_report_pdf_metadata():
    from scripts import portal_pipeline_worker as worker

    reports = worker.reports_from_index(
        "AB_03-05-2010",
        {
            "files": [
                {
                    "fileKey": "AB_03-05-2010__patient-facing__v1__2026-03-21.pdf",
                    "originalName": "AB_03-05-2010__patient-facing__v1__2026-03-21.pdf",
                    "logicalName": "patient-facing.pdf",
                    "contentType": "application/pdf",
                    "documentKind": None,
                },
                {
                    "fileKey": "AB_03-05-2010__report__v1__2026-03-21.pdf",
                    "originalName": "clinic report.pdf",
                    "logicalName": "qeeg-report__session-2026-03-21.pdf",
                    "uploadedAt": 2,
                    "size": 2048,
                    "contentType": "application/pdf",
                    "documentKind": "report",
                    "reportBirthdate": "03-05-2010",
                    "sessionDate": "2026-03-21",
                },
                {
                    "fileKey": "AB_03-05-2010__notes__v1__2026-03-21.txt",
                    "originalName": "notes.txt",
                    "contentType": "text/plain",
                },
            ]
        },
    )

    assert len(reports) == 1
    assert reports[0].file_key == "AB_03-05-2010__report__v1__2026-03-21.pdf"
    assert reports[0].original_name == "clinic report.pdf"
    assert reports[0].document_kind == "report"


def test_reports_from_index_falls_back_to_source_pdf_heuristic():
    from scripts import portal_pipeline_worker as worker

    reports = worker.reports_from_index(
        "CD_09-23-1982",
        {
            "files": [
                {
                    "fileKey": "CD_09-23-1982__D_EEG_Dec_redacted__v1__2026-03-19.pdf",
                    "originalName": "D_EEG_Dec_redacted.pdf",
                    "contentType": "application/pdf",
                },
                {
                    "fileKey": "CD_09-23-1982__CD_09-23-1982__v1__2026-03-19.pdf",
                    "originalName": "CD_09-23-1982.pdf",
                    "contentType": "application/pdf",
                },
            ]
        },
    )

    assert [report.original_name for report in reports] == ["D_EEG_Dec_redacted.pdf"]


def test_reports_from_index_does_not_filter_clinic_analysis_named_report():
    from scripts import portal_pipeline_worker as worker

    patient_id = "AB_03-05-2010"
    reports = worker.reports_from_index(
        patient_id,
        {
            "files": [
                {
                    "fileKey": f"{patient_id}__analysis_report__v1__2026-03-21.pdf",
                    "originalName": "analysis_report.pdf",
                    "contentType": "application/pdf",
                    "documentKind": "report",
                    "uploadedBy": "clinic",
                },
                {
                    "fileKey": f"{patient_id}__analysis__v1__2026-03-21.pdf",
                    "originalName": "generated analysis.pdf",
                    "contentType": "application/pdf",
                    "uploadedBy": "local-sync",
                },
            ]
        },
    )

    assert len(reports) == 1
    assert reports[0].file_key == f"{patient_id}__analysis_report__v1__2026-03-21.pdf"


def test_reports_from_index_uses_file_key_as_single_report_identity():
    from scripts import portal_pipeline_worker as worker

    patient_id = "AB_03-05-2010"
    reports = worker.reports_from_index(
        patient_id,
        {
            "files": [
                {
                    "fileKey": f"{patient_id}__qeeg-report__session-2026-03-21__v2__2026-03-21__upload-b.pdf",
                    "originalName": "same-session-second.pdf",
                    "logicalName": "qeeg-report__session-2026-03-21.pdf",
                    "uploadedAt": 2,
                    "size": 101,
                    "contentType": "application/pdf",
                    "documentKind": "report",
                    "uploadedBy": "clinic",
                },
            ]
        },
    )

    assert len(reports) == 1
    assert worker.source_local_filename(reports[0]) == (
        f"{patient_id}__qeeg-report__session-2026-03-21__v2__2026-03-21__upload-b.pdf"
    )
    assert worker.completion_candidate_filenames(reports[0]) == {
        f"{patient_id}__qeeg-report__session-2026-03-21__v2__2026-03-21__upload-b.pdf"
    }


def test_reports_from_index_deduplicates_local_sync_echoes():
    from scripts import portal_pipeline_worker as worker

    reports = worker.reports_from_index(
        "AB_03-05-2010",
        {
            "files": [
                {
                    "fileKey": "AB_03-05-2010_DK_20Tx_toxic-brain-injury_Redacted_v1_2026-03-21.pdf",
                    "originalName": "DK_20Tx_toxic-brain-injury_Redacted.pdf",
                    "logicalName": "DK_20Tx_toxic-brain-injury_Redacted.pdf",
                    "uploadedAt": 10,
                    "size": 100,
                    "contentType": "application/pdf",
                    "documentKind": "report",
                    "uploadedBy": "clinic",
                },
                {
                    "fileKey": "AB_03-05-2010__DK_20Tx_toxic-brain-injury_Redacted__v1__2026-04-12.pdf",
                    "originalName": "DK_20Tx_toxic-brain-injury_Redacted.pdf",
                    "logicalName": "DK_20Tx_toxic-brain-injury_Redacted.pdf",
                    "uploadedAt": 20,
                    "size": 100,
                    "uploadedBy": "local-sync",
                },
            ]
        },
    )

    assert len(reports) == 1
    assert reports[0].uploaded_by == "clinic"


def test_reports_from_index_keeps_same_session_versions_distinct():
    from scripts import portal_pipeline_worker as worker

    patient_id = "AB_03-05-2010"
    reports = worker.reports_from_index(
        patient_id,
        {
            "files": [
                {
                    "fileKey": f"{patient_id}__qeeg-report__session-2026-03-21__v1__2026-03-21.pdf",
                    "originalName": "same-session-first.pdf",
                    "logicalName": "qeeg-report__session-2026-03-21.pdf",
                    "uploadedAt": 1,
                    "size": 100,
                    "contentType": "application/pdf",
                    "documentKind": "report",
                    "uploadedBy": "clinic",
                },
                {
                    "fileKey": f"{patient_id}__qeeg-report__session-2026-03-21__v2__2026-03-21.pdf",
                    "originalName": "same-session-second.pdf",
                    "logicalName": "qeeg-report__session-2026-03-21.pdf",
                    "uploadedAt": 2,
                    "size": 101,
                    "contentType": "application/pdf",
                    "documentKind": "report",
                    "uploadedBy": "clinic",
                },
            ]
        },
    )

    assert len(reports) == 2
    assert [worker.source_local_filename(report) for report in reports] == [
        f"{patient_id}__qeeg-report__session-2026-03-21__v1__2026-03-21.pdf",
        f"{patient_id}__qeeg-report__session-2026-03-21__v2__2026-03-21.pdf",
    ]


def test_reports_from_job_payload_uses_versioned_file_key_as_local_source_name():
    from scripts import portal_pipeline_worker as worker

    reports = worker.reports_from_job_payload(
        "AB_03-05-2010",
        {
            "uploadedAt": 1774087200000,
            "uploadedBy": "clinic",
            "reportFiles": [
                {
                    "fileKey": "AB_03-05-2010__qeeg-report__session-2026-03-21__v2__2026-03-21.pdf",
                    "originalName": "same-name.pdf",
                    "logicalName": "qeeg-report__session-2026-03-21.pdf",
                    "contentType": "application/pdf",
                    "documentKind": "report",
                    "size": 4096,
                }
            ],
        },
    )

    assert len(reports) == 1
    assert reports[0].from_job is True
    assert (
        worker.source_local_filename(reports[0])
        == "AB_03-05-2010__qeeg-report__session-2026-03-21__v2__2026-03-21.pdf"
    )


def test_completion_candidates_for_job_report_are_version_exact():
    from scripts import portal_pipeline_worker as worker

    report = worker.PortalReport(
        patient_id="AB_03-05-2010",
        file_key="AB_03-05-2010__qeeg-report__session-2026-03-21__v2__2026-03-21.pdf",
        original_name="same-name.pdf",
        logical_name="qeeg-report__session-2026-03-21.pdf",
        uploaded_at=1,
        size=10,
        content_type="application/pdf",
        document_kind="report",
        from_job=True,
    )

    assert worker.completion_candidate_filenames(report) == {
        "AB_03-05-2010__qeeg-report__session-2026-03-21__v2__2026-03-21.pdf"
    }


def test_analysis_artifacts_exist_recognizes_council_and_patient_facing(tmp_path: Path):
    from scripts import portal_pipeline_worker as worker

    patient_dir = tmp_path / "AB_03-05-2010"
    patient_dir.mkdir()
    (patient_dir / "raw.pdf").write_bytes(b"%PDF-1.4")

    assert not worker.analysis_artifacts_exist(patient_dir, "AB_03-05-2010")

    council_file = patient_dir / "council" / "run-id" / "stage-4" / "gpt.md"
    council_file.parent.mkdir(parents=True)
    council_file.write_text("analysis", encoding="utf-8")

    assert worker.analysis_artifacts_exist(patient_dir, "AB_03-05-2010")


def test_should_run_pipeline_when_report_downloaded(temp_data_dir, tmp_path: Path, monkeypatch):
    from scripts import portal_pipeline_worker as worker

    report = worker.PortalReport(
        patient_id="AB_03-05-2010",
        file_key="AB_03-05-2010__report__v1__2026-03-21.pdf",
        original_name="report.pdf",
        logical_name="report.pdf",
        uploaded_at=1,
        size=10,
        content_type="application/pdf",
        document_kind="report",
    )
    should_run, note = worker.should_run_pipeline_for_patient(
        portal_dir=tmp_path,
        patient_id="AB_03-05-2010",
        reports=[report],
        downloaded=[str(tmp_path / "AB_03-05-2010" / "report.pdf")],
    )

    assert should_run
    assert note == "downloaded missing report PDFs"


def test_should_not_duplicate_active_run_when_no_analysis_yet(tmp_path: Path, monkeypatch):
    from scripts import portal_pipeline_worker as worker

    report = worker.PortalReport(
        patient_id="AB_03-05-2010",
        file_key="AB_03-05-2010__report__v1__2026-03-21.pdf",
        original_name="report.pdf",
        logical_name="report.pdf",
        uploaded_at=1,
        size=10,
        content_type="application/pdf",
        document_kind="report",
    )
    monkeypatch.setattr(worker, "_matching_active_run_exists", lambda *_args, **_kwargs: True)

    should_run, note = worker.should_run_pipeline_for_patient(
        portal_dir=tmp_path,
        patient_id="AB_03-05-2010",
        reports=[report],
        downloaded=[],
    )

    assert not should_run
    assert note == "matching run already active for report.pdf"


def test_should_skip_when_all_reports_have_complete_runs(temp_data_dir, tmp_path: Path, monkeypatch):
    from scripts import portal_pipeline_worker as worker

    patient_id = "AB_03-05-2010"
    patient_dir = tmp_path / patient_id
    patient_dir.mkdir()
    (patient_dir / f"{patient_id}.md").write_text("final", encoding="utf-8")
    report = worker.PortalReport(
        patient_id=patient_id,
        file_key="AB_03-05-2010__report__v1__2026-03-21.pdf",
        original_name="report.pdf",
        logical_name="report.pdf",
        uploaded_at=1,
        size=10,
        content_type="application/pdf",
        document_kind="report",
    )
    monkeypatch.setattr(worker, "_matching_complete_run_exists", lambda *_args, **_kwargs: True)

    should_run, note = worker.should_run_pipeline_for_patient(
        portal_dir=tmp_path,
        patient_id=patient_id,
        reports=[report],
        downloaded=[],
    )

    assert not should_run
    assert note == "matching delivery-ready runs already exist for all report PDFs"


def test_matching_complete_run_requires_delivery_ready_artifacts(temp_data_dir):
    from backend import storage
    from scripts import portal_pipeline_worker as worker

    patient_label = "AB_03-05-2010"
    report_name = "report.pdf"
    with storage.session_scope() as session:
        patient = storage.create_patient(session, label=patient_label, notes="")
        report_dir = Path(temp_data_dir) / "reports" / patient.id / "report-1"
        report_dir.mkdir(parents=True)
        stored_path = report_dir / "original.pdf"
        extracted_path = report_dir / "extracted.txt"
        stored_path.write_bytes(b"%PDF-1.4")
        extracted_path.write_text("extracted", encoding="utf-8")
        report = storage.create_report(
            session,
            report_id="report-1",
            patient_id=patient.id,
            filename=report_name,
            mime_type="application/pdf",
            stored_path=stored_path,
            extracted_text_path=extracted_path,
        )
        run = storage.create_run(
            session,
            patient_id=patient.id,
            report_id=report.id,
            council_model_ids=["mock-council-a"],
            consolidator_model_id="mock-consolidator",
        )
        storage.update_run_status(session, run.id, status="complete")

    assert not worker._matching_complete_run_exists(patient_label, {report_name})

    with storage.session_scope() as session:
        artifact_dir = Path(temp_data_dir) / "artifacts" / run.id
        for stage, name, text in (
            ("stage-2", "review.json", "{}"),
            ("stage-3", "revision.md", "# Revision"),
            ("stage-6", "final.md", "# Final"),
        ):
            path = artifact_dir / stage / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="utf-8")
        storage.create_artifact(
            session,
            run_id=run.id,
            stage_num=2,
            stage_name="peer_review",
            model_id="mock-council-a",
            kind="peer_review",
            content_path=artifact_dir / "stage-2" / "review.json",
            content_type="application/json",
        )
        storage.create_artifact(
            session,
            run_id=run.id,
            stage_num=3,
            stage_name="revision",
            model_id="mock-council-a",
            kind="revision",
            content_path=artifact_dir / "stage-3" / "revision.md",
            content_type="text/markdown",
        )
        storage.create_artifact(
            session,
            run_id=run.id,
            stage_num=6,
            stage_name="final_draft",
            model_id="mock-council-a",
            kind="final_draft",
            content_path=artifact_dir / "stage-6" / "final.md",
            content_type="text/markdown",
        )

    assert worker._matching_complete_run_exists(patient_label, {report_name})


def test_should_run_when_any_report_lacks_complete_run_even_if_artifact_exists(
    tmp_path: Path, monkeypatch
):
    from scripts import portal_pipeline_worker as worker

    patient_id = "AB_03-05-2010"
    patient_dir = tmp_path / patient_id
    patient_dir.mkdir()
    (patient_dir / f"{patient_id}.md").write_text("old final", encoding="utf-8")
    reports = [
        worker.PortalReport(
            patient_id=patient_id,
            file_key=f"{patient_id}__report-one__v1__2026-03-21.pdf",
            original_name="report-one.pdf",
            logical_name="report-one.pdf",
            uploaded_at=1,
            size=10,
            content_type="application/pdf",
            document_kind="report",
        ),
        worker.PortalReport(
            patient_id=patient_id,
            file_key=f"{patient_id}__report-two__v1__2026-03-21.pdf",
            original_name="report-two.pdf",
            logical_name="report-two.pdf",
            uploaded_at=2,
            size=10,
            content_type="application/pdf",
            document_kind="report",
        ),
    ]

    def fake_complete(_patient_id, filenames):
        return "report-one.pdf" in filenames

    monkeypatch.setattr(worker, "_matching_complete_run_exists", fake_complete)

    should_run, note = worker.should_run_pipeline_for_patient(
        portal_dir=tmp_path,
        patient_id=patient_id,
        reports=reports,
        downloaded=[],
    )

    assert should_run
    assert note == "report PDFs without delivery-ready runs: report-two.pdf"


def test_should_run_incomplete_reports_when_another_report_is_active(
    tmp_path: Path, monkeypatch
):
    from scripts import portal_pipeline_worker as worker

    patient_id = "AB_03-05-2010"
    reports = [
        worker.PortalReport(
            patient_id=patient_id,
            file_key=f"{patient_id}__a__v1__2026-03-21.pdf",
            original_name="a.pdf",
            logical_name="a.pdf",
            uploaded_at=1,
            size=10,
            content_type="application/pdf",
            document_kind="report",
        ),
        worker.PortalReport(
            patient_id=patient_id,
            file_key=f"{patient_id}__b__v1__2026-03-21.pdf",
            original_name="b.pdf",
            logical_name="b.pdf",
            uploaded_at=2,
            size=10,
            content_type="application/pdf",
            document_kind="report",
        ),
    ]

    monkeypatch.setattr(
        worker,
        "_matching_active_run_exists",
        lambda _patient_id, filename: filename == "a.pdf",
    )
    monkeypatch.setattr(worker, "_matching_complete_run_exists", lambda *_args, **_kwargs: False)

    should_run, note = worker.should_run_pipeline_for_patient(
        portal_dir=tmp_path,
        patient_id=patient_id,
        reports=reports,
        downloaded=[],
    )

    assert should_run
    assert note == (
        "report PDFs without delivery-ready runs: b.pdf; "
        "active report(s) skipped this cycle: a.pdf"
    )


def test_merge_reports_keeps_index_reports_when_job_markers_exist():
    from scripts import portal_pipeline_worker as worker

    patient_id = "AB_03-05-2010"
    job_report = worker.PortalReport(
        patient_id=patient_id,
        file_key=f"{patient_id}__report-one__v1__2026-03-21.pdf",
        original_name="report-one.pdf",
        logical_name="report-one.pdf",
        uploaded_at=1,
        size=10,
        content_type="application/pdf",
        document_kind="report",
        from_job=True,
    )
    index_reports = [
        worker.PortalReport(
            patient_id=patient_id,
            file_key=f"{patient_id}__report-one__v1__2026-03-21.pdf",
            original_name="report-one.pdf",
            logical_name="report-one.pdf",
            uploaded_at=1,
            size=10,
            content_type="application/pdf",
            document_kind="report",
        ),
        worker.PortalReport(
            patient_id=patient_id,
            file_key=f"{patient_id}__report-two__v1__2026-03-21.pdf",
            original_name="report-two.pdf",
            logical_name="report-two.pdf",
            uploaded_at=2,
            size=10,
            content_type="application/pdf",
            document_kind="report",
        ),
    ]

    merged = worker.merge_reports(job_reports=[job_report], index_reports=index_reports)

    assert [report.file_key for report in merged] == [
        f"{patient_id}__report-one__v1__2026-03-21.pdf",
        f"{patient_id}__report-two__v1__2026-03-21.pdf",
    ]
    assert merged[0].from_job is True


def test_reports_from_file_keys_recovers_pdf_blobs_without_index():
    from scripts import portal_pipeline_worker as worker

    patient_id = "AB_03-05-2010"
    reports = worker.reports_from_file_keys(
        patient_id,
        [
            f"patients/{patient_id}/files/{patient_id}__raw-report__v1__2026-03-21.pdf",
            f"patients/{patient_id}/files/{patient_id}__patient-facing__v1__2026-03-21.pdf",
        ],
    )

    assert len(reports) == 1
    assert reports[0].file_key == f"{patient_id}__raw-report__v1__2026-03-21.pdf"
    assert worker.source_local_filename(reports[0]) == reports[0].file_key


def test_matching_active_run_exists_keeps_fresh_created_row_active(temp_data_dir):
    from backend import storage
    from scripts import portal_pipeline_worker as worker

    patient_label = "AB_03-05-2010"
    filename = "AB_03-05-2010__report__v1__2026-03-21.pdf"
    with storage.session_scope() as session:
        patient = storage.create_patient(session, label=patient_label, notes="")
        report = storage.create_report(
            session,
            patient_id=patient.id,
            filename=filename,
            mime_type="application/pdf",
            stored_path=temp_data_dir / "report.pdf",
            extracted_text_path=temp_data_dir / "report.txt",
        )
        storage.create_run(
            session,
            patient_id=patient.id,
            report_id=report.id,
            council_model_ids=["gpt-5.4"],
            consolidator_model_id="claude-sonnet-4-6",
        )

    assert worker._matching_active_run_exists(patient_label, filename) is True


def test_matching_active_run_exists_ignores_stale_running_row(temp_data_dir, monkeypatch):
    from backend import storage
    from backend.orchestration import progress_jsonl_path
    from scripts import portal_pipeline_worker as worker

    monkeypatch.setenv("QEEG_RUN_STALE_AFTER_S", "300")

    patient_label = "AB_03-05-2010"
    filename = "AB_03-05-2010__report__v1__2026-03-21.pdf"
    with storage.session_scope() as session:
        patient = storage.create_patient(session, label=patient_label, notes="")
        report = storage.create_report(
            session,
            patient_id=patient.id,
            filename=filename,
            mime_type="application/pdf",
            stored_path=temp_data_dir / "report.pdf",
            extracted_text_path=temp_data_dir / "report.txt",
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
                "status": "heartbeat",
                "timestamp": "2026-04-12T10:00:00Z",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    assert worker._matching_active_run_exists(patient_label, filename) is False


def test_process_patient_runs_from_job_payload_without_index(tmp_path: Path, monkeypatch):
    from scripts import portal_pipeline_worker as worker

    class FakeClient:
        writes = []

        def get_json(self, key):
            assert key == "patients/AB_03-05-2010/$index.json"
            return None

        def list_keys(self, prefix):
            assert prefix == "patients/AB_03-05-2010/files/"
            return []

        def download(self, key, dest):
            assert key == (
                "patients/AB_03-05-2010/files/"
                "AB_03-05-2010__qeeg-report__session-2026-03-21__v2__2026-03-21.pdf"
            )
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(b"%PDF-1.4")

        def set_json(self, key, payload):
            self.writes.append((key, payload))

    job_report = worker.PortalReport(
        patient_id="AB_03-05-2010",
        file_key="AB_03-05-2010__qeeg-report__session-2026-03-21__v2__2026-03-21.pdf",
        original_name="same-name.pdf",
        logical_name="qeeg-report__session-2026-03-21.pdf",
        uploaded_at=1,
        size=8,
        content_type="application/pdf",
        document_kind="report",
        from_job=True,
    )
    monkeypatch.setattr(worker, "_matching_active_run_exists", lambda *_args, **_kwargs: False)
    monkeypatch.setattr(worker, "_matching_complete_run_exists", lambda *_args, **_kwargs: False)
    monkeypatch.setattr(
        worker,
        "run_batch_for_patient",
        lambda *args, **kwargs: subprocess.CompletedProcess(args=args, returncode=0, stdout="ok", stderr=""),
    )

    result = worker.process_patient(
        client=FakeClient(),
        portal_dir=tmp_path / "portal",
        status_dir=tmp_path / "status",
        patient_id="AB_03-05-2010",
        job_reports=[job_report],
        dry_run=False,
        allow_paid_runs=True,
    )

    assert result.status == "complete"
    assert result.ran_batch is True
    assert result.downloaded == [
        str(
            tmp_path
            / "portal"
            / "AB_03-05-2010"
            / "AB_03-05-2010__qeeg-report__session-2026-03-21__v2__2026-03-21.pdf"
        )
    ]
    assert (tmp_path / "status" / "AB_03-05-2010.json").exists()


@pytest.mark.parametrize("from_job", [False, True])
def test_process_patient_never_starts_unapproved_paid_work(
    tmp_path: Path, monkeypatch, from_job: bool
):
    from scripts import portal_pipeline_worker as worker

    patient_id = "AB_03-05-2010"
    file_key = f"{patient_id}__report__v1__2026-03-21.pdf"

    class FakeClient:
        def get_json(self, key):
            if key == f"patients/{patient_id}/$meta.json":
                return None
            if key == f"patients/{patient_id}/$index.json":
                if from_job:
                    return None
                return {
                    "files": [
                        {
                            "fileKey": file_key,
                            "originalName": "report.pdf",
                            "logicalName": "report.pdf",
                            "uploadedAt": 1,
                            "size": 8,
                            "contentType": "application/pdf",
                            "documentKind": "report",
                        }
                    ]
                }
            raise AssertionError(key)

        def list_keys(self, prefix):
            assert prefix == f"patients/{patient_id}/files/"
            return [] if from_job else [f"{prefix}{file_key}"]

        def download(self, key, dest):
            assert key == f"patients/{patient_id}/files/{file_key}"
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(b"%PDF-1.4")

        def set_json(self, key, payload):
            pass

    job_reports = []
    if from_job:
        job_reports = [
            worker.PortalReport(
                patient_id=patient_id,
                file_key=file_key,
                original_name="report.pdf",
                logical_name="report.pdf",
                uploaded_at=1,
                size=8,
                content_type="application/pdf",
                document_kind="report",
                from_job=True,
            )
        ]

    paid_calls = []
    monkeypatch.setattr(worker, "_matching_active_run_exists", lambda *_args, **_kwargs: False)
    monkeypatch.setattr(worker, "_matching_complete_run_exists", lambda *_args, **_kwargs: False)
    monkeypatch.setattr(
        worker,
        "run_batch_for_patient",
        lambda *args, **kwargs: paid_calls.append((args, kwargs)),
    )

    result = worker.process_patient(
        client=FakeClient(),
        portal_dir=tmp_path / "portal",
        status_dir=tmp_path / "status",
        patient_id=patient_id,
        job_reports=job_reports,
        dry_run=False,
    )

    assert result.status == "awaiting_confirmation"
    assert result.ran_batch is False
    assert "explicit approval" in result.note
    assert paid_calls == []


@pytest.mark.parametrize(
    ("once", "include_labels", "expected"),
    [
        (False, set(), False),
        (True, set(), True),
        (False, {"AB_03-05-2010"}, True),
    ],
)
def test_patient_discovery_scope_separates_continuous_jobs_from_manual_audits(
    once: bool, include_labels: set[str], expected: bool
):
    from scripts import portal_pipeline_worker as worker

    assert worker.should_discover_all_patients(
        once=once, include_labels=include_labels
    ) is expected


@pytest.mark.parametrize(
    ("include_labels", "allow_paid_runs", "expected"),
    [
        (set(), False, False),
        ({"AB_03-05-2010"}, False, True),
        (set(), True, True),
    ],
)
def test_paid_work_requires_patient_selection_or_explicit_flag(
    include_labels: set[str], allow_paid_runs: bool, expected: bool
):
    from scripts import portal_pipeline_worker as worker

    assert worker.paid_runs_are_authorized(
        include_labels=include_labels, allow_paid_runs=allow_paid_runs
    ) is expected


def test_process_patient_records_final_remote_status_failure(tmp_path: Path, monkeypatch):
    from scripts import portal_pipeline_worker as worker

    class FakeClient:
        def __init__(self):
            self.write_count = 0

        def get_json(self, key):
            assert key == "patients/AB_03-05-2010/$index.json"
            return {
                "files": [
                    {
                        "fileKey": "AB_03-05-2010__report__v1__2026-03-21.pdf",
                        "originalName": "report.pdf",
                        "logicalName": "report.pdf",
                        "uploadedAt": 1,
                        "size": 8,
                        "contentType": "application/pdf",
                        "documentKind": "report",
                    }
                ]
            }

        def list_keys(self, prefix):
            assert prefix == "patients/AB_03-05-2010/files/"
            return ["patients/AB_03-05-2010/files/AB_03-05-2010__report__v1__2026-03-21.pdf"]

        def download(self, key, dest):
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(b"%PDF-1.4")

        def set_json(self, key, payload):
            self.write_count += 1
            if self.write_count == 2:
                raise RuntimeError("netlify write failed")

    monkeypatch.setattr(worker, "_matching_active_run_exists", lambda *_args, **_kwargs: False)
    monkeypatch.setattr(worker, "_matching_complete_run_exists", lambda *_args, **_kwargs: False)
    monkeypatch.setattr(
        worker,
        "run_batch_for_patient",
        lambda *args, **kwargs: subprocess.CompletedProcess(args=args, returncode=0, stdout="ok", stderr=""),
    )

    result = worker.process_patient(
        client=FakeClient(),
        portal_dir=tmp_path / "portal",
        status_dir=tmp_path / "status",
        patient_id="AB_03-05-2010",
        job_reports=[],
        dry_run=False,
        allow_paid_runs=True,
    )

    local_status = (tmp_path / "status" / "AB_03-05-2010.json").read_text(
        encoding="utf-8"
    )
    assert result.status == "complete"
    assert "remote status publish failed: netlify write failed" in result.note
    assert "remote status publish failed: netlify write failed" in local_status


def test_process_patient_dry_run_reports_would_download_without_claiming_download(
    tmp_path: Path, monkeypatch
):
    from scripts import portal_pipeline_worker as worker

    class FakeClient:
        def get_json(self, key):
            assert key == "patients/AB_03-05-2010/$index.json"
            return None

        def list_keys(self, prefix):
            assert prefix == "patients/AB_03-05-2010/files/"
            return [
                "patients/AB_03-05-2010/files/AB_03-05-2010__report__v1__2026-03-21.pdf"
            ]

    monkeypatch.setattr(worker, "_matching_active_run_exists", lambda *_args, **_kwargs: False)
    monkeypatch.setattr(worker, "_matching_complete_run_exists", lambda *_args, **_kwargs: False)

    result = worker.process_patient(
        client=FakeClient(),
        portal_dir=tmp_path / "portal",
        status_dir=tmp_path / "status",
        patient_id="AB_03-05-2010",
        job_reports=[],
        dry_run=True,
    )

    assert result.status == "dry_run_run"
    assert result.downloaded == []
    assert result.would_download == [
        str(
            tmp_path
            / "portal"
            / "AB_03-05-2010"
            / "AB_03-05-2010__report__v1__2026-03-21.pdf"
        )
    ]
    assert result.note == "would download missing report PDFs"


def test_portal_keys_route_only_on_canonical_ids():
    """Blob, job, and status keys all carry the canonical clinic id, never a DOB."""
    from scripts import portal_pipeline_worker as worker

    assert worker.is_valid_patient_id("BT_12-11-1963") is True
    assert worker.is_valid_patient_id("BT_12-11-1963_2") is True
    assert worker.is_valid_patient_id("03-05-2010-0") is False
    assert worker.is_valid_patient_id("BT_12-11-1963_1") is False

    assert (
        worker.patient_id_from_meta_key("patients/BT_12-11-1963/$meta.json")
        == "BT_12-11-1963"
    )
    assert worker.patient_id_from_meta_key("patients/03-05-2010-0/$meta.json") is None
    assert (
        worker.patient_id_from_job_key("pipeline/jobs/BT_12-11-1963/job-1.json")
        == "BT_12-11-1963"
    )
    assert worker.patient_id_from_job_key("pipeline/jobs/03-05-2010-0/job-1.json") is None
    assert (
        worker.patient_id_from_file_key("patients/BT_12-11-1963/files/report.pdf")
        == "BT_12-11-1963"
    )
    assert (
        worker.patient_id_from_file_key("patients/03-05-2010-0/files/report.pdf") is None
    )


def test_sync_remote_patient_identity_reads_the_dob_out_of_the_canonical_id(
    tmp_path: Path,
):
    """The id is the only source of the date of birth and the collision ordinal."""
    from scripts import portal_pipeline_worker as worker

    patient_id = "AB_03-05-2010_2"
    (tmp_path / patient_id).mkdir()

    worker.sync_remote_patient_identity(
        portal_dir=tmp_path,
        patient_id=patient_id,
        remote_meta={
            "identity": {"schemaVersion": 1, "firstInitial": "A", "lastInitial": "B"}
        },
    )

    stored = json.loads((tmp_path / patient_id / "$meta.json").read_text(encoding="utf-8"))
    assert stored["birthdate"] == "03-05-2010"
    assert stored["index"] == 2


def test_sync_remote_patient_identity_rejects_initials_the_id_contradicts(
    tmp_path: Path,
):
    """Hub initials that disagree with the id would be a second source of truth."""
    from scripts import portal_pipeline_worker as worker

    patient_id = "AB_03-05-2010"
    (tmp_path / patient_id).mkdir()

    with pytest.raises(ValueError, match="conflict"):
        worker.sync_remote_patient_identity(
            portal_dir=tmp_path,
            patient_id=patient_id,
            remote_meta={
                "identity": {"schemaVersion": 1, "firstInitial": "C", "lastInitial": "D"}
            },
        )


def test_worker_refuses_an_include_label_that_is_not_a_clinic_id(
    tmp_path: Path, monkeypatch, capsys
):
    """Dropping it silently would widen the run to the whole portal."""
    from scripts import portal_pipeline_worker as worker

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "portal_pipeline_worker.py",
            "--once",
            "--include-label",
            "bt_12-11-1963",
            "--portal-dir",
            str(tmp_path / "portal"),
            "--status-dir",
            str(tmp_path / "status"),
        ],
    )

    assert worker.main() == 2
    assert "bt_12-11-1963" in capsys.readouterr().err


# --------------------------------------------------- hub new-patient uploads


def _fake_save_report_upload(temp_data_dir):
    """Stand in for extraction, which has its own tests and needs a real PDF.

    Writes the same two files the real one does so the stored paths on the
    report row point at something.
    """

    def _save(*, patient_id, report_id, filename, provided_mime_type, file_bytes, **_):
        folder = Path(temp_data_dir) / "reports" / patient_id / report_id
        folder.mkdir(parents=True, exist_ok=True)
        stored = folder / filename
        extracted = folder / "extracted.txt"
        stored.write_bytes(file_bytes)
        extracted.write_text("extracted text", encoding="utf-8")
        return stored, extracted, provided_mime_type, "extracted text"

    return _save


# The hub names the file by its FULL blob key, not a bare filename. That shape
# is pinned against the hub's own payload builder by
# test_the_upload_job_matches_what_the_hub_actually_builds below, so this
# literal cannot drift away from the thing that produces it.
UPLOAD_ID = "up-1"
REPORT_FILE_KEY = f"uploads/pending/{UPLOAD_ID}/scan.pdf"


def _upload_job(**resolution):
    return {
        "kind": "new_patient_upload",
        "uploadId": UPLOAD_ID,
        "fileKey": REPORT_FILE_KEY,
        "identity": {
            "firstName": "Barto",
            "lastName": "Tinker",
            "firstInitial": "B",
            "lastInitial": "T",
            "birthdate": "12-11-1963",
        },
        "resolution": resolution,
        "uploadedAt": 1,
        "uploadedBy": "hub",
    }


class _UploadClient:
    """Stands in for the blob store, recording what the worker asked it to do."""

    def __init__(self, jobs, blobs=None):
        self._jobs = dict(jobs)
        # Pending upload blobs: key -> bytes. A submission can carry several.
        self._blobs = dict(blobs or {})
        self.downloads = []
        self.deleted = []
        self.written = {}

    def list_keys(self, prefix):
        return [
            key
            for key in [*self._jobs, *self._blobs]
            if key.startswith(prefix)
        ]

    def get_json(self, key):
        return self._jobs.get(key)

    def download(self, key, dest):
        self.downloads.append(key)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(self._blobs.get(key, b"report bytes"))

    def set_json(self, key, payload):
        self.written[key] = payload
        self._jobs[key] = payload

    def delete(self, key):
        self.deleted.append(key)
        self._jobs.pop(key, None)
        self._blobs.pop(key, None)


def _no_paid_work(monkeypatch, worker):
    monkeypatch.setattr(
        worker,
        "run_batch_for_patient",
        lambda *a, **k: pytest.fail("registering a patient must not run paid analysis"),
    )


def test_new_patient_upload_allocates_the_chart_and_files_the_report(
    temp_data_dir, tmp_path: Path, monkeypatch
):
    """A hub upload with no patient yet gets one, for free."""
    from backend import storage
    from scripts import portal_pipeline_worker as worker

    _no_paid_work(monkeypatch, worker)
    monkeypatch.setattr(
        worker, "save_report_upload", _fake_save_report_upload(temp_data_dir)
    )
    job_key = "pipeline/jobs/up-1/upload.json"
    client = _UploadClient({job_key: _upload_job()})

    jobs = worker.load_new_patient_upload_jobs(client)
    assert [key for key, _ in jobs] == [job_key]

    result = worker.process_new_patient_upload(
        client=client,
        portal_dir=tmp_path / "portal",
        status_dir=tmp_path / "status",
        job_key=job_key,
        payload=jobs[0][1],
    )

    assert result.status == "registered"
    assert result.patient_id == "BT_12-11-1963"
    assert result.ran_batch is False

    with storage.session_scope() as session:
        patients = storage.list_patients(session)
        reports = storage.list_reports(session, patients[0].id)
    assert [p.label for p in patients] == ["BT_12-11-1963"]
    assert [r.filename for r in reports] == ["scan.pdf"]
    assert (tmp_path / "portal" / "BT_12-11-1963" / "scan.pdf").exists()

    # The pending blob and the marker only go once the report is durable.
    assert client.deleted == ["uploads/pending/up-1/scan.pdf", job_key]

    # A crash between registering and cleaning up replays the marker. The
    # filename lookup has to find the report rather than file a second one.
    replay = worker.process_new_patient_upload(
        client=client,
        portal_dir=tmp_path / "portal",
        status_dir=tmp_path / "status",
        job_key=job_key,
        payload=_upload_job(),
    )
    assert replay.status == "already_registered"
    with storage.session_scope() as session:
        patients = storage.list_patients(session)
        assert [p.label for p in patients] == ["BT_12-11-1963"]
        assert len(storage.list_reports(session, patients[0].id)) == 1


def test_new_patient_upload_force_new_takes_the_next_ordinal(
    temp_data_dir, tmp_path: Path, monkeypatch
):
    """Two real people sharing initials and a birthday, resolved by the hub."""
    from backend import storage
    from scripts import portal_pipeline_worker as worker

    _no_paid_work(monkeypatch, worker)
    monkeypatch.setattr(
        worker, "save_report_upload", _fake_save_report_upload(temp_data_dir)
    )
    with storage.session_scope() as session:
        storage.create_patient(
            session,
            label="BT_12-11-1963",
            notes="",
            first_name="Bella",
            last_name="Turner",
            birthdate="12-11-1963",
            first_initial="B",
            last_initial="T",
        )

    job_key = "pipeline/jobs/up-1/upload.json"
    client = _UploadClient({job_key: _upload_job(forceNew=True)})
    result = worker.process_new_patient_upload(
        client=client,
        portal_dir=tmp_path / "portal",
        status_dir=tmp_path / "status",
        job_key=job_key,
        payload=_upload_job(forceNew=True),
    )

    assert result.patient_id == "BT_12-11-1963_2"
    assert result.ran_batch is False
    with storage.session_scope() as session:
        labels = sorted(p.label for p in storage.list_patients(session))
    assert labels == ["BT_12-11-1963", "BT_12-11-1963_2"]


def test_new_patient_upload_without_a_resolution_parks_and_is_skipped_next_cycle(
    temp_data_dir, tmp_path: Path, monkeypatch
):
    """An unanswered name conflict waits for the operator without spinning."""
    from backend import storage
    from scripts import portal_pipeline_worker as worker

    _no_paid_work(monkeypatch, worker)
    with storage.session_scope() as session:
        storage.create_patient(
            session,
            label="BT_12-11-1963",
            notes="",
            first_name="Bella",
            last_name="Turner",
            birthdate="12-11-1963",
            first_initial="B",
            last_initial="T",
        )

    job_key = "pipeline/jobs/up-1/upload.json"
    client = _UploadClient({job_key: _upload_job()})
    result = worker.process_new_patient_upload(
        client=client,
        portal_dir=tmp_path / "portal",
        status_dir=tmp_path / "status",
        job_key=job_key,
        payload=_upload_job(),
    )

    assert result.status == "needs_operator_answer"
    assert result.ran_batch is False
    parked = client.written[job_key]
    assert parked["status"] == "needs_operator_answer"
    assert parked["conflict"]["candidates"] == [
        {"patient_id": "BT_12-11-1963", "name": "Bella Turner"}
    ]
    # Nothing downloaded, nothing deleted: the file and the marker both stay put.
    assert client.downloads == []
    assert client.deleted == []

    # Second cycle: the parked marker is skipped without re-reading the blob.
    assert worker.load_new_patient_upload_jobs(client) == []
    assert client.downloads == []

    with storage.session_scope() as session:
        assert [p.label for p in storage.list_patients(session)] == ["BT_12-11-1963"]


# -------------------------------------------- the parked-upload queue surface


def _uploads_app(temp_data_dir, monkeypatch):
    monkeypatch.setenv("QEEG_MOCK_LLM", "1")
    monkeypatch.setenv("QEEG_PORTAL_RAW_SYNC_WATCHER", "0")
    from fastapi.testclient import TestClient

    from backend import main

    monkeypatch.setattr(
        main, "_ensure_project_clipr_config", lambda: Path(temp_data_dir) / "c.conf"
    )
    monkeypatch.setattr(main, "_sync_home_auth_to_project", lambda: 0)
    return TestClient(main.app, raise_server_exceptions=False)


def _park_an_upload(temp_data_dir, tmp_path, monkeypatch, worker):
    from backend import storage

    with storage.session_scope() as session:
        storage.create_patient(
            session,
            label="BT_12-11-1963",
            notes="",
            first_name="Bella",
            last_name="Turner",
            birthdate="12-11-1963",
            first_initial="B",
            last_initial="T",
        )
    job_key = "pipeline/jobs/up-1/upload.json"
    client = _UploadClient({job_key: _upload_job()})
    worker.process_new_patient_upload(
        client=client,
        portal_dir=tmp_path / "portal",
        status_dir=tmp_path / "status",
        job_key=job_key,
        payload=_upload_job(),
    )
    return client, job_key


def test_a_parked_upload_is_listed_with_its_conflict_candidates(
    temp_data_dir, tmp_path: Path, monkeypatch
):
    """An upload nobody can see is an upload nobody can rescue."""
    from scripts import portal_pipeline_worker as worker

    _no_paid_work(monkeypatch, worker)
    _park_an_upload(temp_data_dir, tmp_path, monkeypatch, worker)

    with _uploads_app(temp_data_dir, monkeypatch) as api:
        listed = api.get("/api/pipeline/uploads")

    assert listed.status_code == 200
    body = listed.json()
    assert [row["uploadId"] for row in body] == ["up-1"]
    assert body[0]["status"] == "needs_operator_answer"
    assert body[0]["identity"]["firstName"] == "Barto"
    assert body[0]["conflict"]["candidates"] == [
        {"patient_id": "BT_12-11-1963", "name": "Bella Turner"}
    ]


def test_resolving_an_unknown_upload_is_a_404(temp_data_dir, monkeypatch):
    with _uploads_app(temp_data_dir, monkeypatch) as api:
        response = api.post(
            "/api/pipeline/uploads/nope/resolution", json={"force_new": True}
        )

    assert response.status_code == 404


@pytest.mark.parametrize(
    ("resolution", "expected_label"),
    [
        ({"attach_to": "BT_12-11-1963"}, "BT_12-11-1963"),
        ({"force_new": True}, "BT_12-11-1963_2"),
    ],
)
def test_an_answered_upload_is_filed_on_the_next_worker_cycle(
    resolution, expected_label, temp_data_dir, tmp_path: Path, monkeypatch
):
    """The operator answers; the worker files it through the same path."""
    from backend import storage
    from scripts import portal_pipeline_worker as worker

    _no_paid_work(monkeypatch, worker)
    monkeypatch.setattr(
        worker, "save_report_upload", _fake_save_report_upload(temp_data_dir)
    )
    client, job_key = _park_an_upload(temp_data_dir, tmp_path, monkeypatch, worker)

    with _uploads_app(temp_data_dir, monkeypatch) as api:
        answered = api.post("/api/pipeline/uploads/up-1/resolution", json=resolution)
    assert answered.status_code == 200, answered.text
    assert answered.json()["status"] == "pending"

    # Next cycle: the parked marker is picked up again, now carrying the answer.
    jobs = worker.load_new_patient_upload_jobs(client)
    assert [key for key, _ in jobs] == [job_key]
    result = worker.process_new_patient_upload(
        client=client,
        portal_dir=tmp_path / "portal",
        status_dir=tmp_path / "status",
        job_key=job_key,
        payload=jobs[0][1],
    )

    assert result.patient_id == expected_label
    assert result.ran_batch is False
    with storage.session_scope() as session:
        patient = next(
            p for p in storage.list_patients(session) if p.label == expected_label
        )
        assert [r.filename for r in storage.list_reports(session, patient.id)] == [
            "scan.pdf"
        ]

    # Answering again after it is filed says so rather than failing.
    with _uploads_app(temp_data_dir, monkeypatch) as api:
        again = api.post("/api/pipeline/uploads/up-1/resolution", json=resolution)
    assert again.status_code == 200
    assert again.json()["status"] == "registered"
    assert again.json()["patient_id"] == expected_label


def test_one_broken_upload_does_not_abandon_the_rest_of_the_cycle(
    temp_data_dir, tmp_path: Path, monkeypatch
):
    """A failing upload records and steps aside; the queue keeps moving."""
    from backend import pipeline_uploads, storage
    from scripts import portal_pipeline_worker as worker

    _no_paid_work(monkeypatch, worker)
    monkeypatch.setattr(
        worker, "save_report_upload", _fake_save_report_upload(temp_data_dir)
    )

    # Each job names a key under its own upload prefix, as the hub builds it.
    broken = {
        **_upload_job(),
        "uploadId": "up-broken",
        "fileKey": "uploads/pending/up-broken/scan.pdf",
    }
    healthy = {
        **_upload_job(),
        "uploadId": "up-ok",
        "fileKey": "uploads/pending/up-ok/scan.pdf",
        "identity": {
            "firstName": "Cara",
            "lastName": "Dale",
            "firstInitial": "C",
            "lastInitial": "D",
            "birthdate": "04-02-1975",
        },
    }

    class _HalfBrokenClient(_UploadClient):
        def download(self, key, dest):
            if "up-broken" in key:
                raise RuntimeError("netlify blobs:get exited 1")
            return super().download(key, dest)

    client = _HalfBrokenClient(
        {
            "pipeline/jobs/up-broken/upload.json": broken,
            "pipeline/jobs/up-ok/upload.json": healthy,
        }
    )

    results = [
        worker.process_new_patient_upload(
            client=client,
            portal_dir=tmp_path / "portal",
            status_dir=tmp_path / "status",
            job_key=key,
            payload=payload,
        )
        for key, payload in worker.load_new_patient_upload_jobs(client)
    ]

    by_upload = {result.patient_id: result for result in results}
    assert by_upload["up-broken"].status == "failed"
    assert "netlify blobs:get exited 1" in by_upload["up-broken"].note
    assert by_upload["CD_04-02-1975"].status == "registered"

    # The failure is visible on the queue surface, not just swallowed.
    failed = pipeline_uploads.read_upload("up-broken")
    assert failed["status"] == "failed"
    assert "netlify blobs:get exited 1" in failed["error"]

    # Only the healthy upload was cleaned up; the broken one keeps its pending
    # blob and its marker so the next cycle retries it.
    assert client.deleted == [
        "uploads/pending/up-ok/scan.pdf",
        "pipeline/jobs/up-ok/upload.json",
    ]

    # The broken upload's chart was allocated before the download failed, and it
    # stays: the retry matches it by name and files the report there instead of
    # burning a second ordinal on the same person.
    with storage.session_scope() as session:
        labels = sorted(p.label for p in storage.list_patients(session))
        assert labels == ["BT_12-11-1963", "CD_04-02-1975"]
        barto = next(p for p in storage.list_patients(session) if p.label == "BT_12-11-1963")
        assert storage.list_reports(session, barto.id) == []


def test_a_malformed_upload_id_fails_that_job_without_raising(
    temp_data_dir, tmp_path: Path, monkeypatch
):
    """An id that cannot name a file is that job's problem, not the cycle's."""
    from scripts import portal_pipeline_worker as worker

    _no_paid_work(monkeypatch, worker)
    result = worker.process_new_patient_upload(
        client=_UploadClient({}),
        portal_dir=tmp_path / "portal",
        status_dir=tmp_path / "status",
        job_key="pipeline/jobs/../upload.json",
        payload={**_upload_job(), "uploadId": "../../etc/passwd"},
    )

    assert result.status == "failed"
    assert "uploadId" in result.note


def test_the_whole_submission_is_filed_not_just_the_report(
    temp_data_dir, tmp_path: Path, monkeypatch
):
    """A submission is a report plus whatever else the clinic sent with it."""
    from backend import storage
    from scripts import portal_pipeline_worker as worker

    _no_paid_work(monkeypatch, worker)
    monkeypatch.setattr(
        worker, "save_report_upload", _fake_save_report_upload(temp_data_dir)
    )
    job_key = "pipeline/jobs/new-patient/1712345678-up-1.json"
    client = _UploadClient(
        {job_key: _upload_job()},
        blobs={
            "uploads/pending/up-1/scan.pdf": b"%PDF-1.4 report",
            "uploads/pending/up-1/intake-form.pdf": b"%PDF-1.4 intake",
        },
    )

    # The hub's real key shape carries no patient id, so enumeration has to find
    # it by payload rather than by parsing the key.
    jobs = worker.load_new_patient_upload_jobs(client)
    assert [key for key, _ in jobs] == [job_key]

    result = worker.process_new_patient_upload(
        client=client,
        portal_dir=tmp_path / "portal",
        status_dir=tmp_path / "status",
        job_key=job_key,
        payload=jobs[0][1],
    )

    assert result.status == "registered"
    assert "intake-form.pdf" in result.note

    with storage.session_scope() as session:
        patient = storage.list_patients(session)[0]
        reports = storage.list_reports(session, patient.id)
        files = storage.list_patient_files(session, patient.id)
    assert patient.label == "BT_12-11-1963"
    assert [r.filename for r in reports] == ["scan.pdf"]
    assert [f.filename for f in files] == ["intake-form.pdf"]
    assert (tmp_path / "portal" / "BT_12-11-1963" / "intake-form.pdf").exists()

    # Everything pending is gone, marker last.
    assert client.deleted == [
        "uploads/pending/up-1/scan.pdf",
        "uploads/pending/up-1/intake-form.pdf",
        job_key,
    ]

    # Replaying the marker files nothing twice.
    replay = worker.process_new_patient_upload(
        client=client,
        portal_dir=tmp_path / "portal",
        status_dir=tmp_path / "status",
        job_key=job_key,
        payload=_upload_job(),
    )
    assert replay.status == "already_registered"
    with storage.session_scope() as session:
        patients = storage.list_patients(session)
        assert len(patients) == 1
        assert len(storage.list_reports(session, patients[0].id)) == 1
        assert len(storage.list_patient_files(session, patients[0].id)) == 1


def test_one_bad_extra_file_keeps_the_report_and_the_other_files(
    temp_data_dir, tmp_path: Path, monkeypatch
):
    """One unreadable attachment must not cost the report or its siblings."""
    from backend import storage
    from scripts import portal_pipeline_worker as worker

    _no_paid_work(monkeypatch, worker)
    monkeypatch.setattr(
        worker, "save_report_upload", _fake_save_report_upload(temp_data_dir)
    )
    job_key = "pipeline/jobs/new-patient/1712345678-up-1.json"

    class _OneBadFileClient(_UploadClient):
        fail_download = True

        def download(self, key, dest):
            if self.fail_download and key.endswith("broken.pdf"):
                raise RuntimeError("netlify blobs:get exited 1")
            return super().download(key, dest)

    client = _OneBadFileClient(
        {job_key: _upload_job()},
        blobs={
            "uploads/pending/up-1/scan.pdf": b"%PDF-1.4 report",
            "uploads/pending/up-1/aaa-good.pdf": b"%PDF-1.4 good",
            "uploads/pending/up-1/broken.pdf": b"%PDF-1.4 broken",
        },
    )

    result = worker.process_new_patient_upload(
        client=client,
        portal_dir=tmp_path / "portal",
        status_dir=tmp_path / "status",
        job_key=job_key,
        payload=_upload_job(),
    )

    assert result.status == "registered"
    assert "broken.pdf" in result.note

    with storage.session_scope() as session:
        patient = storage.list_patients(session)[0]
        assert [r.filename for r in storage.list_reports(session, patient.id)] == [
            "scan.pdf"
        ]
        assert [
            f.filename for f in storage.list_patient_files(session, patient.id)
        ] == ["aaa-good.pdf"]

    # Nothing under the pending prefix is dropped while any of the submission
    # is still outstanding. Deleting the report first left the next cycle
    # re-downloading a blob that was already gone — an exception every cycle
    # forever, with the upload recorded as failed even though the report had
    # registered.
    assert client.deleted == []
    assert job_key not in client.deleted

    # So the retry can actually recover: the report blob is still readable, the
    # report is not registered twice, and the file that failed now lands.
    client.fail_download = False
    retry = worker.process_new_patient_upload(
        client=client,
        portal_dir=tmp_path / "portal",
        status_dir=tmp_path / "status",
        job_key=job_key,
        payload=_upload_job(),
    )

    assert retry.status in ("registered", "already_registered")
    with storage.session_scope() as session:
        patient = storage.list_patients(session)[0]
        assert [r.filename for r in storage.list_reports(session, patient.id)] == [
            "scan.pdf"
        ]
        assert sorted(
            f.filename for f in storage.list_patient_files(session, patient.id)
        ) == ["aaa-good.pdf", "broken.pdf"]
    # Only once everything has landed does the submission get cleared.
    assert sorted(client.deleted) == sorted(
        [
            "uploads/pending/up-1/scan.pdf",
            "uploads/pending/up-1/aaa-good.pdf",
            "uploads/pending/up-1/broken.pdf",
            job_key,
        ]
    )


# --------------------------------------------------------------------------- #
# The engine/hub job contract
# --------------------------------------------------------------------------- #


def _thrylen_repo() -> Path | None:
    """The hub checkout whose payload builder defines this contract."""
    import os

    candidates = [os.getenv("QEEG_THRYLEN_REPO", "")]
    here = Path(__file__).resolve().parents[2]
    candidates += [
        str(here.parent / "thrylen"),
        str(Path.home() / ".sdd-worktrees" / "thrylen"),
        str(Path.home() / "thrylen"),
    ]
    for candidate in candidates:
        if not candidate:
            continue
        path = Path(candidate)
        if (path / "netlify" / "functions" / "qeeg-upload.js").is_file():
            return path
    return None


def test_the_upload_job_matches_what_the_hub_actually_builds():
    """Both sides had a green suite against contradictory fixtures.

    The hub emitted the full blob key and the worker expected a bare filename,
    so in production every new-patient upload looked for
    `uploads/pending/<id>/uploads/pending/<id>/<name>`, found nothing, and
    failed every cycle forever. Two hand-written fixtures cannot catch that, so
    this one runs the hub's real builder and checks the engine's fixture
    against its actual output.
    """
    import json
    import shutil
    import subprocess

    repo = _thrylen_repo()
    if repo is None:
        pytest.skip("no thrylen checkout to read the hub's payload builder from")
    if shutil.which("node") is None:
        pytest.skip("node is needed to run the hub's payload builder")

    script = """
    import { buildPendingUploadKey, buildNewPatientJobPayload }
      from './netlify/functions/qeeg-upload.js';
    const [uploadId, filename] = process.argv.slice(-2);
    const fileKey = buildPendingUploadKey({ uploadId, filename });
    process.stdout.write(JSON.stringify(buildNewPatientJobPayload({
      uploadId,
      fileKey,
      identity: {
        firstName: 'Barto', lastName: 'Tinker',
        firstInitial: 'B', lastInitial: 'T', birthdate: '12-11-1963',
      },
      resolution: {},
      uploadedAt: 1,
      uploadedBy: 'hub',
    })));
    """
    result = subprocess.run(
        ["node", "--input-type=module", "-e", script, "--", UPLOAD_ID, "scan.pdf"],
        cwd=repo, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    from_the_hub = json.loads(result.stdout)

    ours = _upload_job()

    # The field the two sides disagreed about.
    assert from_the_hub["fileKey"] == REPORT_FILE_KEY
    assert from_the_hub["fileKey"].startswith(f"uploads/pending/{UPLOAD_ID}/")
    # And the rest of the payload the worker reads.
    assert from_the_hub["kind"] == ours["kind"]
    assert from_the_hub["uploadId"] == ours["uploadId"]
    assert from_the_hub["identity"] == ours["identity"]
    assert from_the_hub["resolution"] == ours["resolution"]
    assert set(from_the_hub) == set(ours)


def test_a_job_naming_a_key_outside_its_upload_prefix_is_refused(
    temp_data_dir, tmp_path: Path, monkeypatch
):
    """Reaching for a key the hub could not have written is worth saying out
    loud rather than failing on a download that was never going to resolve."""
    from scripts import portal_pipeline_worker as worker

    _no_paid_work(monkeypatch, worker)
    job = _upload_job()
    job["fileKey"] = "scan.pdf"  # the shape the worker used to assume
    client = _UploadClient({"pipeline/jobs/new-patient/1-up-1.json": job})

    result = worker.process_new_patient_upload(
        client=client,
        portal_dir=tmp_path / "portal",
        status_dir=tmp_path / "status",
        job_key="pipeline/jobs/new-patient/1-up-1.json",
        payload=job,
    )

    assert result.status == "failed"
    assert "uploads/pending/up-1/" in result.note
    assert client.downloads == []
    assert client.deleted == []
