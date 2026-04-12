from __future__ import annotations

from pathlib import Path
import subprocess


def test_reports_from_index_prefers_report_pdf_metadata():
    from scripts import portal_pipeline_worker as worker

    reports = worker.reports_from_index(
        "03-05-2010-0",
        {
            "files": [
                {
                    "fileKey": "03-05-2010-0__patient-facing__v1__2026-03-21.pdf",
                    "originalName": "03-05-2010-0__patient-facing__v1__2026-03-21.pdf",
                    "logicalName": "patient-facing.pdf",
                    "contentType": "application/pdf",
                    "documentKind": None,
                },
                {
                    "fileKey": "03-05-2010-0__report__v1__2026-03-21.pdf",
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
                    "fileKey": "03-05-2010-0__notes__v1__2026-03-21.txt",
                    "originalName": "notes.txt",
                    "contentType": "text/plain",
                },
            ]
        },
    )

    assert len(reports) == 1
    assert reports[0].file_key == "03-05-2010-0__report__v1__2026-03-21.pdf"
    assert reports[0].original_name == "clinic report.pdf"
    assert reports[0].document_kind == "report"


def test_reports_from_index_falls_back_to_source_pdf_heuristic():
    from scripts import portal_pipeline_worker as worker

    reports = worker.reports_from_index(
        "09-23-1982-0",
        {
            "files": [
                {
                    "fileKey": "09-23-1982-0__D_EEG_Dec_redacted__v1__2026-03-19.pdf",
                    "originalName": "D_EEG_Dec_redacted.pdf",
                    "contentType": "application/pdf",
                },
                {
                    "fileKey": "09-23-1982-0__09-23-1982-0__v1__2026-03-19.pdf",
                    "originalName": "09-23-1982-0.pdf",
                    "contentType": "application/pdf",
                },
            ]
        },
    )

    assert [report.original_name for report in reports] == ["D_EEG_Dec_redacted.pdf"]


def test_reports_from_index_does_not_filter_clinic_analysis_named_report():
    from scripts import portal_pipeline_worker as worker

    patient_id = "03-05-2010-0"
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

    patient_id = "03-05-2010-0"
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
        "03-05-2010-0",
        {
            "files": [
                {
                    "fileKey": "03-05-2010-0_DK_20Tx_toxic-brain-injury_Redacted_v1_2026-03-21.pdf",
                    "originalName": "DK_20Tx_toxic-brain-injury_Redacted.pdf",
                    "logicalName": "DK_20Tx_toxic-brain-injury_Redacted.pdf",
                    "uploadedAt": 10,
                    "size": 100,
                    "contentType": "application/pdf",
                    "documentKind": "report",
                    "uploadedBy": "clinic",
                },
                {
                    "fileKey": "03-05-2010-0__DK_20Tx_toxic-brain-injury_Redacted__v1__2026-04-12.pdf",
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

    patient_id = "03-05-2010-0"
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
        "03-05-2010-0",
        {
            "uploadedAt": 1774087200000,
            "uploadedBy": "clinic",
            "reportFiles": [
                {
                    "fileKey": "03-05-2010-0__qeeg-report__session-2026-03-21__v2__2026-03-21.pdf",
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
        == "03-05-2010-0__qeeg-report__session-2026-03-21__v2__2026-03-21.pdf"
    )


def test_completion_candidates_for_job_report_are_version_exact():
    from scripts import portal_pipeline_worker as worker

    report = worker.PortalReport(
        patient_id="03-05-2010-0",
        file_key="03-05-2010-0__qeeg-report__session-2026-03-21__v2__2026-03-21.pdf",
        original_name="same-name.pdf",
        logical_name="qeeg-report__session-2026-03-21.pdf",
        uploaded_at=1,
        size=10,
        content_type="application/pdf",
        document_kind="report",
        from_job=True,
    )

    assert worker.completion_candidate_filenames(report) == {
        "03-05-2010-0__qeeg-report__session-2026-03-21__v2__2026-03-21.pdf"
    }


def test_analysis_artifacts_exist_recognizes_council_and_patient_facing(tmp_path: Path):
    from scripts import portal_pipeline_worker as worker

    patient_dir = tmp_path / "03-05-2010-0"
    patient_dir.mkdir()
    (patient_dir / "raw.pdf").write_bytes(b"%PDF-1.4")

    assert not worker.analysis_artifacts_exist(patient_dir, "03-05-2010-0")

    council_file = patient_dir / "council" / "run-id" / "stage-4" / "gpt.md"
    council_file.parent.mkdir(parents=True)
    council_file.write_text("analysis", encoding="utf-8")

    assert worker.analysis_artifacts_exist(patient_dir, "03-05-2010-0")


def test_should_run_pipeline_when_report_downloaded(tmp_path: Path, monkeypatch):
    from scripts import portal_pipeline_worker as worker

    report = worker.PortalReport(
        patient_id="03-05-2010-0",
        file_key="03-05-2010-0__report__v1__2026-03-21.pdf",
        original_name="report.pdf",
        logical_name="report.pdf",
        uploaded_at=1,
        size=10,
        content_type="application/pdf",
        document_kind="report",
    )
    should_run, note = worker.should_run_pipeline_for_patient(
        portal_dir=tmp_path,
        patient_id="03-05-2010-0",
        reports=[report],
        downloaded=[str(tmp_path / "03-05-2010-0" / "report.pdf")],
    )

    assert should_run
    assert note == "downloaded missing report PDFs"


def test_should_not_duplicate_active_run_when_no_analysis_yet(tmp_path: Path, monkeypatch):
    from scripts import portal_pipeline_worker as worker

    report = worker.PortalReport(
        patient_id="03-05-2010-0",
        file_key="03-05-2010-0__report__v1__2026-03-21.pdf",
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
        patient_id="03-05-2010-0",
        reports=[report],
        downloaded=[],
    )

    assert not should_run
    assert note == "matching run already active for report.pdf"


def test_should_skip_when_all_reports_have_complete_runs(tmp_path: Path, monkeypatch):
    from scripts import portal_pipeline_worker as worker

    patient_id = "03-05-2010-0"
    patient_dir = tmp_path / patient_id
    patient_dir.mkdir()
    (patient_dir / f"{patient_id}.md").write_text("final", encoding="utf-8")
    report = worker.PortalReport(
        patient_id=patient_id,
        file_key="03-05-2010-0__report__v1__2026-03-21.pdf",
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
    assert note == "matching complete runs already exist for all report PDFs"


def test_should_run_when_any_report_lacks_complete_run_even_if_artifact_exists(
    tmp_path: Path, monkeypatch
):
    from scripts import portal_pipeline_worker as worker

    patient_id = "03-05-2010-0"
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
    assert note == "report PDFs without complete runs: report-two.pdf"


def test_should_run_incomplete_reports_when_another_report_is_active(
    tmp_path: Path, monkeypatch
):
    from scripts import portal_pipeline_worker as worker

    patient_id = "03-05-2010-0"
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
        "report PDFs without complete runs: b.pdf; "
        "active report(s) skipped this cycle: a.pdf"
    )


def test_merge_reports_keeps_index_reports_when_job_markers_exist():
    from scripts import portal_pipeline_worker as worker

    patient_id = "03-05-2010-0"
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

    patient_id = "03-05-2010-0"
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


def test_process_patient_runs_from_job_payload_without_index(tmp_path: Path, monkeypatch):
    from scripts import portal_pipeline_worker as worker

    class FakeClient:
        writes = []

        def get_json(self, key):
            assert key == "patients/03-05-2010-0/$index.json"
            return None

        def list_keys(self, prefix):
            assert prefix == "patients/03-05-2010-0/files/"
            return []

        def download(self, key, dest):
            assert key == (
                "patients/03-05-2010-0/files/"
                "03-05-2010-0__qeeg-report__session-2026-03-21__v2__2026-03-21.pdf"
            )
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(b"%PDF-1.4")

        def set_json(self, key, payload):
            self.writes.append((key, payload))

    job_report = worker.PortalReport(
        patient_id="03-05-2010-0",
        file_key="03-05-2010-0__qeeg-report__session-2026-03-21__v2__2026-03-21.pdf",
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
        patient_id="03-05-2010-0",
        job_reports=[job_report],
        dry_run=False,
    )

    assert result.status == "complete"
    assert result.ran_batch is True
    assert result.downloaded == [
        str(
            tmp_path
            / "portal"
            / "03-05-2010-0"
            / "03-05-2010-0__qeeg-report__session-2026-03-21__v2__2026-03-21.pdf"
        )
    ]
    assert (tmp_path / "status" / "03-05-2010-0.json").exists()


def test_process_patient_records_final_remote_status_failure(tmp_path: Path, monkeypatch):
    from scripts import portal_pipeline_worker as worker

    class FakeClient:
        def __init__(self):
            self.write_count = 0

        def get_json(self, key):
            assert key == "patients/03-05-2010-0/$index.json"
            return {
                "files": [
                    {
                        "fileKey": "03-05-2010-0__report__v1__2026-03-21.pdf",
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
            assert prefix == "patients/03-05-2010-0/files/"
            return ["patients/03-05-2010-0/files/03-05-2010-0__report__v1__2026-03-21.pdf"]

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
        patient_id="03-05-2010-0",
        job_reports=[],
        dry_run=False,
    )

    local_status = (tmp_path / "status" / "03-05-2010-0.json").read_text(
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
            assert key == "patients/03-05-2010-0/$index.json"
            return None

        def list_keys(self, prefix):
            assert prefix == "patients/03-05-2010-0/files/"
            return [
                "patients/03-05-2010-0/files/03-05-2010-0__report__v1__2026-03-21.pdf"
            ]

    monkeypatch.setattr(worker, "_matching_active_run_exists", lambda *_args, **_kwargs: False)
    monkeypatch.setattr(worker, "_matching_complete_run_exists", lambda *_args, **_kwargs: False)

    result = worker.process_patient(
        client=FakeClient(),
        portal_dir=tmp_path / "portal",
        status_dir=tmp_path / "status",
        patient_id="03-05-2010-0",
        job_reports=[],
        dry_run=True,
    )

    assert result.status == "dry_run_run"
    assert result.downloaded == []
    assert result.would_download == [
        str(
            tmp_path
            / "portal"
            / "03-05-2010-0"
            / "03-05-2010-0__report__v1__2026-03-21.pdf"
        )
    ]
    assert result.note == "would download missing report PDFs"
