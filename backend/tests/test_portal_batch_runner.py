from __future__ import annotations

from pathlib import Path

import pytest


def test_discover_batch_tasks_filters_generated_outputs(tmp_path: Path):
    from scripts import run_portal_council_batch as script

    patient_dir = tmp_path / "RS_01-01-2001"
    patient_dir.mkdir(parents=True)
    for name in (
        "real_source.pdf",
        "another-source.pdf",
        "RS_01-01-2001.pdf",
        "main.pdf",
        "RS_01-01-2001__patient-facing__v1__2026-03-17.pdf",
        "RS_01-01-2001__single-agent-5session-v1__2026-03-17__patient-facing.pdf",
        "RS_01-01-2001__analysis__v1__2026-03-17.pdf",
        "RS_01-01-2001_analysis_pdf.pdf",
        "RS_01-01-2001__guide__v1__2026-03-17.pdf",
        "guide.pdf",
        "RS_01-01-2001__analysis_report__v1__2026-03-17.pdf",
    ):
        (patient_dir / name).write_bytes(b"%PDF-1.4")

    manifest_dir = tmp_path / "MK_01-01-2013"
    manifest_dir.mkdir(parents=True)
    (manifest_dir / "combined_5sessions.manifest.json").write_text("{}", encoding="utf-8")
    (manifest_dir / "special.pdf").write_bytes(b"%PDF-1.4")

    tasks = script._discover_batch_tasks(
        portal_dir=tmp_path,
        exclude_labels=set(),
        skip_manifest_special_cases=True,
    )

    assert [task.pdf_path.name for task in tasks] == [
        "RS_01-01-2001__analysis_report__v1__2026-03-17.pdf",
        "another-source.pdf",
        "real_source.pdf",
    ]


def test_discover_batch_tasks_deduplicates_identical_sync_echoes(tmp_path: Path):
    from scripts import run_portal_council_batch as script

    patient_id = "RS_01-01-2001"
    patient_dir = tmp_path / patient_id
    patient_dir.mkdir(parents=True)
    original = patient_dir / "source-report.pdf"
    original.write_bytes(b"%PDF-1.4\nsource-a")
    for version in (1, 598):
        echo = patient_dir / (
            f"{patient_id}__source-report__v{version}__2026-08-02.pdf"
        )
        echo.write_bytes(original.read_bytes())
    legacy_echo = patient_dir / (
        f"{patient_id}_source_report_v4_2026-01-24.pdf"
    )
    legacy_echo.write_bytes(original.read_bytes())
    nested_echo = patient_dir / (
        f"{patient_id}__{patient_id}_source_report_v4_2026-01-24"
        "__v1__2026-04-12.pdf"
    )
    nested_echo.write_bytes(b"%PDF-1.4\nremote-sync-metadata-revision")
    (patient_dir / "different-session.pdf").write_bytes(
        b"%PDF-1.4\nsource-b"
    )

    tasks = script._discover_batch_tasks(
        portal_dir=tmp_path,
        exclude_labels=set(),
        skip_manifest_special_cases=True,
    )

    assert [task.pdf_path.name for task in tasks] == [
        "different-session.pdf",
        "source-report.pdf",
    ]


def test_dry_run_does_not_require_cliproxy(tmp_path: Path, temp_data_dir, monkeypatch):
    import asyncio
    import sys

    from scripts import run_portal_council_batch as script

    patient_dir = tmp_path / "RS_01-01-2001"
    patient_dir.mkdir(parents=True)
    (patient_dir / "report.pdf").write_bytes(b"%PDF-1.4")

    def fail_if_constructed(*_args, **_kwargs):
        raise AssertionError("dry-run should not construct CLIProxy client")

    monkeypatch.setattr(script, "AsyncOpenAICompatClient", fail_if_constructed)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_portal_council_batch.py",
            "--portal-dir",
            str(tmp_path),
            "--dry-run",
        ],
    )

    assert asyncio.run(script.main()) == 0


def test_batch_lock_rejects_concurrent_runs(temp_data_dir):
    from scripts import run_portal_council_batch as script

    first = script._acquire_batch_lock()
    try:
        try:
            script._acquire_batch_lock()
        except RuntimeError as exc:
            assert "Another qEEG council batch is already running" in str(exc)
        else:
            raise AssertionError("expected batch lock contention")
    finally:
        first.close()


def test_report_assets_ready_uses_report_extracted_path_not_report_id(temp_data_dir):
    from backend import storage
    from scripts import run_portal_council_batch as script

    with storage.session_scope() as session:
        patient = storage.create_patient(session, label="JP_01-19-1966", notes="")
        asset_dir = temp_data_dir / "reports" / patient.id / "upload-folder"
        pages_dir = asset_dir / "pages"
        pages_dir.mkdir(parents=True)
        stored_path = asset_dir / "original.pdf"
        extracted_path = asset_dir / "extracted.txt"
        stored_path.write_bytes(b"%PDF-1.4")
        extracted_path.write_text("raw", encoding="utf-8")
        (asset_dir / "extracted_enhanced.txt").write_text("enhanced", encoding="utf-8")
        (asset_dir / "metadata.json").write_text("{}", encoding="utf-8")
        (pages_dir / "page-1.png").write_bytes(b"png")
        report = storage.create_report(
            session,
            report_id="db-report-id",
            patient_id=patient.id,
            filename="report.pdf",
            mime_type="application/pdf",
            stored_path=stored_path,
            extracted_text_path=extracted_path,
        )

        assert script._report_assets_ready(report)


def test_latest_publishable_run_ignores_unreviewed_complete_runs(temp_data_dir):
    from backend import storage
    from scripts import run_portal_council_batch as script

    with storage.session_scope() as session:
        patient = storage.create_patient(session, label="JP_01-19-1966", notes="")
        report = storage.create_report(
            session,
            report_id="report-publishable-check",
            patient_id=patient.id,
            filename="report.pdf",
            mime_type="application/pdf",
            stored_path=temp_data_dir / "report.pdf",
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
        stage6 = temp_data_dir / "artifacts" / run.id / "stage-6" / "final.md"
        stage6.parent.mkdir(parents=True)
        stage6.write_text("# Final", encoding="utf-8")
        storage.create_artifact(
            session,
            run_id=run.id,
            stage_num=6,
            stage_name="final_draft",
            model_id="mock-council-a",
            kind="final_draft",
            content_path=stage6,
            content_type="text/markdown",
        )

    assert script._latest_publishable_run_for_report(report.id) is None


def test_patient_facing_failure_blocks_batch_publish_and_sync(
    temp_data_dir, monkeypatch
):
    import asyncio

    from backend import storage
    from scripts import run_portal_council_batch as script

    patient_label = "JP_01-19-1966"
    portal_dir = temp_data_dir / "portal" / patient_label
    portal_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = portal_dir / "report.pdf"
    pdf_path.write_bytes(b"%PDF-1.4")

    with storage.session_scope() as session:
        patient = storage.create_patient(session, label=patient_label, notes="")
        report = storage.create_report(
            session,
            report_id="report-patient-facing-failure",
            patient_id=patient.id,
            filename="report.pdf",
            mime_type="application/pdf",
            stored_path=temp_data_dir / "report.pdf",
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
        for stage_num, stage_name, kind, suffix, content_type in (
            (2, "peer_review", "peer_review", "json", "application/json"),
            (3, "revision", "revision", "md", "text/markdown"),
            (6, "final_draft", "final_draft", "md", "text/markdown"),
        ):
            artifact_path = (
                temp_data_dir
                / "artifacts"
                / run.id
                / f"stage-{stage_num}"
                / f"mock-council-a.{suffix}"
            )
            artifact_path.parent.mkdir(parents=True, exist_ok=True)
            artifact_path.write_text("# artifact", encoding="utf-8")
            storage.create_artifact(
                session,
                run_id=run.id,
                stage_num=stage_num,
                stage_name=stage_name,
                model_id="mock-council-a",
                kind=kind,
                content_path=artifact_path,
                content_type=content_type,
            )

    async def fail_patient_facing(_run_id: str) -> bool:
        return False

    exported: list[str] = []
    synced: list[str] = []
    monkeypatch.setattr(
        script, "_generate_patient_facing_outputs", fail_patient_facing
    )
    monkeypatch.setattr(script, "_export_run", lambda run_id: exported.append(run_id))
    monkeypatch.setattr(
        script, "sync_patient_to_thrylen", lambda label: synced.append(label)
    )

    outcome = asyncio.run(
        script._process_task(
            task=script.BatchTask(
                patient_label=patient_label,
                patient_dir=portal_dir,
                pdf_path=pdf_path,
            ),
            workflow=None,  # type: ignore[arg-type]
            discovered_models=["mock-council-a"],
            skip_complete=True,
            reextract_existing=False,
            dry_run=False,
        )
    )

    assert outcome.status == "failed"
    assert "Patient-facing generation did not complete" in outcome.note
    assert exported == []
    assert synced == []


def test_resolve_patient_prefers_candidate_with_matching_complete_run(temp_data_dir):
    from backend import storage
    from scripts import run_portal_council_batch as script

    portal_pdf = temp_data_dir / "portal" / "JP_01-19-1966" / "report.pdf"
    portal_pdf.parent.mkdir(parents=True, exist_ok=True)
    portal_pdf.write_bytes(b"%PDF-1.4")

    with storage.session_scope() as session:
        older = storage.create_patient(session, label="JP_01-19-1966", notes="")
        newer = storage.create_patient(session, label="JP_01-19-1966", notes="")

        older_report = storage.create_report(
            session,
            patient_id=older.id,
            filename="report.pdf",
            mime_type="application/pdf",
            stored_path=temp_data_dir / "older.pdf",
            extracted_text_path=temp_data_dir / "older.txt",
        )
        newer_report = storage.create_report(
            session,
            patient_id=newer.id,
            filename="report.pdf",
            mime_type="application/pdf",
            stored_path=temp_data_dir / "newer.pdf",
            extracted_text_path=temp_data_dir / "newer.txt",
        )

        storage.create_run(
            session,
            patient_id=older.id,
            report_id=older_report.id,
            council_model_ids=["m1"],
            consolidator_model_id="m1",
        )
        failed_run = storage.create_run(
            session,
            patient_id=newer.id,
            report_id=newer_report.id,
            council_model_ids=["m1"],
            consolidator_model_id="m1",
        )
        storage.update_run_status(session, failed_run.id, status="failed", error_message="boom")
        complete_run = storage.create_run(
            session,
            patient_id=older.id,
            report_id=older_report.id,
            council_model_ids=["m1"],
            consolidator_model_id="m1",
        )
        storage.update_run_status(session, complete_run.id, status="complete")

        chosen = script._resolve_patient_for_label(
            session,
            patient_label="JP_01-19-1966",
            portal_pdfs=[portal_pdf],
        )

    assert chosen.id == older.id


def test_choose_existing_report_prefers_exact_filename_over_wrong_row(temp_data_dir):
    from backend import storage
    from scripts import run_portal_council_batch as script

    with storage.session_scope() as session:
        patient = storage.create_patient(session, label="CD_09-23-1982", notes="")
        wrong = storage.create_report(
            session,
            patient_id=patient.id,
            filename="wrong.pdf",
            mime_type="application/pdf",
            stored_path=temp_data_dir / "wrong.pdf",
            extracted_text_path=temp_data_dir / "wrong.txt",
        )
        right = storage.create_report(
            session,
            patient_id=patient.id,
            filename="expected.pdf",
            mime_type="application/pdf",
            stored_path=temp_data_dir / "expected.pdf",
            extracted_text_path=temp_data_dir / "expected.txt",
        )
        storage.create_run(
            session,
            patient_id=patient.id,
            report_id=wrong.id,
            council_model_ids=["m1"],
            consolidator_model_id="m1",
        )
        good_run = storage.create_run(
            session,
            patient_id=patient.id,
            report_id=right.id,
            council_model_ids=["m1"],
            consolidator_model_id="m1",
        )
        storage.update_run_status(session, good_run.id, status="complete")

        chosen = script._choose_existing_report(
            session,
            patient_id=patient.id,
            filename="expected.pdf",
        )

    assert chosen is not None
    assert chosen.id == right.id


def test_pick_model_id_requires_exact_gpt54_family_match():
    from scripts import run_portal_council_batch as script

    discovered = ["gpt-5", "claude-sonnet-4-6", "gemini-3.1-flash-lite-preview"]

    assert script._pick_model_id("gpt-5.4", discovered) is None


def test_resolve_model_selection_prefers_current_configured_models(temp_data_dir, monkeypatch):
    from backend import storage
    from backend.config import CouncilModelConfig
    from scripts import run_portal_council_batch as script

    monkeypatch.setattr(
        script,
        "COUNCIL_MODELS",
        [
            CouncilModelConfig(id="gpt-5.5", name="GPT-5.5", source="test"),
            CouncilModelConfig(id="claude-sonnet-4-6", name="Claude", source="test"),
            CouncilModelConfig(
                id="gemini-3.1-flash-lite-preview", name="Gemini", source="test"
            ),
        ],
    )
    monkeypatch.setattr(script, "DEFAULT_CONSOLIDATOR", "gpt-5.5")

    with storage.session_scope() as session:
        patient = storage.create_patient(session, label="DF_02-04-1988", notes="")
        report = storage.create_report(
            session,
            patient_id=patient.id,
            filename="report.pdf",
            mime_type="application/pdf",
            stored_path=temp_data_dir / "report.pdf",
            extracted_text_path=temp_data_dir / "report.txt",
        )
        old_run = storage.create_run(
            session,
            patient_id=patient.id,
            report_id=report.id,
            council_model_ids=["gpt-5.2", "claude-opus-4-6"],
            consolidator_model_id="claude-opus-4-6",
        )
        storage.update_run_status(session, old_run.id, status="complete")

        council_model_ids, consolidator = script._resolve_model_selection_for_run(
            session,
            patient_id=patient.id,
            discovered_models=[
                "gpt-5.5",
                "claude-sonnet-4-6",
                "gemini-3.1-flash-lite-preview",
            ],
        )

    assert council_model_ids == [
        "gpt-5.5",
        "claude-sonnet-4-6",
        "gemini-3.1-flash-lite-preview",
    ]
    assert consolidator == "gpt-5.5"


def test_resolve_model_selection_excludes_opus_by_default(temp_data_dir, monkeypatch):
    from backend import storage
    from backend.config import CouncilModelConfig
    from scripts import run_portal_council_batch as script

    monkeypatch.delenv("QEEG_ALLOW_OPUS_MODELS", raising=False)
    monkeypatch.setattr(
        script,
        "COUNCIL_MODELS",
        [
            CouncilModelConfig(id="claude-opus-4-6", name="Opus", source="test"),
            CouncilModelConfig(id="claude-sonnet-4-6", name="Sonnet", source="test"),
        ],
    )
    monkeypatch.setattr(script, "DEFAULT_CONSOLIDATOR", "claude-opus-4-6")

    with storage.session_scope() as session:
        patient = storage.create_patient(session, label="KM_02-05-1987", notes="")
        council_model_ids, consolidator = script._resolve_model_selection_for_run(
            session,
            patient_id=patient.id,
            discovered_models=["claude-opus-4-6", "claude-sonnet-4-6"],
        )

    assert council_model_ids == ["claude-sonnet-4-6"]
    assert consolidator == "claude-sonnet-4-6"


def test_resolve_model_selection_raises_when_only_disallowed_models_exist(
    temp_data_dir, monkeypatch
):
    from backend import storage
    from backend.config import CouncilModelConfig
    from scripts import run_portal_council_batch as script

    monkeypatch.delenv("QEEG_ALLOW_OPUS_MODELS", raising=False)
    monkeypatch.setattr(
        script,
        "COUNCIL_MODELS",
        [
            CouncilModelConfig(id="claude-opus-4-6", name="Opus", source="test"),
        ],
    )
    monkeypatch.setattr(script, "DEFAULT_CONSOLIDATOR", "claude-opus-4-6")

    with storage.session_scope() as session:
        patient = storage.create_patient(session, label="KM_02-05-1987", notes="")
        with pytest.raises(RuntimeError, match="Configured council role models"):
            script._resolve_model_selection_for_run(
                session,
                patient_id=patient.id,
                discovered_models=["claude-opus-4-6"],
            )


def test_resolve_model_selection_never_revives_historical_batch_models(
    temp_data_dir, monkeypatch
):
    from backend import storage
    from backend.config import CouncilModelConfig
    from scripts import run_portal_council_batch as script

    monkeypatch.setattr(
        script,
        "COUNCIL_MODELS",
        [
            CouncilModelConfig(
                id="openai/gpt-5.6-sol",
                name="GPT-5.6 Sol",
                source="test",
            )
        ],
    )
    monkeypatch.setattr(script, "DEFAULT_CONSOLIDATOR", "openai/gpt-5.6-sol")

    with storage.session_scope() as session:
        patient = storage.create_patient(session, label="KM_02-05-1987", notes="")
        report = storage.create_report(
            session,
            patient_id=patient.id,
            filename="report.pdf",
            mime_type="application/pdf",
            stored_path=temp_data_dir / "report.pdf",
            extracted_text_path=temp_data_dir / "report.txt",
        )
        storage.create_run(
            session,
            patient_id=patient.id,
            report_id=report.id,
            council_model_ids=["claude-sonnet-4-6"],
            consolidator_model_id="claude-sonnet-4-6",
        )

        with pytest.raises(RuntimeError, match="Configured council role models"):
            script._resolve_model_selection_for_run(
                session,
                patient_id=patient.id,
                discovered_models=["claude-sonnet-4-6"],
            )


def test_latest_resume_candidate_prefers_recent_incomplete_run(temp_data_dir):
    from backend import storage
    from scripts import run_portal_council_batch as script

    with storage.session_scope() as session:
        patient = storage.create_patient(session, label="WX_12-15-2002", notes="")
        report = storage.create_report(
            session,
            patient_id=patient.id,
            filename="report.pdf",
            mime_type="application/pdf",
            stored_path=temp_data_dir / "report.pdf",
            extracted_text_path=temp_data_dir / "report.txt",
        )
        first = storage.create_run(
            session,
            patient_id=patient.id,
            report_id=report.id,
            council_model_ids=["m1"],
            consolidator_model_id="m1",
        )
        second = storage.create_run(
            session,
            patient_id=patient.id,
            report_id=report.id,
            council_model_ids=["m1"],
            consolidator_model_id="m1",
        )
        storage.claim_run_start(session, second.id)
        storage.update_run_status(session, first.id, status="failed", error_message="boom")

    resume = script._latest_resume_candidate_run_for_report(report.id)

    assert resume is not None
    assert resume.id == second.id


@pytest.mark.parametrize("run_status", ["created", "running", "failed", "needs_auth"])
def test_latest_resume_candidate_accepts_every_restartable_status(
    temp_data_dir, run_status
):
    from backend import storage
    from scripts import run_portal_council_batch as script

    with storage.session_scope() as session:
        patient = storage.create_patient(
            session,
            label=f"resume-{run_status}",
            notes="",
        )
        report = storage.create_report(
            session,
            patient_id=patient.id,
            filename=f"{run_status}.pdf",
            mime_type="application/pdf",
            stored_path=temp_data_dir / f"{run_status}.pdf",
            extracted_text_path=temp_data_dir / f"{run_status}.txt",
        )
        run = storage.create_run(
            session,
            patient_id=patient.id,
            report_id=report.id,
            council_model_ids=["m1", "m2"],
            consolidator_model_id="writer",
        )
        if run_status == "running":
            storage.claim_run_start(session, run.id)
        elif run_status in {"failed", "needs_auth"}:
            storage.update_run_status(
                session,
                run.id,
                status=run_status,
                error_message="retryable upstream failure",
            )

    resume = script._latest_resume_candidate_run_for_report(report.id)

    assert resume is not None
    assert resume.id == run.id


def test_force_mode_does_not_resume_incomplete_run(temp_data_dir):
    from backend import storage
    from scripts import run_portal_council_batch as script

    with storage.session_scope() as session:
        patient = storage.create_patient(session, label="PQ_10-31-2008", notes="")
        report = storage.create_report(
            session,
            patient_id=patient.id,
            filename="report.pdf",
            mime_type="application/pdf",
            stored_path=temp_data_dir / "report.pdf",
            extracted_text_path=temp_data_dir / "report.txt",
        )
        run = storage.create_run(
            session,
            patient_id=patient.id,
            report_id=report.id,
            council_model_ids=["old-model"],
            consolidator_model_id="old-model",
        )
        storage.claim_run_start(session, run.id)

    assert script._resume_candidate_for_report(report.id, skip_complete=True) is not None
    assert script._resume_candidate_for_report(report.id, skip_complete=False) is None


def test_dry_run_force_does_not_claim_it_would_resume(temp_data_dir):
    import asyncio

    from backend import storage
    from scripts import run_portal_council_batch as script

    patient_label = "PQ_10-31-2008"
    portal_dir = temp_data_dir / patient_label
    portal_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = portal_dir / "report.pdf"
    pdf_path.write_bytes(b"%PDF-1.4")

    with storage.session_scope() as session:
        patient = storage.create_patient(session, label=patient_label, notes="")
        report = storage.create_report(
            session,
            patient_id=patient.id,
            filename="report.pdf",
            mime_type="application/pdf",
            stored_path=temp_data_dir / "report.pdf",
            extracted_text_path=temp_data_dir / "report.txt",
        )
        run = storage.create_run(
            session,
            patient_id=patient.id,
            report_id=report.id,
            council_model_ids=["old-model"],
            consolidator_model_id="old-model",
        )
        storage.claim_run_start(session, run.id)

    outcome = asyncio.run(
        script._process_task(
            task=script.BatchTask(
                patient_label=patient_label,
                patient_dir=portal_dir,
                pdf_path=pdf_path,
            ),
            workflow=None,  # type: ignore[arg-type]
            discovered_models=["gpt-5.4"],
            skip_complete=False,
            reextract_existing=False,
            dry_run=True,
        )
    )

    assert outcome.status == "dry_run"
    assert outcome.run_id is None


def test_format_progress_event_includes_tail_friendly_details():
    from scripts import run_portal_council_batch as script

    line = script._format_progress_event(
        {
            "stage_num": 1,
            "stage_name": "Initial Analyses",
            "task": "data_pack_chunk",
            "model_id": "gemini-3.1-pro-preview",
            "status": "heartbeat",
            "chunk_index": 2,
            "chunk_count": 4,
            "pages": [9, 10, 11],
            "elapsed_s": 60,
            "heartbeat_count": 2,
        }
    )

    assert line == (
        "stage=1:Initial Analyses heartbeat task=data_pack_chunk "
        "model_id=gemini-3.1-pro-preview chunk=2/4 pages=9,10,11 "
        "elapsed_s=60 heartbeat_count=2"
    )


def test_discover_batch_tasks_skips_legacy_dob_folders(tmp_path: Path):
    """A pre-cutover folder name is not a patient id, so it plans no paid work."""
    from scripts import run_portal_council_batch as script

    legacy_dir = tmp_path / "01-01-2001-0"
    legacy_dir.mkdir(parents=True)
    (legacy_dir / "source.pdf").write_bytes(b"%PDF-1.4")

    canonical_dir = tmp_path / "BT_12-11-1963"
    canonical_dir.mkdir(parents=True)
    (canonical_dir / "source.pdf").write_bytes(b"%PDF-1.4")

    tasks = script._discover_batch_tasks(
        portal_dir=tmp_path,
        exclude_labels=set(),
        skip_manifest_special_cases=True,
    )

    assert [task.patient_label for task in tasks] == ["BT_12-11-1963"]
