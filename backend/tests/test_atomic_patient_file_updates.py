from __future__ import annotations

import io
import asyncio
import os
from pathlib import Path

import pytest


@pytest.fixture
def ready_run(temp_data_dir):
    from backend import storage
    from backend.tests.test_main_invariants import _create_report

    with storage.session_scope() as session:
        patient = storage.create_patient(session, label="ZZ_01-01-1900", notes="")
    report = _create_report(storage, temp_data_dir, patient_id=patient.id)
    with storage.session_scope() as session:
        run = storage.create_run(
            session,
            patient_id=patient.id,
            report_id=report.id,
            council_model_ids=["synthetic"],
            consolidator_model_id="synthetic",
        )
        storage.update_run_status(session, run.id, status="complete")
        for number, kind, extension, content_type in [
            (2, "peer_review", ".json", "application/json"),
            (3, "revision", ".md", "text/markdown"),
            (6, "final_draft", ".md", "text/markdown"),
        ]:
            path = (
                Path(temp_data_dir)
                / "artifacts"
                / run.id
                / f"stage-{number}"
                / ("synthetic" + extension)
            )
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("{}" if number == 2 else "replacement bytes")
            artifact = storage.create_artifact(
                session,
                run_id=run.id,
                stage_num=number,
                stage_name=kind,
                model_id="synthetic",
                kind=kind,
                content_path=path,
                content_type=content_type,
            )
        storage.select_artifact(session, run.id, artifact.id)
    return run


@pytest.fixture(
    params=["register", "upload", "report", "council", "batch_council", "stage"]
)
def producer(request, temp_data_dir, monkeypatch):
    """Exercise the actual writers which can reuse a registered output path."""
    from backend import patient_files, reports, storage

    root = Path(temp_data_dir)
    source = root / "incoming.txt"
    source.write_bytes(b"replacement bytes")
    if request.param == "stage":
        from backend.council.paths import _artifact_path
        from backend.council.types import StageDef
        from backend.council.workflow.llm_calls import _LLMCallsMixin

        run = request.getfixturevalue("ready_run")
        target = _artifact_path(run.id, 6, "synthetic", ".md")

        def update():
            return asyncio.run(
                _LLMCallsMixin()._write_artifact(
                    run_id=run.id,
                    stage=StageDef(
                        6, "final_draft", "final_draft", "text/markdown", ".md"
                    ),
                    model_id="synthetic",
                    text=source.read_text(),
                )
            )

        return target, source, update
    if request.param == "register":
        from scripts.register_patient_file import register_patient_file

        with storage.session_scope() as session:
            storage.create_patient(session, label="ZZ_01-01-1900", notes="")

        def update():
            return register_patient_file(
                patient_label="ZZ_01-01-1900", src_path=source, filename="result.txt"
            )

        initial = update()
        target = Path(initial["stored_path"])
        return target, source, update
    if request.param == "upload":
        target = patient_files.patient_file_original_path(
            "patient", "file", "result.txt"
        )

        def update():
            return patient_files.save_patient_file_upload(
                patient_id="patient",
                file_id="file",
                filename="result.txt",
                provided_mime_type="text/plain",
                src=io.BytesIO(source.read_bytes()),
            )

        return target, source, update
    if request.param == "report":
        directory = root / "reports" / "patient" / "existing-upload"
        target = directory / "original.txt"

        def update():
            return reports.save_report_upload(
                patient_id="patient",
                report_id="existing-report",
                filename="result.txt",
                provided_mime_type="text/plain",
                file_bytes=source.read_bytes(),
                target_dir=directory,
            )

        return target, source, update

    run_id = "existing-run"
    label = "ZZ_01-01-1900"
    source = root / "artifacts" / run_id / "stage-4" / "council.md"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"replacement bytes")
    target = (
        root / "portal_patients" / label / "council" / run_id / "stage-4" / source.name
    )
    if request.param == "council":
        from scripts import stage_council_artifacts as script

        # This legacy script derives its data directory from __file__.
        monkeypatch.setattr(script, "__file__", str(root / "scripts" / "stage.py"))
        # Its derived root has an extra data segment.
        source = root / "data" / source.relative_to(root)
        source.parent.mkdir(parents=True)
        source.write_bytes(b"replacement bytes")
        target = root / "data" / target.relative_to(root)
        monkeypatch.setattr(
            script,
            "get_completed_runs_by_patient_label",
            lambda: {label: [(run_id, "source.pdf")]},
        )
        monkeypatch.setattr(script, "sync_patient_to_thrylen", lambda label: 0)
        return target, source, script.stage_artifacts

    from scripts import run_portal_council_batch as script

    monkeypatch.setattr(script, "ARTIFACTS_DIR", root / "artifacts")
    monkeypatch.setattr(
        script, "_portal_patients_dir", lambda: root / "portal_patients"
    )
    return (
        target,
        source,
        lambda: script._stage_run_artifacts(patient_label=label, run_id=run_id),
    )


@pytest.mark.parametrize("link_kind", ["symlink", "hardlink", "regular"])
def test_existing_output_replacement_preserves_retained_history(
    producer, link_kind, tmp_path
):
    target, source, update = producer
    target.parent.mkdir(parents=True, exist_ok=True)
    target.unlink(missing_ok=True)
    retained = tmp_path / "shared-original"
    retained.write_bytes(b"retained history")
    # The incremental staging script should see a newer incoming file.
    os.utime(retained, (1, 1))
    retained.chmod(0o444)
    sibling = tmp_path / "another-registered-output"
    sibling.symlink_to(retained)
    if link_kind == "symlink":
        target.symlink_to(retained)
    elif link_kind == "hardlink":
        os.link(retained, target)
    else:
        target.write_bytes(b"old independent bytes")
        os.utime(target, (1, 1))

    update()

    assert target.read_bytes() == source.read_bytes() == b"replacement bytes"
    assert not target.is_symlink()
    assert not os.path.samefile(target, retained)
    assert retained.read_bytes() == sibling.read_bytes() == b"retained history"
    assert retained.stat().st_mode & 0o777 == 0o444
    assert not list(target.parent.glob(".pending-*"))


@pytest.mark.parametrize("link_kind", ["symlink", "hardlink", "regular"])
@pytest.mark.parametrize("writer", ["api", "batch"])
def test_export_replaces_both_outputs_without_changing_history(
    ready_run, temp_data_dir, monkeypatch, link_kind, writer
):
    from backend import main

    root = Path(temp_data_dir)
    monkeypatch.setattr(main, "EXPORTS_DIR", root / "exports")
    monkeypatch.setattr(main, "_portal_patients_dir", lambda: root / "portal_patients")
    monkeypatch.setattr(main, "_schedule_portal_sync", lambda **kwargs: None)
    retained = root / "retained-export"
    retained.write_bytes(b"retained previous export")
    retained.chmod(0o444)
    directory = root / "exports" / ready_run.id
    directory.mkdir(parents=True)
    for suffix in ("md", "pdf"):
        target = directory / f"final.{suffix}"
        if link_kind == "symlink":
            target.symlink_to(retained)
        elif link_kind == "hardlink":
            os.link(retained, target)
        else:
            target.write_bytes(retained.read_bytes())

    if writer == "api":
        result = asyncio.run(main.export(ready_run.id))
        assert result["ok"]
    else:
        from scripts import run_portal_council_batch as script

        monkeypatch.setattr(script, "EXPORTS_DIR", root / "exports")
        monkeypatch.setattr(
            script, "_portal_patients_dir", lambda: root / "portal_patients"
        )
        assert script._export_run(ready_run.id) == (
            directory / "final.md",
            directory / "final.pdf",
        )
    assert (directory / "final.md").read_text() == "replacement bytes"
    assert (directory / "final.pdf").read_bytes().startswith(b"%PDF-")
    assert retained.read_bytes() == b"retained previous export"
    assert retained.stat().st_mode & 0o777 == 0o444
    for suffix in ("md", "pdf"):
        target = directory / f"final.{suffix}"
        assert not target.is_symlink()
        assert not os.path.samefile(target, retained)
    assert not list(directory.glob(".pending-*"))


@pytest.mark.parametrize("link_kind", ["symlink", "hardlink", "regular"])
def test_interrupted_upload_keeps_previous_complete_output(temp_data_dir, link_kind):
    from backend.patient_files import (
        patient_file_original_path,
        save_patient_file_upload,
    )

    target = patient_file_original_path("patient", "file", "video.mp4")
    target.parent.mkdir(parents=True)
    retained = Path(temp_data_dir) / "shared-video"
    retained.write_bytes(b"complete previous video")
    retained.chmod(0o444)
    if link_kind == "symlink":
        target.symlink_to(retained)
    elif link_kind == "hardlink":
        os.link(retained, target)
    else:
        target.write_bytes(retained.read_bytes())
    old_inode = target.lstat().st_ino

    class InterruptedStream:
        calls = 0

        def read(self, size):
            self.calls += 1
            if self.calls == 1:
                return b"partial replacement"
            raise OSError("interrupted input")

    with pytest.raises(OSError, match="interrupted input"):
        save_patient_file_upload(
            patient_id="patient",
            file_id="file",
            filename="video.mp4",
            provided_mime_type="video/mp4",
            src=InterruptedStream(),
        )

    assert target.lstat().st_ino == old_inode
    assert target.read_bytes() == retained.read_bytes() == b"complete previous video"
    assert not list(target.parent.glob(".pending-*"))
