from __future__ import annotations

import asyncio
import json
import subprocess
from pathlib import Path

import pytest


def test_qeeg_process_does_not_compete_with_launchd_sync_by_default(monkeypatch):
    from backend import portal_sync

    monkeypatch.delenv("QEEG_PORTAL_NETLIFY_SYNC_ON_PUBLISH", raising=False)

    assert portal_sync.sync_patient_to_thrylen("MK_01-01-2013") is False
    assert portal_sync.spawn_portal_sync("MK_01-01-2013") is False


def test_sync_lock_acquires_nonblocking_and_releases(tmp_path: Path, monkeypatch):
    from backend import portal_sync

    operations: list[int] = []

    def fake_flock(_fd: int, operation: int) -> None:
        operations.append(operation)

    monkeypatch.setattr(portal_sync.fcntl, "flock", fake_flock)

    with portal_sync._sync_lock(tmp_path) as acquired:
        assert acquired is True

    assert operations == [
        portal_sync.fcntl.LOCK_EX | portal_sync.fcntl.LOCK_NB,
        portal_sync.fcntl.LOCK_UN,
    ]


def test_sync_lock_times_out_instead_of_waiting_forever(
    tmp_path: Path, monkeypatch
):
    from backend import portal_sync

    monotonic_values = iter((10.0, 10.0, 10.02))
    sleeps: list[float] = []

    def always_busy(_fd: int, operation: int) -> None:
        assert operation == portal_sync.fcntl.LOCK_EX | portal_sync.fcntl.LOCK_NB
        raise BlockingIOError

    monkeypatch.setenv("QEEG_PORTAL_SYNC_LOCK_TIMEOUT_S", "0.01")
    monkeypatch.setattr(portal_sync.fcntl, "flock", always_busy)
    monkeypatch.setattr(
        portal_sync.time, "monotonic", lambda: next(monotonic_values)
    )
    monkeypatch.setattr(portal_sync.time, "sleep", sleeps.append)

    with portal_sync._sync_lock(tmp_path) as acquired:
        assert acquired is False

    assert sleeps == [pytest.approx(0.01)]


def test_filter_and_merge_sync_state_preserve_other_patients():
    from backend import portal_sync

    patient_id = "MK_01-01-2013"
    other_id = "HT_09-05-1954"
    base_state = {
        "patients": {
            patient_id: {"createdAt": 1},
            other_id: {"createdAt": 2},
        },
        "files": {
            f"{patient_id}/old.pdf": {"version": 1},
            f"{other_id}/keep.pdf": {"version": 9},
        },
    }

    scoped = portal_sync._filter_sync_state_for_patient(base_state, patient_id)
    assert scoped == {
        "patients": {patient_id: {"createdAt": 1}},
        "files": {f"{patient_id}/old.pdf": {"version": 1}},
    }

    synced = {
        "patients": {patient_id: {"createdAt": 10}},
        "files": {f"{patient_id}/new.pdf": {"version": 2}},
    }
    merged = portal_sync._merge_sync_state_for_patient(base_state, synced, patient_id)

    assert merged["patients"][patient_id] == {"createdAt": 10}
    assert merged["patients"][other_id] == {"createdAt": 2}
    assert merged["files"][f"{patient_id}/new.pdf"] == {"version": 2}
    assert f"{patient_id}/old.pdf" not in merged["files"]
    assert merged["files"][f"{other_id}/keep.pdf"] == {"version": 9}


def test_sync_patient_to_thrylen_scopes_state_and_merges_updates(
    tmp_path: Path, monkeypatch
):
    from backend import portal_sync

    patient_id = "MK_01-01-2013"
    other_id = "HT_09-05-1954"

    portal_root = tmp_path / "portal_patients"
    patient_dir = portal_root / patient_id
    patient_dir.mkdir(parents=True, exist_ok=True)
    (patient_dir / "existing.pdf").write_bytes(b"%PDF-1.4\n")
    (patient_dir / "fresh.md").write_text("# fresh\n", encoding="utf-8")
    nested_dir = patient_dir / "council" / "run-1" / "stage-1"
    nested_dir.mkdir(parents=True, exist_ok=True)
    (nested_dir / "_data_pack.json").write_text("{}", encoding="utf-8")

    state_path = portal_root / ".qeeg_portal_sync_state.json"
    state_path.write_text(
        json.dumps(
            {
                "patients": {
                    patient_id: {"createdAt": 1, "createdBy": "local-sync"},
                    other_id: {"createdAt": 2, "createdBy": "local-sync"},
                },
                "files": {
                    f"{patient_id}/existing.pdf": {
                        "size": 9,
                        "mtimeMs": 100,
                        "remoteFileKey": f"{patient_id}__existing__v1__2026-01-01.pdf",
                        "logicalName": "existing.pdf",
                        "version": 1,
                        "uploadedAt": 1000,
                    },
                    f"{other_id}/keep.pdf": {
                        "size": 9,
                        "mtimeMs": 200,
                        "remoteFileKey": f"{other_id}__keep__v1__2026-01-01.pdf",
                        "logicalName": "keep.pdf",
                        "version": 1,
                        "uploadedAt": 2000,
                    },
                },
            }
        ),
        encoding="utf-8",
    )

    sync_repo = tmp_path / "thrylen"
    sync_script = sync_repo / "scripts" / "qeeg_patients_sync.mjs"
    sync_script.parent.mkdir(parents=True, exist_ok=True)
    sync_script.write_text("// fake sync\n", encoding="utf-8")

    monkeypatch.setenv("QEEG_PORTAL_PATIENTS_DIR", str(portal_root))
    monkeypatch.setenv("QEEG_PORTAL_SYNC_REPO", str(sync_repo))
    monkeypatch.setenv("QEEG_PORTAL_NETLIFY_SYNC_ON_PUBLISH", "1")
    monkeypatch.setattr(
        portal_sync.shutil,
        "which",
        lambda name: "/usr/bin/node" if name == "node" else None,
    )

    observed: dict[str, Path | str] = {}

    def fake_run(cmd, cwd, capture_output, text, check, timeout):
        observed["cwd"] = cwd
        observed["timeout"] = timeout
        temp_root = Path(cmd[-1])
        observed["temp_root"] = temp_root

        scoped_state = json.loads(
            (temp_root / ".qeeg_portal_sync_state.json").read_text(encoding="utf-8")
        )
        assert set(scoped_state["patients"]) == {patient_id}
        assert set(scoped_state["files"]) == {f"{patient_id}/existing.pdf"}
        assert (temp_root / patient_id / "fresh.md").exists()
        assert (
            temp_root / patient_id / "council" / "run-1" / "stage-1" / "_data_pack.json"
        ).exists()
        assert not (temp_root / other_id).exists()

        temp_state = {
            "patients": {patient_id: {"createdAt": 1, "createdBy": "local-sync"}},
            "files": {
                f"{patient_id}/existing.pdf": scoped_state["files"][
                    f"{patient_id}/existing.pdf"
                ],
                f"{patient_id}/fresh.md": {
                    "size": 8,
                    "mtimeMs": 300,
                    "remoteFileKey": f"{patient_id}__fresh__v1__2026-03-17.md",
                    "logicalName": "fresh.md",
                    "version": 1,
                    "uploadedAt": 3000,
                },
            },
        }
        (temp_root / ".qeeg_portal_sync_state.json").write_text(
            json.dumps(temp_state), encoding="utf-8"
        )
        return subprocess.CompletedProcess(cmd, 0, stdout="Done.\n", stderr="")

    monkeypatch.setattr(portal_sync.subprocess, "run", fake_run)

    assert portal_sync.sync_patient_to_thrylen(patient_id) is True
    assert observed["cwd"] == str(sync_repo)
    assert observed["timeout"] == 900.0

    merged_state = json.loads(state_path.read_text(encoding="utf-8"))
    assert merged_state["patients"][other_id] == {
        "createdAt": 2,
        "createdBy": "local-sync",
    }
    assert merged_state["files"][f"{other_id}/keep.pdf"]["remoteFileKey"] == (
        f"{other_id}__keep__v1__2026-01-01.pdf"
    )
    assert merged_state["files"][f"{patient_id}/fresh.md"]["remoteFileKey"] == (
        f"{patient_id}__fresh__v1__2026-03-17.md"
    )


def test_sync_timeout_persists_partial_file_progress_and_requeues_patient(
    tmp_path: Path, monkeypatch
):
    from backend import portal_sync

    patient_id = "MK_01-01-2013"
    other_id = "HT_09-05-1954"
    portal_root = tmp_path / "portal_patients"
    patient_dir = portal_root / patient_id
    patient_dir.mkdir(parents=True)
    (patient_dir / "source.pdf").write_bytes(b"%PDF-1.4\n")

    state_path = portal_root / ".qeeg_portal_sync_state.json"
    state_path.write_text(
        json.dumps(
            {
                "patients": {other_id: {"createdAt": 2}},
                "files": {f"{other_id}/keep.pdf": {"version": 9}},
            }
        ),
        encoding="utf-8",
    )
    watch_state_path = portal_root / ".qeeg_portal_sync_watch_state.json"
    watch_state_path.write_text(
        json.dumps(
            {
                "patients": {
                    patient_id: [1, 10, 100],
                    other_id: [2, 20, 200],
                }
            }
        ),
        encoding="utf-8",
    )

    sync_repo = tmp_path / "thrylen"
    sync_script = sync_repo / "scripts" / "qeeg_patients_sync.mjs"
    sync_script.parent.mkdir(parents=True)
    sync_script.write_text("// fake sync\n", encoding="utf-8")

    monkeypatch.setenv("QEEG_PORTAL_PATIENTS_DIR", str(portal_root))
    monkeypatch.setenv("QEEG_PORTAL_SYNC_REPO", str(sync_repo))
    monkeypatch.setenv("QEEG_PORTAL_NETLIFY_SYNC_ON_PUBLISH", "1")
    monkeypatch.setattr(
        portal_sync.shutil,
        "which",
        lambda name: "/usr/bin/node" if name == "node" else None,
    )

    def fake_run(cmd, cwd, capture_output, text, check, timeout):
        temp_root = Path(cmd[-1])
        temp_state_path = temp_root / ".qeeg_portal_sync_state.json"
        partial_state = json.loads(temp_state_path.read_text(encoding="utf-8"))
        partial_state["patients"][patient_id] = {"createdAt": 1}
        partial_state["files"][f"{patient_id}/source.pdf"] = {
            "size": 9,
            "mtimeMs": 100,
            "remoteFileKey": f"{patient_id}__source__v1__2026-08-02.pdf",
            "logicalName": "source.pdf",
            "version": 1,
        }
        temp_state_path.write_text(
            json.dumps(partial_state), encoding="utf-8"
        )
        raise subprocess.TimeoutExpired(cmd, timeout)

    monkeypatch.setattr(portal_sync.subprocess, "run", fake_run)

    assert portal_sync.sync_patient_to_thrylen(patient_id) is False

    persisted = json.loads(state_path.read_text(encoding="utf-8"))
    assert f"{patient_id}/source.pdf" in persisted["files"]
    assert persisted["files"][f"{other_id}/keep.pdf"] == {"version": 9}
    retry_state = json.loads(watch_state_path.read_text(encoding="utf-8"))
    assert patient_id not in retry_state["patients"]
    assert retry_state["patients"][other_id] == [2, 20, 200]


def test_spawn_portal_sync_skips_when_another_sync_holds_the_global_reservation(
    tmp_path: Path, monkeypatch
):
    from backend import portal_sync

    patient_id = "MK_01-01-2013"
    portal_root = tmp_path / "portal_patients"
    portal_root.mkdir()
    sync_repo = tmp_path / "thrylen"
    sync_script = sync_repo / "scripts" / "qeeg_patients_sync.mjs"
    sync_script.parent.mkdir(parents=True)
    sync_script.write_text("// fake sync\n", encoding="utf-8")

    monkeypatch.setenv("QEEG_PORTAL_PATIENTS_DIR", str(portal_root))
    monkeypatch.setenv("QEEG_PORTAL_SYNC_REPO", str(sync_repo))
    monkeypatch.setenv("QEEG_PORTAL_NETLIFY_SYNC_ON_PUBLISH", "1")
    monkeypatch.setattr(
        portal_sync.shutil,
        "which",
        lambda name: "/usr/bin/node" if name == "node" else None,
    )
    monkeypatch.setattr(
        portal_sync.subprocess,
        "Popen",
        lambda *_args, **_kwargs: pytest.fail("duplicate sync must not spawn"),
    )

    lock_path = portal_sync._sync_spawn_lock_path(portal_root)
    with lock_path.open("a+", encoding="utf-8") as lock_file:
        portal_sync.fcntl.flock(
            lock_file.fileno(),
            portal_sync.fcntl.LOCK_EX | portal_sync.fcntl.LOCK_NB,
        )
        assert portal_sync.spawn_portal_sync(patient_id) is False


def test_spawn_portal_sync_passes_the_global_reservation_to_the_child(
    tmp_path: Path, monkeypatch
):
    from backend import portal_sync

    patient_id = "MK_01-01-2013"
    portal_root = tmp_path / "portal_patients"
    portal_root.mkdir()
    sync_repo = tmp_path / "thrylen"
    sync_script = sync_repo / "scripts" / "qeeg_patients_sync.mjs"
    sync_script.parent.mkdir(parents=True)
    sync_script.write_text("// fake sync\n", encoding="utf-8")
    observed: dict[str, object] = {}

    monkeypatch.setenv("QEEG_PORTAL_PATIENTS_DIR", str(portal_root))
    monkeypatch.setenv("QEEG_PORTAL_SYNC_REPO", str(sync_repo))
    monkeypatch.setenv("QEEG_PORTAL_NETLIFY_SYNC_ON_PUBLISH", "1")
    monkeypatch.setattr(
        portal_sync.shutil,
        "which",
        lambda name: "/usr/bin/node" if name == "node" else None,
    )

    def fake_popen(*args, **kwargs):
        observed["args"] = args
        observed["kwargs"] = kwargs
        return object()

    monkeypatch.setattr(portal_sync.subprocess, "Popen", fake_popen)

    assert portal_sync.spawn_portal_sync(patient_id) is True
    assert observed["kwargs"]["start_new_session"] is True
    assert len(observed["kwargs"]["pass_fds"]) == 1


def test_source_pdfs_missing_complete_runs_flags_followups_not_generated_outputs(
    tmp_path: Path, monkeypatch
):
    from backend import portal_sync

    patient_id = "GH_08-10-1989"
    patient_dir = tmp_path / patient_id
    patient_dir.mkdir()
    (patient_dir / "DK_Mid_10Tx_Toxic-brain-injury.pdf").write_bytes(b"%PDF-1.4")
    (patient_dir / "DK_20Tx_toxic-brain-injury_Redacted.pdf").write_bytes(
        b"%PDF-1.4"
    )
    (patient_dir / f"{patient_id}.pdf").write_bytes(b"%PDF-1.4")
    (
        patient_dir / "GH_08-10-1989__patient-facing__v1__2026-02-09.pdf"
    ).write_bytes(b"%PDF-1.4")
    (patient_dir / f"{patient_id}__guide__v1__2026-03-17.pdf").write_bytes(
        b"%PDF-1.4"
    )
    (patient_dir / "guide.pdf").write_bytes(b"%PDF-1.4")

    monkeypatch.setattr(
        portal_sync,
        "_report_run_statuses_by_filename",
        lambda _label: {"DK_Mid_10Tx_Toxic-brain-injury.pdf": {"complete"}},
    )

    missing_complete, active_runs = portal_sync._source_pdfs_missing_complete_runs(
        patient_dir, patient_id
    )

    assert missing_complete == ["DK_20Tx_toxic-brain-injury_Redacted.pdf"]
    assert active_runs == []


def test_source_pdf_classifier_allows_clinic_analysis_report_names(tmp_path: Path):
    from backend import portal_sync
    from scripts import portal_pipeline_worker
    from scripts import run_portal_council_batch

    patient_id = "GH_08-10-1989"
    source_path = tmp_path / f"{patient_id}__analysis_report__v1__2026-02-09.pdf"
    generated_path = tmp_path / f"{patient_id}__analysis__v1__2026-02-09.pdf"
    generated_sync_echo = (
        tmp_path / f"{patient_id}__{patient_id}__v2897__2026-08-03__retry.pdf"
    )
    source_path.write_bytes(b"%PDF-1.4")
    generated_path.write_bytes(b"%PDF-1.4")
    generated_sync_echo.write_bytes(b"%PDF-1.4")

    assert portal_sync._is_source_pdf(patient_id, source_path)
    assert run_portal_council_batch._is_source_pdf(patient_id, source_path)
    assert not portal_pipeline_worker._looks_generated_pdf(patient_id, source_path.name)

    assert not portal_sync._is_source_pdf(patient_id, generated_path)
    assert not run_portal_council_batch._is_source_pdf(patient_id, generated_path)
    assert not portal_sync._is_source_pdf(patient_id, generated_sync_echo)
    assert not run_portal_council_batch._is_source_pdf(
        patient_id, generated_sync_echo
    )
    assert portal_pipeline_worker._looks_generated_pdf(patient_id, generated_path.name)


def test_source_pdfs_missing_complete_runs_keeps_fresh_created_rows_active(
    tmp_path: Path, temp_data_dir: Path, monkeypatch
):
    from backend import portal_sync
    from backend import storage

    patient_id = "GH_08-10-1989_2"
    patient_dir = tmp_path / patient_id
    patient_dir.mkdir()
    filename = "DK_20Tx_toxic-brain-injury_Redacted.pdf"
    (patient_dir / filename).write_bytes(b"%PDF-1.4")

    with storage.session_scope() as session:
        patient = storage.create_patient(session, label=patient_id, notes="")
        report = storage.create_report(
            session,
            patient_id=patient.id,
            filename=filename,
            mime_type="application/pdf",
            stored_path=tmp_path / "report.pdf",
            extracted_text_path=tmp_path / "report.txt",
        )
        storage.create_run(
            session,
            patient_id=patient.id,
            report_id=report.id,
            council_model_ids=["gpt-5.4"],
            consolidator_model_id="claude-sonnet-4-6",
        )

    missing_complete, active_runs = portal_sync._source_pdfs_missing_complete_runs(
        patient_dir, patient_id
    )

    assert missing_complete == []
    assert active_runs == [filename]


def test_source_pdfs_missing_complete_runs_ignores_stale_running_rows(
    tmp_path: Path, temp_data_dir: Path, monkeypatch
):
    from backend import portal_sync
    from backend import storage
    from backend.orchestration import progress_jsonl_path

    monkeypatch.setenv("QEEG_RUN_STALE_AFTER_S", "300")

    patient_id = "GH_08-10-1989"
    patient_dir = tmp_path / patient_id
    patient_dir.mkdir()
    filename = "DK_20Tx_toxic-brain-injury_Redacted.pdf"
    (patient_dir / filename).write_bytes(b"%PDF-1.4")

    with storage.session_scope() as session:
        patient = storage.create_patient(session, label=patient_id, notes="")
        report = storage.create_report(
            session,
            patient_id=patient.id,
            filename=filename,
            mime_type="application/pdf",
            stored_path=tmp_path / "report.pdf",
            extracted_text_path=tmp_path / "report.txt",
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

    missing_complete, active_runs = portal_sync._source_pdfs_missing_complete_runs(
        patient_dir, patient_id
    )

    assert missing_complete == [filename]
    assert active_runs == []


@pytest.mark.asyncio
async def test_watch_portal_patients_forever_syncs_stable_raw_changes(
    tmp_path: Path, monkeypatch
):
    from backend import portal_sync

    patient_id = "MK_01-01-2013"
    snapshots = [
        {patient_id: (1, 100, 1000)},
        {patient_id: (2, 200, 2000)},
        {patient_id: (2, 200, 2000)},
    ]
    sync_calls: list[str] = []
    sleep_calls = 0

    def fake_snapshot(_root_dir):
        if snapshots:
            return snapshots.pop(0)
        return {patient_id: (2, 200, 2000)}

    def fake_spawn(label: str) -> bool:
        sync_calls.append(label)
        return True

    async def fake_sleep(_seconds: float):
        nonlocal sleep_calls
        sleep_calls += 1
        if sleep_calls >= 3:
            raise asyncio.CancelledError

    monkeypatch.setenv("QEEG_PORTAL_RAW_SYNC_WATCHER", "1")
    monkeypatch.setenv("QEEG_PORTAL_NETLIFY_SYNC_ON_PUBLISH", "1")
    monkeypatch.setenv("QEEG_PORTAL_PATIENTS_DIR", str(tmp_path))
    monkeypatch.setenv("QEEG_PORTAL_RAW_SYNC_POLL_S", "0.01")
    monkeypatch.setenv("QEEG_PORTAL_RAW_SYNC_STABLE_POLLS", "2")
    (tmp_path / ".qeeg_portal_sync_watch_state.json").write_text(
        json.dumps({"patients": {patient_id: [1, 100, 1000]}}), encoding="utf-8"
    )
    monkeypatch.setattr(
        portal_sync, "_snapshot_portal_patient_fingerprints", fake_snapshot
    )
    monkeypatch.setattr(portal_sync, "spawn_portal_sync", fake_spawn)
    monkeypatch.setattr(portal_sync.asyncio, "sleep", fake_sleep)

    with pytest.raises(asyncio.CancelledError):
        await portal_sync.watch_portal_patients_forever()

    assert sync_calls == [patient_id]


@pytest.mark.asyncio
async def test_watch_portal_patients_forever_retries_snapshot_cleared_by_child(
    tmp_path: Path, monkeypatch
):
    from backend import portal_sync

    patient_id = "MK_01-01-2013"
    fingerprint = (2, 200, 2000)
    sync_calls: list[str] = []
    sleep_calls = 0
    sync_state_path = tmp_path / ".qeeg_portal_sync_watch_state.json"
    sync_state_path.write_text(
        json.dumps({"patients": {patient_id: [1, 100, 1000]}}),
        encoding="utf-8",
    )

    def fake_snapshot(_root_dir):
        return {patient_id: fingerprint}

    def fake_spawn(label: str) -> bool:
        sync_calls.append(label)
        return True

    async def fake_sleep(_seconds: float):
        nonlocal sleep_calls
        sleep_calls += 1
        if sleep_calls == 2:
            sync_state_path.write_text(
                json.dumps({"patients": {}}), encoding="utf-8"
            )
        if sleep_calls >= 3:
            raise asyncio.CancelledError

    monkeypatch.setenv("QEEG_PORTAL_RAW_SYNC_WATCHER", "1")
    monkeypatch.setenv("QEEG_PORTAL_NETLIFY_SYNC_ON_PUBLISH", "1")
    monkeypatch.setenv("QEEG_PORTAL_PATIENTS_DIR", str(tmp_path))
    monkeypatch.setenv("QEEG_PORTAL_RAW_SYNC_POLL_S", "0.01")
    monkeypatch.setenv("QEEG_PORTAL_RAW_SYNC_STABLE_POLLS", "2")
    monkeypatch.setattr(
        portal_sync, "_snapshot_portal_patient_fingerprints", fake_snapshot
    )
    monkeypatch.setattr(portal_sync, "spawn_portal_sync", fake_spawn)
    monkeypatch.setattr(portal_sync.asyncio, "sleep", fake_sleep)

    with pytest.raises(asyncio.CancelledError):
        await portal_sync.watch_portal_patients_forever()

    assert sync_calls == [patient_id, patient_id]


@pytest.mark.asyncio
async def test_watch_portal_patients_forever_seeds_sync_state_without_mass_resync(
    tmp_path: Path, monkeypatch
):
    from backend import portal_sync

    patient_id = "MK_01-01-2013"
    snapshots = [
        {patient_id: (2, 200, 2000)},
        {patient_id: (2, 200, 2000)},
    ]
    sync_calls: list[str] = []
    sleep_calls = 0

    def fake_snapshot(_root_dir):
        if snapshots:
            return snapshots.pop(0)
        return {patient_id: (2, 200, 2000)}

    def fake_spawn(label: str) -> bool:
        sync_calls.append(label)
        return True

    async def fake_sleep(_seconds: float):
        nonlocal sleep_calls
        sleep_calls += 1
        if sleep_calls >= 2:
            raise asyncio.CancelledError

    monkeypatch.setenv("QEEG_PORTAL_RAW_SYNC_WATCHER", "1")
    monkeypatch.setenv("QEEG_PORTAL_PATIENTS_DIR", str(tmp_path))
    monkeypatch.setenv("QEEG_PORTAL_RAW_SYNC_POLL_S", "0.01")
    monkeypatch.setenv("QEEG_PORTAL_RAW_SYNC_STABLE_POLLS", "2")
    monkeypatch.setattr(
        portal_sync, "_snapshot_portal_patient_fingerprints", fake_snapshot
    )
    monkeypatch.setattr(portal_sync, "spawn_portal_sync", fake_spawn)
    monkeypatch.setattr(portal_sync.asyncio, "sleep", fake_sleep)

    with pytest.raises(asyncio.CancelledError):
        await portal_sync.watch_portal_patients_forever()

    seeded = json.loads(
        (tmp_path / ".qeeg_portal_sync_watch_state.json").read_text(encoding="utf-8")
    )
    assert seeded["patients"][patient_id] == [2, 200, 2000]
    assert sync_calls == []


@pytest.mark.asyncio
async def test_watch_portal_patients_forever_spawns_local_pipeline_for_missing_followup(
    tmp_path: Path, monkeypatch
):
    from backend import portal_sync

    patient_id = "GH_08-10-1989"
    snapshots = [
        {patient_id: (2, 200, 2000)},
        {patient_id: (2, 200, 2000)},
    ]
    pipeline_calls: list[str] = []
    sleep_calls = 0

    def fake_snapshot(_root_dir):
        if snapshots:
            return snapshots.pop(0)
        return {patient_id: (2, 200, 2000)}

    def fake_spawn_sync(_label: str) -> bool:
        return True

    def fake_missing_complete(_patient_dir: Path, _patient_id: str):
        return (["DK_20Tx_toxic-brain-injury_Redacted.pdf"], [])

    def fake_spawn_pipeline(label: str) -> bool:
        pipeline_calls.append(label)
        return True

    async def fake_sleep(_seconds: float):
        nonlocal sleep_calls
        sleep_calls += 1
        if sleep_calls >= 2:
            raise asyncio.CancelledError

    monkeypatch.setenv("QEEG_PORTAL_RAW_SYNC_WATCHER", "1")
    monkeypatch.setenv("QEEG_PORTAL_LOCAL_PIPELINE_WATCHER", "1")
    monkeypatch.setenv("QEEG_PORTAL_PATIENTS_DIR", str(tmp_path))
    monkeypatch.setenv("QEEG_PORTAL_RAW_SYNC_POLL_S", "0.01")
    monkeypatch.setenv("QEEG_PORTAL_RAW_SYNC_STABLE_POLLS", "2")
    (tmp_path / ".qeeg_portal_local_pipeline_state.json").write_text(
        json.dumps({"patients": {patient_id: [1, 100, 1000]}}), encoding="utf-8"
    )
    monkeypatch.setattr(
        portal_sync, "_snapshot_portal_patient_fingerprints", fake_snapshot
    )
    monkeypatch.setattr(portal_sync, "spawn_portal_sync", fake_spawn_sync)
    monkeypatch.setattr(
        portal_sync,
        "_source_pdfs_missing_complete_runs",
        fake_missing_complete,
    )
    monkeypatch.setattr(
        portal_sync, "spawn_portal_pipeline", fake_spawn_pipeline
    )
    monkeypatch.setattr(portal_sync.asyncio, "sleep", fake_sleep)

    with pytest.raises(asyncio.CancelledError):
        await portal_sync.watch_portal_patients_forever()

    assert pipeline_calls == [patient_id]


def test_portal_sync_paths_route_only_on_canonical_ids(tmp_path: Path, monkeypatch):
    """Every sync entry point refuses a legacy ``MM-DD-YYYY-N`` key outright."""
    from backend import portal_sync

    portal_root = tmp_path / "portal_patients"
    (portal_root / "09-05-1954-0").mkdir(parents=True)
    (portal_root / "BT_12-11-1963").mkdir(parents=True)
    sync_repo = tmp_path / "thrylen"
    sync_script = sync_repo / "scripts" / "qeeg_patients_sync.mjs"
    sync_script.parent.mkdir(parents=True)
    sync_script.write_text("// fake sync\n", encoding="utf-8")

    monkeypatch.setenv("QEEG_PORTAL_PATIENTS_DIR", str(portal_root))
    monkeypatch.setenv("QEEG_PORTAL_SYNC_REPO", str(sync_repo))
    monkeypatch.setenv("QEEG_PORTAL_NETLIFY_SYNC_ON_PUBLISH", "1")
    monkeypatch.setattr(
        portal_sync.shutil,
        "which",
        lambda name: "/usr/bin/node" if name == "node" else None,
    )
    monkeypatch.setattr(
        portal_sync.subprocess,
        "Popen",
        lambda *_args, **_kwargs: pytest.fail("a legacy key must never spawn work"),
    )

    assert portal_sync.sync_patient_to_thrylen("09-05-1954-0") is False
    assert portal_sync.spawn_portal_sync("09-05-1954-0") is False
    assert portal_sync.spawn_portal_pipeline("09-05-1954-0") is False

    snapshots = portal_sync._snapshot_portal_patient_fingerprints(portal_root)
    assert list(snapshots) == ["BT_12-11-1963"]
