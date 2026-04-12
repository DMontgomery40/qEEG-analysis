from __future__ import annotations

import asyncio
import json
import subprocess
from pathlib import Path

import pytest


def test_filter_and_merge_sync_state_preserve_other_patients():
    from backend import portal_sync

    patient_id = "01-01-2013-0"
    other_id = "09-05-1954-0"
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

    patient_id = "01-01-2013-0"
    other_id = "09-05-1954-0"

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

    def fake_run(cmd, cwd, capture_output, text, check):
        observed["cwd"] = cwd
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


def test_source_pdfs_missing_complete_runs_flags_followups_not_generated_outputs(
    tmp_path: Path, monkeypatch
):
    from backend import portal_sync

    patient_id = "08-10-1989-0"
    patient_dir = tmp_path / patient_id
    patient_dir.mkdir()
    (patient_dir / "DK_Mid_10Tx_Toxic-brain-injury.pdf").write_bytes(b"%PDF-1.4")
    (patient_dir / "DK_20Tx_toxic-brain-injury_Redacted.pdf").write_bytes(
        b"%PDF-1.4"
    )
    (patient_dir / f"{patient_id}.pdf").write_bytes(b"%PDF-1.4")
    (
        patient_dir / "08-10-1989-0__patient-facing__v1__2026-02-09.pdf"
    ).write_bytes(b"%PDF-1.4")

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


@pytest.mark.asyncio
async def test_watch_portal_patients_forever_syncs_stable_raw_changes(
    tmp_path: Path, monkeypatch
):
    from backend import portal_sync

    patient_id = "01-01-2013-0"
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

    assert sync_calls == [patient_id]


@pytest.mark.asyncio
async def test_watch_portal_patients_forever_spawns_local_pipeline_for_missing_followup(
    tmp_path: Path, monkeypatch
):
    from backend import portal_sync

    patient_id = "08-10-1989-0"
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
