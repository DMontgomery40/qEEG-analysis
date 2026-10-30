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


def test_source_pdf_classifier_allows_clinic_analysis_report_names(tmp_path: Path):
    from backend.portal_files import is_source_report_pdf, looks_generated_portal_pdf

    patient_id = "GH_08-10-1989"
    source_path = tmp_path / f"{patient_id}__analysis_report__v1__2026-02-09.pdf"
    generated_path = tmp_path / f"{patient_id}__analysis__v1__2026-02-09.pdf"
    generated_sync_echo = (
        tmp_path / f"{patient_id}__{patient_id}__v2897__2026-08-03__retry.pdf"
    )
    source_path.write_bytes(b"%PDF-1.4")
    generated_path.write_bytes(b"%PDF-1.4")
    generated_sync_echo.write_bytes(b"%PDF-1.4")

    assert is_source_report_pdf(patient_id, source_path)
    assert not looks_generated_portal_pdf(patient_id, source_path.name)

    assert not is_source_report_pdf(patient_id, generated_path)
    assert not is_source_report_pdf(patient_id, generated_sync_echo)
    assert looks_generated_portal_pdf(patient_id, generated_path.name)


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


def test_mirror_preserves_managed_content_aliases_and_skips_unowned_links(tmp_path):
    import hashlib
    from backend.portal_sync import _mirror_tree_with_hardlinks

    src = tmp_path / "patient"
    src.mkdir()
    dest = tmp_path / "mirror"
    dest.mkdir()
    digest = hashlib.sha256(b"original").hexdigest()
    body = src / ".content" / digest / "original"
    body.parent.mkdir(parents=True)
    body.write_bytes(b"original")
    (src / "report.pdf").symlink_to(body)
    outside = tmp_path / "outside"
    outside.write_bytes(b"private")
    (src / "outside.pdf").symlink_to(outside)
    other = src / "arbitrary.txt"
    other.write_bytes(b"unmanaged")
    (src / "unmanaged.pdf").symlink_to(other)
    (src / "missing.pdf").symlink_to(src / ".content" / "missing" / "original")
    _mirror_tree_with_hardlinks(src, dest)
    assert (dest / "report.pdf").read_bytes() == b"original"
    assert not (dest / "report.pdf").is_symlink()
    assert not (dest / ".content").exists()
    assert not any(
        (dest / name).exists()
        for name in ["outside.pdf", "unmanaged.pdf", "missing.pdf"]
    )
    body.write_bytes(b"corrupted")
    with pytest.raises(ValueError, match="content"):
        _mirror_tree_with_hardlinks(src, tmp_path / "bad-mirror")

def test_mirror_skips_cyclic_links_and_preserves_other_files(tmp_path):
    from backend.portal_sync import _mirror_tree_with_hardlinks

    source = tmp_path / "source"
    source.mkdir()
    dest = tmp_path / "dest"
    (source / "loop").symlink_to("loop")
    (source / "cycle-a").symlink_to("cycle-b")
    (source / "cycle-b").symlink_to("cycle-a")
    (source / "report.pdf").write_bytes(b"valid")
    _mirror_tree_with_hardlinks(source, dest)
    assert (dest / "report.pdf").read_bytes() == b"valid"
    assert not (dest / "loop").exists()


def test_a_changed_shared_body_skips_the_sync_and_returns_false(tmp_path, monkeypatch):
    # sync_patient_to_thrylen answers True or False; scripts that call it
    # directly must not die on a corrupted .content body.
    import hashlib

    from backend import portal_sync

    portal_root = tmp_path / "portal_patients"
    folder = portal_root / "MK_01-01-2013"
    digest = hashlib.sha256(b"original").hexdigest()
    body = folder / ".content" / digest / "original"
    body.parent.mkdir(parents=True)
    body.write_bytes(b"corrupted")
    (folder / "report.pdf").symlink_to(f".content/{digest}/original")
    sync_repo = tmp_path / "thrylen"
    (sync_repo / "scripts").mkdir(parents=True)
    (sync_repo / "scripts" / "qeeg_patients_sync.mjs").write_text("// fake\n")
    monkeypatch.setenv("QEEG_PORTAL_PATIENTS_DIR", str(portal_root))
    monkeypatch.setenv("QEEG_PORTAL_SYNC_REPO", str(sync_repo))
    monkeypatch.setenv("QEEG_PORTAL_NETLIFY_SYNC_ON_PUBLISH", "1")
    monkeypatch.setattr(
        portal_sync.shutil, "which", lambda name: "/usr/bin/node" if name == "node" else None
    )
    monkeypatch.setattr(
        portal_sync.subprocess,
        "run",
        lambda *_a, **_k: pytest.fail("changed bytes must never be published"),
    )

    assert portal_sync.sync_patient_to_thrylen("MK_01-01-2013") is False


def test_mirror_hashes_each_shared_body_once(tmp_path, monkeypatch):
    import hashlib

    from backend import portal_sync

    src = tmp_path / "patient"
    digest = hashlib.sha256(b"video").hexdigest()
    body = src / ".content" / digest / "original"
    body.parent.mkdir(parents=True)
    body.write_bytes(b"video")
    for name in ("a.mp4", "b.mp4", "c.mp4"):
        (src / name).symlink_to(f".content/{digest}/original")
    opened = []
    real_open = Path.open

    def counting_open(self, *args, **kwargs):
        if self == body:
            opened.append(self)
        return real_open(self, *args, **kwargs)

    monkeypatch.setattr(Path, "open", counting_open)
    portal_sync._mirror_tree_with_hardlinks(src, tmp_path / "mirror")
    assert [(tmp_path / "mirror" / n).read_bytes() for n in ("a.mp4", "b.mp4", "c.mp4")] == [b"video"] * 3
    assert len(opened) == 1
