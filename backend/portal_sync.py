from __future__ import annotations

import argparse
import asyncio
import fcntl
import json
import os
import shutil
import subprocess
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from . import storage
from . import config as cfg
from .logging_utils import get_logger

LOGGER = get_logger(__name__)


def _truthy_env(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    value = raw.strip().lower()
    if value in {"", "0", "false", "no", "off", "n"}:
        return False
    if value in {"1", "true", "yes", "on", "y"}:
        return True
    return default


def _normalize_portal_patient_id(label: str) -> str | None:
    import re

    m = re.fullmatch(
        r"\s*(?P<mm>\d{1,2})-(?P<dd>\d{1,2})-(?P<yyyy>\d{4})-(?P<n>\d{1,3})\s*",
        label or "",
    )
    if not m:
        return None
    mm = int(m.group("mm"))
    dd = int(m.group("dd"))
    yyyy = int(m.group("yyyy"))
    n = int(m.group("n"))
    if not (1 <= mm <= 12 and 1 <= dd <= 31 and 1900 <= yyyy <= 2100 and 0 <= n <= 999):
        return None
    return f"{mm:02d}-{dd:02d}-{yyyy:04d}-{n}"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def portal_patients_dir() -> Path:
    configured = (os.getenv("QEEG_PORTAL_PATIENTS_DIR") or "").strip()
    if configured:
        return Path(configured).expanduser()
    return cfg.DATA_DIR / "portal_patients"


def portal_sync_repo() -> Path:
    configured = (os.getenv("QEEG_PORTAL_SYNC_REPO") or "").strip()
    if configured:
        return Path(configured).expanduser()
    return _repo_root().parent / "thrylen"


def _sync_state_path(root_dir: Path) -> Path:
    return root_dir / ".qeeg_portal_sync_state.json"


def _pipeline_watch_state_path(root_dir: Path) -> Path:
    return root_dir / ".qeeg_portal_local_pipeline_state.json"


def _load_sync_state(path: Path) -> dict[str, Any]:
    try:
        raw = path.read_text(encoding="utf-8")
        parsed = json.loads(raw)
    except Exception:
        return {"patients": {}, "files": {}}
    if not isinstance(parsed, dict):
        return {"patients": {}, "files": {}}
    patients = parsed.get("patients")
    files = parsed.get("files")
    return {
        "patients": patients if isinstance(patients, dict) else {},
        "files": files if isinstance(files, dict) else {},
    }


def _write_sync_state(path: Path, state: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.partial")
    tmp_path.write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")
    tmp_path.replace(path)


def _load_pipeline_watch_state(path: Path) -> dict[str, tuple[int, int, int]]:
    try:
        raw = path.read_text(encoding="utf-8")
        parsed = json.loads(raw)
    except Exception:
        return {}
    if not isinstance(parsed, dict):
        return {}
    patients = parsed.get("patients")
    if not isinstance(patients, dict):
        return {}
    state: dict[str, tuple[int, int, int]] = {}
    for patient_id, fingerprint in patients.items():
        if not isinstance(patient_id, str) or not isinstance(fingerprint, list):
            continue
        if len(fingerprint) != 3:
            continue
        try:
            state[patient_id] = tuple(int(part) for part in fingerprint)  # type: ignore[assignment]
        except Exception:
            continue
    return state


def _write_pipeline_watch_state(
    path: Path, state: dict[str, tuple[int, int, int]]
) -> None:
    payload = {
        "patients": {
            patient_id: list(fingerprint)
            for patient_id, fingerprint in sorted(state.items())
        }
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.partial")
    tmp_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    tmp_path.replace(path)


def _filter_sync_state_for_patient(
    state: dict[str, Any], patient_id: str
) -> dict[str, Any]:
    patient_state = {}
    if isinstance(state.get("patients"), dict) and patient_id in state["patients"]:
        patient_state[patient_id] = state["patients"][patient_id]

    file_state = {}
    if isinstance(state.get("files"), dict):
        for key, value in state["files"].items():
            if key == patient_id or str(key).startswith(f"{patient_id}/"):
                file_state[key] = value

    return {"patients": patient_state, "files": file_state}


def _merge_sync_state_for_patient(
    base_state: dict[str, Any], patient_state: dict[str, Any], patient_id: str
) -> dict[str, Any]:
    merged = {
        "patients": dict(base_state.get("patients") or {}),
        "files": dict(base_state.get("files") or {}),
    }

    merged["patients"].pop(patient_id, None)
    for key in list(merged["files"].keys()):
        if key == patient_id or str(key).startswith(f"{patient_id}/"):
            del merged["files"][key]

    if (
        isinstance(patient_state.get("patients"), dict)
        and patient_id in patient_state["patients"]
    ):
        merged["patients"][patient_id] = patient_state["patients"][patient_id]
    if isinstance(patient_state.get("files"), dict):
        for key, value in patient_state["files"].items():
            if key == patient_id or str(key).startswith(f"{patient_id}/"):
                merged["files"][key] = value

    return merged


def _mirror_tree_with_hardlinks(src_dir: Path, dest_dir: Path) -> None:
    for path in src_dir.rglob("*"):
        rel_path = path.relative_to(src_dir)
        dest_path = dest_dir / rel_path
        if path.is_dir():
            dest_path.mkdir(parents=True, exist_ok=True)
            continue
        if path.is_symlink():
            continue
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.link(path, dest_path)
        except Exception:
            shutil.copy2(path, dest_path)


def _sync_command(*, temp_root_dir: Path) -> tuple[list[str], Path] | None:
    sync_repo = portal_sync_repo()
    sync_script = sync_repo / "scripts" / "qeeg_patients_sync.mjs"
    node_bin = shutil.which("node")
    if not sync_repo.exists() or not sync_script.exists():
        return None
    if node_bin is None:
        return None
    return [node_bin, str(sync_script), "--dir", str(temp_root_dir)], sync_repo


@contextmanager
def _sync_lock(root_dir: Path) -> Iterator[None]:
    lock_path = root_dir / ".qeeg_portal_netlify_sync.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+", encoding="utf-8") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def sync_patient_to_thrylen(patient_label: str) -> bool:
    if not _truthy_env("QEEG_PORTAL_NETLIFY_SYNC_ON_PUBLISH", True):
        return False

    patient_id = _normalize_portal_patient_id(patient_label)
    if patient_id is None:
        LOGGER.warning(
            "portal_sync_skipped_invalid_patient_label",
            patient_label=patient_label,
        )
        return False

    root_dir = portal_patients_dir()
    patient_dir = root_dir / patient_id
    if not patient_dir.exists():
        LOGGER.warning(
            "portal_sync_skipped_missing_patient_dir",
            patient_label=patient_id,
            patient_dir=str(patient_dir),
        )
        return False

    command_info = _sync_command(temp_root_dir=Path("/tmp"))
    if command_info is None:
        LOGGER.warning(
            "portal_sync_unavailable",
            patient_label=patient_id,
            sync_repo=str(portal_sync_repo()),
        )
        return False

    with _sync_lock(root_dir):
        state_path = _sync_state_path(root_dir)
        base_state = _load_sync_state(state_path)
        scoped_state = _filter_sync_state_for_patient(base_state, patient_id)

        with tempfile.TemporaryDirectory(
            prefix=f"qeeg-portal-sync-{patient_id}-"
        ) as temp_root_raw:
            temp_root = Path(temp_root_raw)
            mirrored_patient_dir = temp_root / patient_id
            mirrored_patient_dir.mkdir(parents=True, exist_ok=True)
            _mirror_tree_with_hardlinks(patient_dir, mirrored_patient_dir)

            temp_state_path = _sync_state_path(temp_root)
            _write_sync_state(temp_state_path, scoped_state)

            command_info = _sync_command(temp_root_dir=temp_root)
            if command_info is None:
                LOGGER.warning(
                    "portal_sync_unavailable",
                    patient_label=patient_id,
                    sync_repo=str(portal_sync_repo()),
                )
                return False
            command, cwd = command_info

            proc = subprocess.run(
                command,
                cwd=str(cwd),
                capture_output=True,
                text=True,
                check=False,
            )
            if proc.returncode != 0:
                LOGGER.error(
                    "portal_sync_failed",
                    patient_label=patient_id,
                    returncode=proc.returncode,
                    stdout=(proc.stdout or "")[-2000:],
                    stderr=(proc.stderr or "")[-2000:],
                    operatorHint="Single-patient Netlify sync shells into thrylen/scripts/qeeg_patients_sync.mjs; verify node, netlify auth, and the linked thrylen repo.",
                )
                return False

            synced_state = _load_sync_state(temp_state_path)

        merged_state = _merge_sync_state_for_patient(
            base_state, synced_state, patient_id
        )
        _write_sync_state(state_path, merged_state)

    LOGGER.info(
        "portal_sync_completed",
        patient_label=patient_id,
        sync_repo=str(portal_sync_repo()),
    )
    return True


def spawn_portal_sync(patient_label: str) -> bool:
    if not _truthy_env("QEEG_PORTAL_NETLIFY_SYNC_ON_PUBLISH", True):
        return False

    patient_id = _normalize_portal_patient_id(patient_label)
    if patient_id is None:
        return False

    command_info = _sync_command(temp_root_dir=Path("/tmp"))
    if command_info is None:
        LOGGER.warning(
            "portal_sync_spawn_unavailable",
            patient_label=patient_id,
            sync_repo=str(portal_sync_repo()),
        )
        return False

    cmd = [
        sys.executable,
        "-m",
        "backend.portal_sync",
        "--patient-label",
        patient_id,
    ]
    try:
        subprocess.Popen(
            cmd,
            cwd=str(_repo_root()),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
    except Exception:
        LOGGER.exception(
            "portal_sync_spawn_failed",
            patient_label=patient_id,
            operatorHint="Background portal sync spawn shells back into python -m backend.portal_sync; verify sys.executable, repo cwd, and local process launch permissions.",
        )
        return False

    LOGGER.info("portal_sync_spawned", patient_label=patient_id)
    return True


def _is_source_pdf(patient_id: str, path: Path) -> bool:
    if not path.is_file() or path.suffix.lower() != ".pdf":
        return False

    lower_name = path.name.lower()
    lower_label = patient_id.lower()
    if lower_name == f"{lower_label}.pdf":
        return False
    if lower_name == "main.pdf":
        return False
    if "__patient-facing__" in lower_name or "patient-facing" in lower_name:
        return False
    if "__single-agent" in lower_name:
        return False
    if "__analysis" in lower_name:
        return False
    if lower_name.endswith("_analysis_pdf.pdf"):
        return False
    if lower_name.endswith("__patient-facing.pdf"):
        return False
    return True


def _report_run_statuses_by_filename(patient_label: str) -> dict[str, set[str]]:
    statuses_by_filename: dict[str, set[str]] = {}
    with storage.session_scope() as session:
        patients = storage.find_patients_by_label(session, patient_label)
        for patient in patients:
            for report in storage.list_reports(session, patient.id):
                filename = (report.filename or "").strip()
                if not filename:
                    continue
                statuses = statuses_by_filename.setdefault(filename, set())
                runs = (
                    session.query(storage.Run)
                    .filter(storage.Run.report_id == report.id)
                    .all()
                )
                for run in runs:
                    status = (run.status or "").strip()
                    if status:
                        statuses.add(status)
    return statuses_by_filename


def _source_pdfs_missing_complete_runs(
    patient_dir: Path, patient_id: str
) -> tuple[list[str], list[str]]:
    statuses_by_filename = _report_run_statuses_by_filename(patient_id)
    missing_complete: list[str] = []
    active_runs: list[str] = []

    for path in sorted(patient_dir.glob("*.pdf")):
        if not _is_source_pdf(patient_id, path):
            continue
        filename = path.name
        statuses = statuses_by_filename.get(filename, set())
        if {"created", "running"} & statuses:
            active_runs.append(filename)
            continue
        if "complete" not in statuses:
            missing_complete.append(filename)

    return missing_complete, active_runs


def spawn_portal_pipeline(patient_label: str) -> bool:
    patient_id = _normalize_portal_patient_id(patient_label)
    if patient_id is None:
        return False

    cmd = [
        sys.executable,
        str(_repo_root() / "scripts" / "run_portal_council_batch.py"),
        "--include-label",
        patient_id,
    ]
    log_path = Path("/tmp") / f"qeeg_local_portal_pipeline_{patient_id}.log"
    try:
        with log_path.open("ab") as log_file:
            subprocess.Popen(
                cmd,
                cwd=str(_repo_root()),
                stdout=log_file,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
    except Exception:
        LOGGER.exception(
            "portal_pipeline_spawn_failed",
            patient_label=patient_id,
            operatorHint="Local follow-up pipeline spawn shells into scripts/run_portal_council_batch.py; verify sys.executable, repo cwd, and CLIProxy reachability.",
        )
        return False

    LOGGER.info(
        "portal_pipeline_spawned",
        patient_label=patient_id,
        log_path=str(log_path),
    )
    return True


def _portal_patient_tree_fingerprint(patient_dir: Path) -> tuple[int, int, int]:
    latest_mtime_ns = 0
    file_count = 0
    total_size = 0

    try:
        latest_mtime_ns = max(latest_mtime_ns, patient_dir.stat().st_mtime_ns)
    except Exception:
        return (0, 0, 0)

    for path in patient_dir.rglob("*"):
        try:
            rel_parts = path.relative_to(patient_dir).parts
        except Exception:
            continue
        if any(part.startswith(".") for part in rel_parts):
            continue
        if path.name == "_README.txt":
            continue
        try:
            stat = path.stat()
        except Exception:
            continue
        latest_mtime_ns = max(latest_mtime_ns, int(stat.st_mtime_ns or 0))
        if path.is_file():
            file_count += 1
            total_size += int(stat.st_size or 0)

    return (file_count, total_size, latest_mtime_ns)


def _snapshot_portal_patient_fingerprints(
    root_dir: Path,
) -> dict[str, tuple[int, int, int]]:
    snapshots: dict[str, tuple[int, int, int]] = {}
    if not root_dir.exists():
        return snapshots

    for entry in root_dir.iterdir():
        if not entry.is_dir() or entry.name.startswith("."):
            continue
        patient_id = _normalize_portal_patient_id(entry.name)
        if patient_id is None:
            continue
        snapshots[patient_id] = _portal_patient_tree_fingerprint(entry)

    return snapshots


async def watch_portal_patients_forever() -> None:
    if not _truthy_env("QEEG_PORTAL_RAW_SYNC_WATCHER", True):
        return

    local_pipeline_watcher = _truthy_env(
        "QEEG_PORTAL_LOCAL_PIPELINE_WATCHER", True
    )

    try:
        poll_interval_s = float(os.getenv("QEEG_PORTAL_RAW_SYNC_POLL_S", "5") or "5")
    except Exception:
        poll_interval_s = 5.0
    if poll_interval_s <= 0:
        poll_interval_s = 5.0

    try:
        stable_polls = int(os.getenv("QEEG_PORTAL_RAW_SYNC_STABLE_POLLS", "2") or "2")
    except Exception:
        stable_polls = 2
    if stable_polls < 1:
        stable_polls = 1

    root_dir = portal_patients_dir()
    pipeline_state_path = _pipeline_watch_state_path(root_dir)
    LOGGER.info(
        "portal_raw_sync_watcher_started",
        root_dir=str(root_dir),
        poll_interval_s=poll_interval_s,
        stable_polls=stable_polls,
        local_pipeline_watcher=local_pipeline_watcher,
    )

    previous_snapshots: dict[str, tuple[int, int, int]] | None = None
    stable_counts: dict[str, int] = {}
    last_synced_snapshots: dict[str, tuple[int, int, int]] = {}
    last_pipeline_snapshots: dict[str, tuple[int, int, int]] = (
        _load_pipeline_watch_state(pipeline_state_path)
        if local_pipeline_watcher
        else {}
    )
    seed_pipeline_snapshots = local_pipeline_watcher and not pipeline_state_path.exists()

    try:
        while True:
            current_snapshots = await asyncio.to_thread(
                _snapshot_portal_patient_fingerprints, root_dir
            )

            if previous_snapshots is None:
                previous_snapshots = current_snapshots
                stable_counts = {patient_id: 1 for patient_id in current_snapshots}
                if seed_pipeline_snapshots:
                    last_pipeline_snapshots = dict(current_snapshots)
                    _write_pipeline_watch_state(
                        pipeline_state_path, last_pipeline_snapshots
                    )
                    seed_pipeline_snapshots = False
                await asyncio.sleep(poll_interval_s)
                continue

            removed_patient_ids = set(previous_snapshots) - set(current_snapshots)
            pipeline_state_dirty = False
            for patient_id in removed_patient_ids:
                stable_counts.pop(patient_id, None)
                last_synced_snapshots.pop(patient_id, None)
                if patient_id in last_pipeline_snapshots:
                    last_pipeline_snapshots.pop(patient_id, None)
                    pipeline_state_dirty = True

            for patient_id, fingerprint in current_snapshots.items():
                if previous_snapshots.get(patient_id) == fingerprint:
                    stable_counts[patient_id] = stable_counts.get(patient_id, 1) + 1
                else:
                    stable_counts[patient_id] = 1

                if (
                    fingerprint != last_synced_snapshots.get(patient_id)
                    and stable_counts[patient_id] >= stable_polls
                ):
                    LOGGER.info(
                        "portal_raw_sync_change_detected",
                        patient_label=patient_id,
                        fingerprint=fingerprint,
                    )
                    if spawn_portal_sync(patient_id):
                        last_synced_snapshots[patient_id] = fingerprint

                if (
                    local_pipeline_watcher
                    and fingerprint != last_pipeline_snapshots.get(patient_id)
                    and stable_counts[patient_id] >= stable_polls
                ):
                    patient_dir = root_dir / patient_id
                    missing_complete, active_runs = _source_pdfs_missing_complete_runs(
                        patient_dir, patient_id
                    )
                    if missing_complete:
                        LOGGER.info(
                            "portal_local_pipeline_change_detected",
                            patient_label=patient_id,
                            fingerprint=fingerprint,
                            missing_reports=missing_complete,
                            active_reports=active_runs,
                        )
                        if spawn_portal_pipeline(patient_id):
                            last_pipeline_snapshots[patient_id] = fingerprint
                            pipeline_state_dirty = True
                    elif not active_runs:
                        last_pipeline_snapshots[patient_id] = fingerprint
                        pipeline_state_dirty = True

            previous_snapshots = current_snapshots
            if local_pipeline_watcher and pipeline_state_dirty:
                _write_pipeline_watch_state(
                    pipeline_state_path, last_pipeline_snapshots
                )
            await asyncio.sleep(poll_interval_s)
    except asyncio.CancelledError:
        LOGGER.info("portal_raw_sync_watcher_stopped", root_dir=str(root_dir))
        raise


def _main() -> int:
    parser = argparse.ArgumentParser(description="Sync one portal patient to Thrylen.")
    parser.add_argument("--patient-label", required=True, help="Portal patient label")
    args = parser.parse_args()
    return 0 if sync_patient_to_thrylen(args.patient_label) else 1


if __name__ == "__main__":
    raise SystemExit(_main())
