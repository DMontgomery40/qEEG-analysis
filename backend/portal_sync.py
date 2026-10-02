from __future__ import annotations

import argparse
import fcntl
import json
import os
import shutil
import subprocess

import sys
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from . import storage
from . import config as cfg
from .logging_utils import get_logger
from .portal_files import normalize_portal_patient_id

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
    return normalize_portal_patient_id(label)


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


def _sync_watch_state_path(root_dir: Path) -> Path:
    return root_dir / ".qeeg_portal_sync_watch_state.json"


def _sync_spawn_lock_path(root_dir: Path) -> Path:
    return root_dir / ".qeeg_portal_netlify_sync.spawn.lock"


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


def _persist_scoped_sync_progress(
    *,
    state_path: Path,
    base_state: dict[str, Any],
    temp_state_path: Path,
    patient_id: str,
) -> dict[str, Any]:
    synced_state = _load_sync_state(temp_state_path)
    merged_state = _merge_sync_state_for_patient(
        base_state, synced_state, patient_id
    )
    _write_sync_state(state_path, merged_state)
    return merged_state


def _mark_patient_sync_retryable(root_dir: Path, patient_id: str) -> None:
    watch_state_path = _sync_watch_state_path(root_dir)
    if not watch_state_path.exists():
        return
    watch_state = _load_pipeline_watch_state(watch_state_path)
    if patient_id not in watch_state:
        return
    watch_state.pop(patient_id, None)
    _write_pipeline_watch_state(watch_state_path, watch_state)


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
def _sync_lock(root_dir: Path) -> Iterator[bool]:
    lock_path = root_dir / ".qeeg_portal_netlify_sync.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        timeout_s = float(
            os.getenv("QEEG_PORTAL_SYNC_LOCK_TIMEOUT_S", "30") or "30"
        )
    except Exception:
        timeout_s = 30.0
    if timeout_s < 0:
        timeout_s = 30.0

    with lock_path.open("a+", encoding="utf-8") as lock_file:
        deadline = time.monotonic() + timeout_s
        while True:
            try:
                fcntl.flock(
                    lock_file.fileno(),
                    fcntl.LOCK_EX | fcntl.LOCK_NB,
                )
                break
            except BlockingIOError:
                remaining_s = deadline - time.monotonic()
                if remaining_s <= 0:
                    yield False
                    return
                time.sleep(min(0.1, remaining_s))
        try:
            yield True
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def sync_patient_to_thrylen(patient_label: str) -> bool:
    if not _truthy_env("QEEG_PORTAL_NETLIFY_SYNC_ON_PUBLISH", False):
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

    with _sync_lock(root_dir) as lock_acquired:
        if not lock_acquired:
            _mark_patient_sync_retryable(root_dir, patient_id)
            LOGGER.warning(
                "portal_sync_lock_timed_out",
                patient_label=patient_id,
                operatorHint="Another portal sync exceeded the bounded lock wait; this patient remains retryable after the active sync releases the reservation.",
            )
            return False
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

            try:
                timeout_s = float(
                    os.getenv("QEEG_PORTAL_NETLIFY_SYNC_TIMEOUT_S", "900") or "900"
                )
            except Exception:
                timeout_s = 900.0
            if timeout_s <= 0:
                timeout_s = 900.0
            try:
                proc = subprocess.run(
                    command,
                    cwd=str(cwd),
                    capture_output=True,
                    text=True,
                    check=False,
                    timeout=timeout_s,
                )
            except subprocess.TimeoutExpired:
                _persist_scoped_sync_progress(
                    state_path=state_path,
                    base_state=base_state,
                    temp_state_path=temp_state_path,
                    patient_id=patient_id,
                )
                _mark_patient_sync_retryable(root_dir, patient_id)
                LOGGER.error(
                    "portal_sync_timed_out",
                    patient_label=patient_id,
                    timeout_s=timeout_s,
                    operatorHint="Single-patient Netlify sync exceeded its bounded runtime; the watcher will retry the latest patient-folder state after the active reservation is released.",
                )
                return False
            except Exception:
                _persist_scoped_sync_progress(
                    state_path=state_path,
                    base_state=base_state,
                    temp_state_path=temp_state_path,
                    patient_id=patient_id,
                )
                _mark_patient_sync_retryable(root_dir, patient_id)
                LOGGER.exception(
                    "portal_sync_process_failed",
                    patient_label=patient_id,
                    operatorHint="The single-patient sync process failed after partial progress was saved; the watcher will retry the remaining files.",
                )
                return False
            if proc.returncode != 0:
                _persist_scoped_sync_progress(
                    state_path=state_path,
                    base_state=base_state,
                    temp_state_path=temp_state_path,
                    patient_id=patient_id,
                )
                _mark_patient_sync_retryable(root_dir, patient_id)
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
    if not _truthy_env("QEEG_PORTAL_NETLIFY_SYNC_ON_PUBLISH", False):
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

    root_dir = portal_patients_dir()
    spawn_lock_path = _sync_spawn_lock_path(root_dir)
    spawn_lock_path.parent.mkdir(parents=True, exist_ok=True)
    spawn_lock_file = spawn_lock_path.open("a+", encoding="utf-8")
    try:
        fcntl.flock(
            spawn_lock_file.fileno(),
            fcntl.LOCK_EX | fcntl.LOCK_NB,
        )
    except BlockingIOError:
        spawn_lock_file.close()
        LOGGER.info(
            "portal_sync_already_running",
            patient_label=patient_id,
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
            pass_fds=(spawn_lock_file.fileno(),),
        )
    except Exception:
        spawn_lock_file.close()
        LOGGER.exception(
            "portal_sync_spawn_failed",
            patient_label=patient_id,
            operatorHint="Background portal sync spawn shells back into python -m backend.portal_sync; verify sys.executable, repo cwd, and local process launch permissions.",
        )
        return False
    spawn_lock_file.close()

    LOGGER.info("portal_sync_spawned", patient_label=patient_id)
    return True


def _main() -> int:
    parser = argparse.ArgumentParser(description="Sync one portal patient to Thrylen.")
    parser.add_argument("--patient-label", required=True, help="Portal patient label")
    args = parser.parse_args()
    return 0 if sync_patient_to_thrylen(args.patient_label) else 1


if __name__ == "__main__":
    raise SystemExit(_main())
