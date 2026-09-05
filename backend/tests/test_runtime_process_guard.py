from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest


@pytest.mark.parametrize(
    ("component", "expected_needle"),
    [
        ("portal_watcher", "qeeg_patients_watch.mjs --dir"),
        ("pipeline_worker", "scripts/portal_pipeline_worker.py --poll-seconds"),
        ("backend", "-m backend.main"),
        ("frontend", "frontend/node_modules/.bin/vite"),
    ],
)
def test_runtime_guard_maps_each_component_to_its_live_process_signature(
    tmp_path: Path,
    component: str,
    expected_needle: str,
):
    repo_root = Path(__file__).resolve().parents[2]
    guard_path = repo_root / "scripts" / "qeeg_runtime_guard.sh"
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_pgrep = fake_bin / "pgrep"
    fake_pgrep.write_text(
        "#!/bin/sh\n"
        'case "$*" in\n'
        '  *"$FAKE_PROCESS_NEEDLE"*) echo 123; exit 0 ;;\n'
        "  *) exit 1 ;;\n"
        "esac\n",
        encoding="utf-8",
    )
    fake_pgrep.chmod(0o755)
    fake_lsof = fake_bin / "lsof"
    fake_lsof.write_text('#!/bin/sh\nprintf "p123\\nn%s\\n" "$FAKE_PROCESS_CWD"\n')
    fake_lsof.chmod(0o755)

    env = dict(os.environ)
    env["FAKE_PROCESS_CWD"] = str(repo_root)
    env["PATH"] = f"{fake_bin}:{env.get('PATH', '')}"
    env["FAKE_PROCESS_NEEDLE"] = expected_needle
    command = (
        f'source "{guard_path}"; '
        f'qeeg_component_is_running "{component}" "{repo_root}" "{tmp_path}"'
    )

    result = subprocess.run(
        ["/bin/bash", "-c", command],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr


def test_runtime_guard_rejects_unknown_components(tmp_path: Path):
    repo_root = Path(__file__).resolve().parents[2]
    guard_path = repo_root / "scripts" / "qeeg_runtime_guard.sh"

    result = subprocess.run(
        [
            "/bin/bash",
            "-c",
            f'source "{guard_path}"; '
            f'qeeg_component_is_running "unknown" "{repo_root}" "{tmp_path}"',
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode != 0


def test_start_preflight_uses_runtime_guard_without_starting_services(
    tmp_path: Path,
):
    repo_root = Path(__file__).resolve().parents[2]
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_pgrep = fake_bin / "pgrep"
    fake_pgrep.write_text(
        "#!/bin/sh\n"
        'case "$*" in\n'
        "  *qeeg_patients_watch.mjs*) exit 0 ;;\n"
        "  *) exit 1 ;;\n"
        "esac\n",
        encoding="utf-8",
    )
    fake_pgrep.chmod(0o755)

    env = dict(os.environ)
    env["PATH"] = f"{fake_bin}:{env.get('PATH', '')}"
    env["QEEG_START_PREFLIGHT_ONLY"] = "1"
    env["QEEG_PORTAL_SYNC_DIR"] = str(tmp_path / "portal_patients")

    result = subprocess.run(
        ["/bin/bash", "start.sh"],
        cwd=repo_root,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == [
        "Starting qEEG Council...",
        "",
        "portal_watcher=running",
        "pipeline_worker=absent",
        "backend=absent",
        "frontend=absent",
    ]


@pytest.mark.parametrize("component", ["backend", "pipeline_worker"])
@pytest.mark.parametrize("cwd_kind", ["same", "other", "missing", "symlink"])
def test_runtime_guard_checks_process_checkout(tmp_path, component, cwd_kind):
    repo_root = Path(__file__).resolve().parents[2]
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    alias = tmp_path / "checkout alias"
    alias.symlink_to(repo_root, target_is_directory=True)
    cwd = {
        "same": str(repo_root),
        "other": str(tmp_path),
        "missing": "",
        "symlink": str(alias),
    }[cwd_kind]
    (fake_bin / "pgrep").write_text('#!/bin/sh\nprintf "123\\n456\\n"\n')
    (fake_bin / "lsof").write_text(
        '#!/bin/sh\n[ -n "$FAKE_PROCESS_CWD" ] && printf "p123\\nn%s\\n" "$FAKE_PROCESS_CWD"\n'
    )
    for executable in fake_bin.iterdir():
        executable.chmod(0o755)
    env = {
        **os.environ,
        "PATH": f"{fake_bin}:{os.environ['PATH']}",
        "FAKE_PROCESS_CWD": cwd,
    }
    result = subprocess.run(
        [
            "/bin/bash",
            "-c",
            'source "$1"; qeeg_component_is_running "$2" "$3" "$4"',
            "guard",
            str(repo_root / "scripts/qeeg_runtime_guard.sh"),
            component,
            str(repo_root),
            str(tmp_path),
        ],
        env=env,
        capture_output=True,
        text=True,
    )
    assert (result.returncode == 0) == (cwd_kind in ("same", "symlink")), result.stderr
