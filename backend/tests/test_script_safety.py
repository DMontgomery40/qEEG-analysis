from __future__ import annotations

import os
import json
from pathlib import Path

import pytest


def test_create_combined_report_rejects_duplicate_patient_labels(temp_data_dir):
    from backend import storage
    from scripts import create_combined_council_report as script

    with storage.session_scope() as session:
        storage.create_patient(session, label="HT_09-05-1954", notes="")
        storage.create_patient(session, label="HT_09-05-1954", notes="")

    with pytest.raises(RuntimeError, match="Multiple patients found for label"):
        script._patient_id_for_label("HT_09-05-1954")


def test_single_agent_auto_discovery_rejects_ambiguous_matches(tmp_path: Path):
    from scripts import generate_single_agent_patient_report as script

    source_a = tmp_path / "session-a.pdf"
    source_b = tmp_path / "session-b.pdf"
    source_a.write_text("a", encoding="utf-8")
    source_b.write_text("b", encoding="utf-8")

    manifest_path = tmp_path / "combined.manifest.json"
    manifest = {
        "patient_label": "HT_09-05-1954",
        "sources": [
            {"path": str(source_a)},
            {"path": str(source_b)},
        ],
    }
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    reports_root = tmp_path / "reports"
    for idx in (1, 2):
        report_dir = reports_root / f"patient-{idx}" / f"report-{idx}"
        report_dir.mkdir(parents=True, exist_ok=True)
        metadata = {
            "synthetic_combined": {
                "source_files": [
                    {"path": str(source_a.resolve())},
                    {"path": str(source_b.resolve())},
                ]
            }
        }
        (report_dir / "metadata.json").write_text(
            json.dumps(metadata), encoding="utf-8"
        )

    with pytest.raises(
        RuntimeError, match="Multiple combined report directories match"
    ):
        script._find_combined_report_dir(
            manifest=manifest,
            manifest_path=manifest_path,
            reports_root=reports_root,
        )


def test_data_dir_does_not_follow_the_working_directory(tmp_path, monkeypatch):
    """Six patient rows reached the clinic's live database because DATA_DIR was
    relative: a suite run with the wrong working directory wrote production.

    The path has to be anchored to the repo, so where the process was launched
    from cannot decide which clinic's data it opens.
    """
    import subprocess
    import sys
    from pathlib import Path

    repo_root = Path(__file__).resolve().parents[2]
    probe = "import backend.config as c; print(c.DATA_DIR)"

    from_repo = subprocess.run(
        [sys.executable, "-c", probe],
        cwd=repo_root, capture_output=True, text=True,
        env={"PATH": os.environ.get("PATH", ""), "PYTHONPATH": str(repo_root)},
    )
    from_elsewhere = subprocess.run(
        [sys.executable, "-c", probe],
        cwd=tmp_path, capture_output=True, text=True,
        env={"PATH": os.environ.get("PATH", ""), "PYTHONPATH": str(repo_root)},
    )

    assert from_repo.returncode == 0, from_repo.stderr
    assert from_elsewhere.returncode == 0, from_elsewhere.stderr
    assert from_repo.stdout.strip() == from_elsewhere.stdout.strip()
    assert Path(from_elsewhere.stdout.strip()) == repo_root / "data"
    # And nothing was created in the directory we happened to run from.
    assert not (tmp_path / "data").exists()


def test_an_explicit_data_dir_still_wins():
    import subprocess
    import sys
    from pathlib import Path

    repo_root = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        [sys.executable, "-c", "import backend.config as c; print(c.DATA_DIR)"],
        cwd=repo_root, capture_output=True, text=True,
        env={
            "PATH": os.environ.get("PATH", ""),
            "PYTHONPATH": str(repo_root),
            "DATA_DIR": "/tmp/somewhere-else",
        },
    )

    assert result.stdout.strip() == "/tmp/somewhere-else"
