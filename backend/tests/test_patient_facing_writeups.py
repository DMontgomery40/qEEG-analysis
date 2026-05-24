from __future__ import annotations

import asyncio
import sys


def test_patient_facing_writeups_require_db_source_by_default(
    temp_data_dir, tmp_path, monkeypatch, capsys
):
    from scripts import generate_patient_facing_writeups as script

    patient_label = "01-01-2001-0"
    patient_dir = tmp_path / patient_label
    patient_dir.mkdir(parents=True)
    (patient_dir / "legacy-council.md").write_text("# Legacy", encoding="utf-8")

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "generate_patient_facing_writeups.py",
            "--portal-dir",
            str(tmp_path),
            "--patient-label",
            patient_label,
            "--dry-run",
        ],
    )

    assert asyncio.run(script.main()) == 0
    out = capsys.readouterr().out

    assert "no delivery-ready DB run" in out
    assert "GENERATE" not in out


def test_patient_facing_writeups_can_opt_into_portal_markdown_fallback(
    temp_data_dir, tmp_path, monkeypatch, capsys
):
    from scripts import generate_patient_facing_writeups as script

    patient_label = "01-01-2001-0"
    patient_dir = tmp_path / patient_label
    patient_dir.mkdir(parents=True)
    (patient_dir / "legacy-council.md").write_text("# Legacy", encoding="utf-8")

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "generate_patient_facing_writeups.py",
            "--portal-dir",
            str(tmp_path),
            "--patient-label",
            patient_label,
            "--allow-portal-markdown-fallback",
            "--dry-run",
        ],
    )

    assert asyncio.run(script.main()) == 0
    out = capsys.readouterr().out

    assert "GENERATE 01-01-2001-0: 1 source reports" in out
