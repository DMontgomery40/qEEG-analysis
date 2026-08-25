from __future__ import annotations

import asyncio
import sys

import pytest


def test_patient_facing_writeups_require_db_source_by_default(
    temp_data_dir, tmp_path, monkeypatch, capsys
):
    from scripts import generate_patient_facing_writeups as script

    patient_label = "RS_01-01-2001"
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

    patient_label = "RS_01-01-2001"
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

    assert "GENERATE RS_01-01-2001: 1 source reports" in out


def test_patient_facing_regeneration_uses_ox_alpha_role_default():
    from scripts import generate_patient_facing_writeups as script

    assert script.DEFAULT_PATIENT_FACING_MODEL == "stealth/ox-alpha"


def test_patient_facing_writer_validates_all_required_sections():
    from scripts import generate_patient_facing_writeups as script

    complete = "\n".join(
        [
            "# Your Brain Assessment Summary",
            "Summary.",
            "## 2. Processing Speed and Attention",
            "Details.",
            "## 3. Cognitive Performance",
            "Details.",
            "## 4. Brain Rhythm Patterns",
            "Details.",
            "## 5. What This May Mean",
            "Details.",
            "# Technical Appendix",
            "Details.",
            "## Detailed P300 Site Data",
            "Details.",
            "## Coherence and Network Connectivity",
            "Details.",
            "## Spectral Power Summary",
            "Details.",
        ]
    )
    script._validate_patient_facing_markdown(complete)

    with pytest.raises(ValueError, match="## Spectral Power Summary"):
        script._validate_patient_facing_markdown(
            complete.replace("## Spectral Power Summary", "")
        )
