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


def test_patient_facing_regeneration_uses_glm_53_flash_role_default():
    from scripts import generate_patient_facing_writeups as script

    assert script.DEFAULT_PATIENT_FACING_MODEL == "z-ai/glm-5.3-flash"


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


@pytest.mark.asyncio
async def test_shared_writeup_recipe_is_used_by_cli(
    temp_data_dir, tmp_path, monkeypatch
):
    """The standalone explicit-source action reaches the same shared recipe."""
    from scripts import generate_patient_facing_writeups as script

    source = tmp_path / "source.md"
    source.write_text("Authoritative council findings")
    received = []

    class Client:
        async def list_models(self):
            return ["writer"]

        async def aclose(self):
            pass

    monkeypatch.setattr(script, "AsyncOpenAICompatClient", lambda **kw: Client())

    async def shared(llm, **kwargs):
        received.append(kwargs)

    monkeypatch.setattr(script, "generate_writeup", shared)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "script",
            "--patient-label",
            "ZZ_01-01-1900",
            "--portal-dir",
            str(tmp_path),
            "--source-markdown",
            str(source),
            "--model",
            "writer",
            "--version",
            "explicit-v7",
            "--date",
            "2026-01-02",
            "--overwrite",
            "--no-sync",
        ],
    )
    assert await script.main() == 0
    assert len(received) == 1
    call = received[0]
    assert "Authoritative council findings" in call["prompt"]
    assert call["max_tokens"] == 12000
    assert call["overwrite"] and call["no_sync"]
    assert (
        call["md_path"].name
        == "ZZ_01-01-1900__patient-facing__explicit-v7__2026-01-02.md"
    )
