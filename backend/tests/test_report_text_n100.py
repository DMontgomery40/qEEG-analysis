"""Unit tests for deterministic report-text extraction helpers."""

from __future__ import annotations


def test_n100_central_frontal_ms_normalizes_common_ocr_extra_digit(temp_data_dir):
    """OCR sometimes turns '120' into '1230'; normalization should recover 120."""
    from backend.council.report_text import _facts_from_report_text_n100_central_frontal

    report_text = (
        "=== PAGE 2 / 13 ===\n"
        "CENTRAL-FRONTAL AVERAGE\n"
        "# N100-UV MS\n"
        "\n"
        "38 -5.8 1230\n"
        "38 -5.7 112\n"
        "\n"
        "Blue line indicates 100 msec post stimulus.\n"
        "Maximum N100 reported between 30-120 msec.\n"
    )

    facts = _facts_from_report_text_n100_central_frontal(report_text, expected_sessions=[1, 2])

    assert len(facts) == 2
    assert facts[0]["ms"] == 120
    assert facts[0]["uv"] == -5.8
    assert facts[1]["ms"] == 112

