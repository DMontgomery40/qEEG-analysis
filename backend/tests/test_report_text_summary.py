"""Unit tests for deterministic PAGE-1 summary extraction."""

from __future__ import annotations


def test_state_ratio_repairs_dropped_decimal_ocr_artifact(temp_data_dir):
    from backend.council.report_text import _facts_from_report_text_summary

    report_text = (
        "=== PAGE 1 / 15 ===\n"
        "F3/F4 Eyes Closed Alpha (Power) 0.7 1.0 11 0.9-1.1\n"
    )

    facts = _facts_from_report_text_summary(report_text, expected_sessions=[1, 2, 3])
    vals = {
        f["session_index"]: f["value"]
        for f in facts
        if f.get("fact_type") == "state_metric" and f.get("metric") == "f3_f4_alpha_ratio_ec"
    }
    shown = {
        f["session_index"]: f.get("shown_as")
        for f in facts
        if f.get("fact_type") == "state_metric" and f.get("metric") == "f3_f4_alpha_ratio_ec"
    }

    assert vals == {1: 0.7, 2: 1.0, 3: 1.1}
    assert shown == {1: None, 2: None, 3: None}
