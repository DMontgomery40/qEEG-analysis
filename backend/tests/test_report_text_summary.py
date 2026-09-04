"""Unit tests for deterministic PAGE-1 summary extraction."""

from __future__ import annotations

import pytest


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


@pytest.mark.parametrize("reverse", [False, True])
def test_summary_uses_each_source_page_columns_and_global_aliases(reverse):
    from backend.council.report_text import _facts_from_report_text_summary

    sources = [
        ([(1, 1), (2, 2)], [283, 280], [300, 290]),
        ([(1, 2), (2, 3)], [280, 275], [290, 285]),
        ([(1, 4)], [270], [280]),
    ]
    if reverse:
        sources.reverse()
    pages = []
    expected = []
    for page, (aliases, reaction, delay) in enumerate(sources, 1):
        pages.append(
            f"=== PAGE {page} / 3 ===\n"
            + "".join(
                f"[[QEEG_SESSION_ALIAS local={local} global={glob}]]\n"
                for local, glob in aliases
            )
            + "Physical Reaction Time "
            + " ".join(f"{v} (+20) ms" for v in reaction)
            + " 255-367 ms\nAudio P300 Delay "
            + " ".join(f"{v} ms" for v in delay)
            + " 257-333 ms\n"
        )
        expected.extend(
            (page, local, glob, rt, p300)
            for (local, glob), rt, p300 in zip(aliases, reaction, delay)
        )
    facts = _facts_from_report_text_summary(
        "\n".join(pages), expected_sessions=[1, 2, 3, 4]
    )
    for page, local, glob, rt, p300 in expected:
        selected = {
            f["metric"]: f
            for f in facts
            if f["source_page"] == page and f["session_index"] == glob
        }
        assert selected["physical_reaction_time"]["value"] == rt
        assert selected["physical_reaction_time"]["sd_plus_minus"] == 20
        assert selected["audio_p300_delay"]["value"] == p300
        assert all(
            f["local_session_index"] == local
            and f["session_index_namespace"] == "global"
            for f in selected.values()
        )
    assert len(facts) == 10


@pytest.mark.parametrize("with_sd", [False, True])
@pytest.mark.parametrize("count", [1, 2, 3])
def test_reaction_time_column_count_never_consumes_sd_or_target(with_sd, count):
    from backend.council.report_text import _facts_from_report_text_summary

    values = [283, 280, 275][:count]
    header = "".join(f"Session {i}\n" for i in range(1, count + 1))
    row = (
        "Physical Reaction Time "
        + " ".join(f"{v} (+20) ms" if with_sd else f"{v} ms" for v in values)
        + " 255-367 ms\n"
    )
    facts = _facts_from_report_text_summary(
        header + row, expected_sessions=[1, 2, 3, 4]
    )
    assert [f["value"] for f in facts] == values


def test_n100_extracts_local_rows_on_later_source_pages():
    from backend.council.report_text import _facts_from_report_text_n100_central_frontal

    text = (
        "=== PAGE 1 / 3 ===\nIntroduction\n"
        "=== PAGE 2 / 3 ===\n[[QEEG_SESSION_ALIAS local=1 global=2]]\n"
        "[[QEEG_SESSION_ALIAS local=2 global=3]]\n"
        "CENTRAL-FRONTAL AVERAGE\nN100-UV MS\n36 -4.4 120\n37 -5.4 110\n"
        "=== PAGE 3 / 3 ===\n[[QEEG_SESSION_ALIAS local=1 global=4]]\n"
        "CENTRAL-FRONTAL AVERAGE\nN100-UV MS\n38 -6.4 100\n"
    )
    facts = _facts_from_report_text_n100_central_frontal(
        text, expected_sessions=[1, 2, 3, 4]
    )
    assert [
        (
            f["source_page"],
            f["local_session_index"],
            f["session_index"],
            f["uv"],
            f["ms"],
        )
        for f in facts
    ] == [(2, 1, 2, -4.4, 120), (2, 2, 3, -5.4, 110), (3, 1, 4, -6.4, 100)]


@pytest.mark.parametrize("count", [1, 2])
def test_all_summary_metrics_keep_source_values_on_shorter_later_page(count):
    from backend.council.report_text import _facts_from_report_text_summary

    rows = [
        (
            "Trail Making Test A",
            ["23 sec", "24 sec"],
            "25-39 sec",
            "trail_making_test_a",
            [23, 24],
        ),
        (
            "Trail Making Test B",
            ["49 sec", "50 sec"],
            "55-85 sec",
            "trail_making_test_b",
            [49, 50],
        ),
        (
            "Audio P300 Delay",
            ["285 ms", "280 ms"],
            "257-333 ms",
            "audio_p300_delay",
            [285, 280],
        ),
        (
            "Audio P300 Voltage",
            ["8.1 uV", "9.2 uV"],
            "5-20 uV",
            "audio_p300_voltage",
            [8.1, 9.2],
        ),
        (
            "CZ Eyes Closed Theta/Beta",
            ["2.1", "N/A"],
            "1-3",
            "cz_theta_beta_ratio_ec",
            [2.1, None],
        ),
        (
            "F3/F4 Eyes Closed Alpha",
            ["0.9", "1.1"],
            "0.9-1.1",
            "f3_f4_alpha_ratio_ec",
            [0.9, 1.1],
        ),
        (
            "Frontal",
            ["10.1 Hz", "10.2 Hz"],
            "8-12 Hz",
            "frontal_peak_frequency_ec",
            [10.1, 10.2],
        ),
        (
            "Central-Parietal",
            ["10.3 Hz", "10.4 Hz"],
            "8-12 Hz",
            "central_parietal_peak_frequency_ec",
            [10.3, 10.4],
        ),
        (
            "Occipital",
            ["10.5 Hz", "N/A"],
            "8-12 Hz",
            "occipital_peak_frequency_ec",
            [10.5, None],
        ),
    ]
    text = "=== PAGE 1 / 2 ===\nCover\n=== PAGE 2 / 2 ===\n"
    text += "".join(
        f"[[QEEG_SESSION_ALIAS local={i} global={i + 2}]]\n"
        for i in range(1, count + 1)
    )
    text += "\n".join(
        label + " " + " ".join(cells[:count]) + " " + target
        for label, cells, target, _, _ in rows
    )
    facts = _facts_from_report_text_summary(text, expected_sessions=[1, 2, 3, 4])
    assert len(facts) == len(rows) * count
    for _, _, _, metric, values in rows:
        selected = [f for f in facts if f["metric"] == metric]
        assert [f["value"] for f in selected] == values[:count]
        assert [f["session_index"] for f in selected] == [3, 4][:count]
        assert all(f["source_page"] == 2 for f in selected)


def test_local_column_order_comes_from_source_legend():
    from backend.council.report_text import _facts_from_report_text_summary

    text = (
        "[[QEEG_SESSION_ALIAS local=1 global=2]]\n[[QEEG_SESSION_ALIAS local=2 global=3]]\n"
        "Session 2 (newer) Session 1 (older)\nAudio P300 Delay 280 ms 290 ms 257-333 ms\n"
    )
    facts = _facts_from_report_text_summary(text, expected_sessions=[1, 2, 3])
    assert [
        (f["local_session_index"], f["session_index"], f["value"]) for f in facts
    ] == [(2, 3, 280), (1, 2, 290)]
