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


@pytest.mark.parametrize("units", ["all", "none", "mixed"])
@pytest.mark.parametrize("with_sd", [False, True])
@pytest.mark.parametrize("count", [1, 2, 3])
def test_reaction_time_column_count_never_consumes_sd_or_target(with_sd, count, units):
    from backend.council.report_text import _facts_from_report_text_summary

    values = [283, 280, 275][:count]
    header = "".join(f"Session {i}\n" for i in range(1, count + 1))
    row = (
        "Physical Reaction Time "
        + " ".join(
            str(v)
            + (" (+20)" if with_sd else "")
            + (" ms" if units == "all" or (units == "mixed" and i % 2) else "")
            for i, v in enumerate(values)
        )
        + " 255-367 ms\n"
    )
    facts = _facts_from_report_text_summary(
        header + row, expected_sessions=[1, 2, 3, 4]
    )
    assert [f["value"] for f in facts] == values
    assert [f.get("sd_plus_minus") for f in facts] == (
        [20] if with_sd else [None]
    ) * count


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


# WAVi's glossary page (verbatim) mentions every summary label in prose. On
# 2026-09-29 it produced a phantom second P300 value (300 ms from "P300", 3 µV
# from "C3") for every session, so identical visits in two reports "conflicted".
WAVI_GLOSSARY_PAGE = (
    "P300 Metrics\n"
    "Physical Reaction Time: The average time of the physical response to rare tones, derived from mouse or keyboard input.\n"
    'Reported as "N/A" if there were less than 15 physical responses to rare tones.\n'
    "Audio P300 Delay and Audio P300 Voltage metrics are derived from Central-Parietal (C-P) locations CZ, C3, C4, PZ, P3, and P4 with\n"
    "sufficient yield.\n"
    "Audio P300 Delay: The fastest C-P latency between 240-499 ms after a rare tone, among locations that are at least 3 μV.\n"
    'Reported as "N/A" if no C-P location is at least 3 μV, or no C-P location has a yield of at least 20 rare events.\n'
    "Audio P300 Voltage: The largest C-P amplitude between 240-499 ms after a rare tone.\n"
    'Reported as "< 0 μV" if the voltage at all C-P locations is less than 0 μV.\n'
)


def _metric_values(facts, metric):
    out = {}
    for f in facts:
        if f.get("metric") == metric:
            out.setdefault(f["session_index"], []).append((f["value"], f.get("sd_plus_minus")))
    return out


def test_glossary_page_adds_no_summary_values():
    from backend.council.report_text import _facts_from_report_text_summary

    report_text = (
        "=== PAGE 1 / 12 ===\n"
        "Physical Reaction Time                282 (±56) ms        252–362 ms\n"
        "Audio P300 Delay                      344 ms              265–344 ms\n"
        "Audio P300 Voltage                    4.2 μV              7–18 μV\n"
        "=== PAGE 12 / 12 ===\n" + WAVI_GLOSSARY_PAGE
    )
    facts = _facts_from_report_text_summary(report_text, expected_sessions=[1])

    assert _metric_values(facts, "audio_p300_delay") == {1: [(344, None)]}
    assert _metric_values(facts, "audio_p300_voltage") == {1: [(4.2, None)]}
    assert _metric_values(facts, "physical_reaction_time") == {1: [(282, 56)]}


def test_plus_minus_sd_is_read_like_plus_sd():
    from backend.council.report_text import _facts_from_report_text_summary

    report_text = (
        "=== PAGE 1 / 2 ===\n"
        "Physical Reaction Time      282 (±56) ms      337 (+109) ms      252–362 ms\n"
    )
    facts = _facts_from_report_text_summary(report_text, expected_sessions=[1, 2])
    assert _metric_values(facts, "physical_reaction_time") == {1: [(282, 56)], 2: [(337, 109)]}


# Tesseract's reading of a real four-session WAVi summary (1/14/2026 report):
# the third cell's "±" came through as "£", the others as "+".
_FOUR_SESSION_RT_ROW = (
    "Physical Reaction Time 247 (+45) ms 249 (+48) ms 273 (£54) ms 270 (+62) ms 281-405 ms"
)


@pytest.mark.parametrize(
    "row",
    [
        _FOUR_SESSION_RT_ROW,
        *(
            _FOUR_SESSION_RT_ROW.replace("(+", f"({glyph}").replace("(£", f"({glyph}")
            for glyph in ["+", "£", "t", "=", "±", ""]
        ),
    ],
    ids=["ocr-row", "plus", "pound", "t", "equals", "plus-minus", "missing"],
)
def test_ocr_lookalike_plus_minus_keeps_each_sd_with_its_session(row):
    # The "£" cell was read as 273 with no SD, then 54 took session 4's column,
    # so a merge with a report printing session 4 as "270 (+62) ms" was refused.
    from backend.council.report_text import _facts_from_report_text_summary

    report_text = (
        "=== PAGE 1 / 17 ===\n"
        "Assessment Scores Session 1 Session 2 Session 3 Session 4 Target\n"
        "(10/24/2025) (11/14/2025) (12/8/2025) (1/14/2026) Range\n"
        "Performance Assessments\n" + row + "\n"
        "Trail Making Test A 63 sec 41 sec 50 sec 42 sec 43-74 sec\n"
    )
    facts = _facts_from_report_text_summary(report_text, expected_sessions=[1, 2, 3, 4])
    rt = [f for f in facts if f.get("metric") == "physical_reaction_time"]

    assert _metric_values(facts, "physical_reaction_time") == {
        1: [(247, 45)],
        2: [(249, 48)],
        3: [(273, 54)],
        4: [(270, 62)],
    }
    assert {f["target_range"] for f in rt} == {"281-405 ms"}


def test_ocr_low_yield_marker_fragment_does_not_hide_the_row():
    from backend.council.report_text import _facts_from_report_text_summary

    report_text = (
        "=== PAGE 1 / 12 ===\n"
        "Audio P300 Delay fl 284 ms f= 240 ms 240 ms 252-327 ms\n"
        "=== PAGE 12 / 12 ===\n" + WAVI_GLOSSARY_PAGE
    )
    facts = _facts_from_report_text_summary(report_text, expected_sessions=[1])
    assert _metric_values(facts, "audio_p300_delay") == {1: [(284, None)]}


def test_a_long_blank_run_after_a_label_is_read_in_linear_time():
    # The row check let "\s+" and "\s*" split one blank run every possible
    # way, so a label followed by W spaces and no value cost O(W^2): 16k
    # spaces took 1.3 s and 50k about 12 s, per label.
    import time

    from backend.council.report_text import _facts_from_report_text_summary

    report_text = "=== PAGE 1 / 1 ===\nPhysical Reaction Time" + " " * 50_000 + "x\n"
    started = time.perf_counter()
    facts = _facts_from_report_text_summary(report_text, expected_sessions=[1, 2])
    assert time.perf_counter() - started < 1.0
    assert _metric_values(facts, "physical_reaction_time") == {}
    padded = "=== PAGE 1 / 1 ===\nPhysical Reaction Time" + " " * 50_000 + "282 ms 252-362 ms\n"
    facts = _facts_from_report_text_summary(padded, expected_sessions=[1])
    assert _metric_values(facts, "physical_reaction_time") == {1: [(282, None)]}


def test_a_long_digit_run_in_a_p300_row_is_read_in_linear_time():
    # The reference-range search and the cell scan could start inside a digit
    # run at every position, so a P300 row with a long run of digits cost
    # O(n^2): 16k digits took about 5.5 s across the two patterns.
    import time

    from backend.council.report_text import _facts_from_report_text_summary

    report_text = (
        "=== PAGE 1 / 1 ===\nAudio P300 Delay 300 ms " + "1" * 20_000 + "a\n"
    )
    started = time.perf_counter()
    facts = _facts_from_report_text_summary(report_text, expected_sessions=[1])
    assert time.perf_counter() - started < 1.0
    assert _metric_values(facts, "audio_p300_delay") == {1: [(300, None)]}


def test_ocr_garbled_first_cell_keeps_the_rest_of_the_row():
    from backend.council.report_text import _facts_from_report_text_summary

    report_text = "=== PAGE 1 / 6 ===\nCZ Eyes Closed Theta/Beta (Power) i.1 1.2 0.9 0.8-1.9\n"
    facts = _facts_from_report_text_summary(report_text, expected_sessions=[1, 2, 3])
    vals = _metric_values(facts, "cz_theta_beta_ratio_ec")
    assert vals[2] == [(1.2, None)] and vals[3] == [(0.9, None)]


@pytest.mark.parametrize("metric,label,values,unit,target", [
    ("audio_p300_delay", "Audio P300 Delay", [344, None, 289], "ms", "265-344"),
    ("audio_p300_voltage", "Audio P300 Voltage", [4.2, None, 9.3], "uV", "7-18"),
])
@pytest.mark.parametrize("missing_index", [0, 1, 2])
@pytest.mark.parametrize("separator", ["-", "–", "—"])
def test_p300_missing_cell_keeps_session_and_reference(metric, label, values, unit, target, missing_index, separator):
    from backend.council.report_text import _facts_from_report_text_summary
    values = [344, 310, 289] if metric.endswith("delay") else [4.2, 6.5, 9.3]
    values[missing_index] = None
    target = target.replace("-", separator)
    text = "=== PAGE 1 / 1 ===\nSession 1 Session 2 Session 3\n" + label + " " + " ".join(
        "N/A" if v is None else f"{v} {unit}" for v in values
    ) + f" {target} {unit}\n"
    facts = _facts_from_report_text_summary(text, expected_sessions=[1, 2, 3])
    selected = {f["session_index"]: f for f in facts if f.get("metric") == metric}
    assert {i: selected.get(i, {}).get("value") for i in [1, 2, 3]} == dict(enumerate(values, 1))
    assert all(f["target_range"] == target.replace(separator, "-") + (" ms" if metric.endswith("delay") else " µV") for f in selected.values())
