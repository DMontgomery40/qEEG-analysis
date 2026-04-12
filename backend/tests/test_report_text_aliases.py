from backend.council.report_text import (
    _expected_session_indices,
    _find_summary_pages,
    _page_session_alias_map,
)


def test_combined_report_aliases_drive_expected_sessions_and_summary_page_detection(
    temp_data_dir,
):
    report_text = (
        "=== PAGE 1 / 3 ===\n"
        "[[QEEG_SESSION_ALIAS local=1 global=1 date=2025-10-24]]\n"
        "[[QEEG_SESSION_ALIAS local=2 global=2 date=2025-11-14]]\n"
        "Physical Reaction Time 283 ms 280 ms 255-367 ms\n"
        "\n"
        "=== PAGE 2 / 3 ===\n"
        "[[QEEG_SESSION_ALIAS local=1 global=2 date=2025-11-14]]\n"
        "[[QEEG_SESSION_ALIAS local=2 global=3 date=2025-12-08]]\n"
        "Audio P300 Delay 300 ms 290 ms 257-333 ms\n"
        "\n"
        "=== PAGE 3 / 3 ===\n"
        "[[QEEG_SESSION_ALIAS local=1 global=2 date=2025-11-14]]\n"
        "Coherence matrix page\n"
    )

    assert _page_session_alias_map(report_text) == {
        1: {1: 1, 2: 2},
        2: {1: 2, 2: 3},
        3: {1: 2},
    }
    assert _expected_session_indices(report_text) == [1, 2, 3]
    assert _find_summary_pages(report_text, page_count=3) == [1, 2]


def test_expected_sessions_include_plain_markers_from_non_aliased_pages(temp_data_dir):
    report_text = (
        "=== PAGE 1 / 3 ===\n"
        "[[QEEG_SESSION_ALIAS local=1 global=4 date=2025-10-24]]\n"
        "Session 1 (10/24/2025)\n"
        "\n"
        "=== PAGE 2 / 3 ===\n"
        "Session 5 (11/14/2025)\n"
        "Audio P300 Delay 300 ms\n"
        "\n"
        "=== PAGE 3 / 3 ===\n"
        "Session 6 (12/08/2025)\n"
    )

    assert _expected_session_indices(report_text) == [4, 5, 6]


def test_expected_sessions_ignore_loose_ocr_marker_when_summary_pages_define_count(
    temp_data_dir,
):
    report_text = (
        "=== PAGE 1 / 5 ===\n"
        "Session 1 (2025-12-09)\n"
        "Session 2 (2026-01-21)\n"
        "Session 3 (2026-03-04)\n"
        "Physical Reaction Time 296 ms 293 ms 321 ms 252-362 ms\n"
        "Audio P300 Delay 276 ms 308 ms 320 ms 251-326 ms\n"
        "\n"
        "=== PAGE 2 / 5 ===\n"
        "P300 Rare Comparison\n"
        "\n"
        "=== PAGE 3 / 5 ===\n"
        "Spectrum page\n"
        "\n"
        "=== PAGE 4 / 5 ===\n"
        "Another non-summary page\n"
        "\n"
        "=== PAGE 5 / 5 ===\n"
        "Session 4 PERCENTAGE CHANGE COMPARED TO SESSION #1\n"
        "Coherence matrix page\n"
    )

    assert _find_summary_pages(report_text, page_count=5) == [1]
    assert _expected_session_indices(report_text) == [1, 2, 3]
