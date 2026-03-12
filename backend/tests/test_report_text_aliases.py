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
