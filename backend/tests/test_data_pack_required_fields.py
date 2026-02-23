from backend.council.workflow.data_pack import _DataPackMixin


def test_missing_required_allows_uv_lt0_ms_na_placeholders():
    """
    Some WAVi reports show Session-specific P300 CP values as placeholders like:
    - "UV <0" (no positive P300)
    - "MS N/A"
    We should treat these as "missing-at-source" (i.e., acceptable N/A) rather than failing
    required-field validation.
    """

    expected_sessions = [1]
    facts: list[dict] = []

    def add_fact(**kwargs):
        facts.append(kwargs)

    # Performance metrics (required).
    for metric in ("physical_reaction_time", "trail_making_test_a", "trail_making_test_b"):
        add_fact(
            fact_type="performance_metric",
            metric=metric,
            session_index=1,
            value=None,
            shown_as="N/A",
        )

    # Summary P300 (required).
    for metric in ("audio_p300_delay", "audio_p300_voltage"):
        add_fact(
            fact_type="evoked_potential",
            metric=metric,
            session_index=1,
            value=None,
            shown_as="N/A",
        )

    # Background EEG summary metrics (required).
    for metric in ("cz_theta_beta_ratio_ec", "f3_f4_alpha_ratio_ec"):
        add_fact(
            fact_type="state_metric",
            metric=metric,
            session_index=1,
            value=None,
            shown_as="N/A",
        )

    # Peak frequency by region (required).
    for metric in (
        "frontal_peak_frequency_ec",
        "central_parietal_peak_frequency_ec",
        "occipital_peak_frequency_ec",
    ):
        add_fact(
            fact_type="peak_frequency",
            metric=metric,
            session_index=1,
            value=None,
            shown_as="N/A",
        )

    # CP per-site P300 table (required).
    for site in ("C3", "CZ", "C4", "P3", "P4"):
        add_fact(
            fact_type="p300_cp_site",
            site=site,
            session_index=1,
            uv=None,
            ms=None,
            shown_as="N/A",
        )
    add_fact(
        fact_type="p300_cp_site",
        site="PZ",
        session_index=1,
        uv=None,
        ms=None,
        shown_as="UV <0, MS N/A",
    )

    # CENTRAL-FRONTAL AVERAGE N100 (required).
    add_fact(
        fact_type="n100_central_frontal_average",
        session_index=1,
        uv=None,
        ms=None,
        shown_as="N/A",
    )

    data_pack = {"facts": facts}
    missing = _DataPackMixin._missing_required_fields(data_pack, expected_sessions=expected_sessions)
    assert missing == set()


