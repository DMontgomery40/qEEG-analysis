from backend.council.workflow.data_pack import _DataPackMixin


def test_filter_shadowed_facts_prefers_deterministic_and_avoids_conflicts():
    det = [
        {
            "extraction_method": "deterministic_report_text",
            "fact_type": "state_metric",
            "metric": "f3_f4_alpha_ratio_ec",
            "session_index": 1,
            "source_page": 1,
            "unit": "ratio",
            "value": 11.0,
        }
    ]
    vision = [
        {
            "extraction_method": "vision_llm",
            "fact_type": "state_metric",
            "metric": "f3_f4_alpha_ratio_ec",
            "session_index": 1,
            "source_page": 1,
            "unit": "ratio",
            "value": 1.1,
        }
    ]

    # Baseline: these disagree and would be treated as a hard conflict in strict mode.
    conflicts = _DataPackMixin._find_fact_conflicts(det + vision)
    assert len(conflicts) == 1

    # After filtering, deterministic shadows the redundant vision duplicate.
    filtered_vision = _DataPackMixin._filter_shadowed_facts(det, vision)
    assert filtered_vision == []
    assert _DataPackMixin._find_fact_conflicts(det + filtered_vision) == []


def test_page_session_aliases_remap_and_drop_restarted_series_duplicates():
    facts = [
        {
            "fact_type": "performance_metric",
            "metric": "physical_reaction_time",
            "session_index": 1,
            "source_page": 1,
            "value": 283,
        },
        {
            "fact_type": "performance_metric",
            "metric": "physical_reaction_time",
            "session_index": 2,
            "source_page": 1,
            "value": 280,
        },
        {
            "fact_type": "performance_metric",
            "metric": "physical_reaction_time",
            "session_index": 1,
            "source_page": 18,
            "value": 280,
        },
        {
            "fact_type": "performance_metric",
            "metric": "physical_reaction_time",
            "session_index": 2,
            "source_page": 18,
            "value": 275,
        },
    ]

    normalized = _DataPackMixin._normalize_facts_for_page_session_aliases(
        facts,
        page_session_aliases={
            1: {1: 1, 2: 2},
            18: {1: 2, 2: 3},
        },
    )

    assert normalized == [
        {
            "fact_type": "performance_metric",
            "metric": "physical_reaction_time",
            "session_index": 1,
            "source_page": 1,
            "value": 283,
        },
        {
            "fact_type": "performance_metric",
            "metric": "physical_reaction_time",
            "session_index": 2,
            "source_page": 1,
            "value": 280,
        },
        {
            "fact_type": "performance_metric",
            "metric": "physical_reaction_time",
            "session_index": 3,
            "local_session_index": 2,
            "source_page": 18,
            "value": 275,
        },
    ]
