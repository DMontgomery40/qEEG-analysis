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


