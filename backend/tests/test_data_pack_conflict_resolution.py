import json

import pytest

from backend.council.constants import DATA_PACK_SCHEMA_VERSION
from backend.council.types import PageImage
from backend.council.workflow.data_pack import _DataPackMixin
from backend.tests.test_data_pack_timeouts import (
    _TimeoutCapturingDataPack,
    _create_run_with_report,
)


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
            "local_session_index": 1,
            "session_index_namespace": "global",
            "source_page": 1,
            "value": 283,
        },
        {
            "fact_type": "performance_metric",
            "metric": "physical_reaction_time",
            "session_index": 2,
            "local_session_index": 2,
            "session_index_namespace": "global",
            "source_page": 1,
            "value": 280,
        },
        {
            "fact_type": "performance_metric",
            "metric": "physical_reaction_time",
            "session_index": 3,
            "session_index_namespace": "global",
            "local_session_index": 2,
            "source_page": 18,
            "value": 275,
        },
    ]


def test_overlapping_aliases_are_idempotent_and_preserve_explicit_namespaces():
    aliases = {1: {1: 2, 2: 3}}
    local = [
        dict(
            fact_type="evoked_potential",
            metric="audio_p300_delay",
            value=value,
            session_index=session,
            source_page=1,
            session_index_namespace="local",
        )
        for session, value in [(1, 300), (2, 290)]
    ]
    normalize = _DataPackMixin._normalize_facts_for_page_session_aliases
    canonical = normalize(local, page_session_aliases=aliases)
    assert [f["session_index"] for f in canonical] == [2, 3]
    assert normalize(canonical, page_session_aliases=aliases) == canonical
    assert [f["local_session_index"] for f in canonical] == [1, 2]
    assert all(f["session_index_namespace"] == "global" for f in canonical)
    assert [f["session_index"] for f in local] == [1, 2]

    global_model = [
        dict(f, session_index=s, session_index_namespace="global")
        for f, s in zip(local, [2, 3])
    ]
    assert [
        f["session_index"]
        for f in normalize(global_model, page_session_aliases=aliases)
    ] == [2, 3]


def test_alias_normalization_keeps_conflicting_repeated_visit_values():
    facts = [
        dict(
            fact_type="evoked_potential",
            metric="audio_p300_delay",
            value=value,
            session_index=1,
            source_page=page,
            session_index_namespace="local",
        )
        for page, value in [(1, 300), (2, 301)]
    ]
    normalized = _DataPackMixin._normalize_facts_for_page_session_aliases(
        facts, page_session_aliases={1: {1: 2}, 2: {1: 2}}
    )
    assert [f["value"] for f in normalized] == [300, 301]
    assert len(_DataPackMixin._find_fact_conflicts(normalized)) == 1


class _AliasPackHarness(_TimeoutCapturingDataPack):
    _missing_required_fields = staticmethod(_DataPackMixin._missing_required_fields)

    def __init__(self, responses):
        super().__init__()
        self.responses = iter(responses)

    async def _call_model_multimodal(self, **kwargs):
        return json.dumps(
            {
                "schema_version": DATA_PACK_SCHEMA_VERSION,
                "pages_seen": [1],
                "facts": next(self.responses),
            }
        )


@pytest.mark.asyncio
async def test_global_model_facts_survive_retries_and_repeated_cache_upgrades(
    temp_data_dir,
):
    from backend.council.paths import _data_pack_path

    text = (
        "=== PAGE 1 / 1 ===\n[[QEEG_SESSION_ALIAS local=1 global=2]]\n"
        "[[QEEG_SESSION_ALIAS local=2 global=3]]\n"
        "Audio P300 Delay\nP300 Rare Comparison\n"
    )
    delay2 = dict(
        fact_type="evoked_potential",
        metric="audio_p300_delay",
        session_index=2,
        source_page=1,
        value=290,
    )
    cp3 = dict(
        fact_type="p300_cp_site",
        site="C3",
        session_index=3,
        source_page=1,
        uv=8.1,
        ms=285,
        **{"yield": 40},
    )
    delay3 = dict(delay2, session_index=3, value=285)
    workflow = _AliasPackHarness([[delay2], [cp3], [delay3]])
    run_id, report = _create_run_with_report(text)
    args = dict(
        run_id=run_id,
        report=report,
        report_text=text,
        page_images=[PageImage(page=1, base64_png="ZmFrZQ==")],
        candidate_extractor_model_ids=["vision-model"],
        strict=False,
    )
    pack = await workflow._ensure_data_pack(**args)
    assert {
        (f["fact_type"], f["session_index"], f.get("value")) for f in pack["facts"]
    } == {
        ("evoked_potential", 2, 290),
        ("evoked_potential", 3, 285),
        ("p300_cp_site", 3, None),
    }
    assert all(f["session_index_namespace"] == "global" for f in pack["facts"])
    for _ in range(2):
        assert await workflow._ensure_data_pack(**args) == pack

    # Older saved model packs already use the prompt's global namespace, even
    # when they predate the explicit per-fact marker. Upgrade only this run.
    legacy = dict(
        pack,
        facts=[
            {k: v for k, v in f.items() if k != "session_index_namespace"}
            for f in pack["facts"]
        ],
    )
    _data_pack_path(run_id).write_text(json.dumps(legacy), encoding="utf-8")
    assert await workflow._ensure_data_pack(**args) == pack


@pytest.mark.asyncio
async def test_conflicting_source_summaries_still_fail_strict_mode(temp_data_dir):
    text = (
        "=== PAGE 1 / 2 ===\n[[QEEG_SESSION_ALIAS local=1 global=2]]\nAudio P300 Delay 290 ms 257-333 ms\n"
        "=== PAGE 2 / 2 ===\n[[QEEG_SESSION_ALIAS local=1 global=2]]\nAudio P300 Delay 291 ms 257-333 ms\n"
    )
    run_id, report = _create_run_with_report(text)
    # All other numeric gates are already exercised by the required-fields suite;
    # isolate the real strict conflict path while the model supplies no facts.
    workflow = _TimeoutCapturingDataPack()
    with pytest.raises(RuntimeError, match="conflict"):
        await workflow._ensure_data_pack(
            run_id=run_id,
            report=report,
            report_text=text,
            page_images=[
                PageImage(page=1, base64_png="ZmFrZQ=="),
                PageImage(page=2, base64_png="ZmFrZQ=="),
            ],
            candidate_extractor_model_ids=["vision-model"],
            strict=True,
        )


@pytest.mark.asyncio
async def test_repeated_deterministic_visit_facts_are_stable_across_cache_reads(
    temp_data_dir,
):
    text = (
        "=== PAGE 1 / 2 ===\n[[QEEG_SESSION_ALIAS local=1 global=2]]\nAudio P300 Delay 290 ms 257-333 ms\n"
        "=== PAGE 2 / 2 ===\n[[QEEG_SESSION_ALIAS local=1 global=2]]\n"
        "[[QEEG_SESSION_ALIAS local=2 global=3]]\nAudio P300 Delay 290 ms 285 ms 257-333 ms\n"
    )
    run_id, report = _create_run_with_report(text)
    workflow = _TimeoutCapturingDataPack()
    args = dict(
        run_id=run_id,
        report=report,
        report_text=text,
        page_images=[
            PageImage(page=1, base64_png="ZmFrZQ=="),
            PageImage(page=2, base64_png="ZmFrZQ=="),
        ],
        candidate_extractor_model_ids=["vision-model"],
        strict=True,
    )
    pack = await workflow._ensure_data_pack(**args)
    assert [(f["session_index"], f["value"]) for f in pack["facts"]] == [
        (2, 290),
        (3, 285),
    ]
    for _ in range(2):
        assert await workflow._ensure_data_pack(**args) == pack


def test_global_facts_preserve_source_local_index_without_remapping():
    fact = dict(
        fact_type="evoked_potential",
        metric="audio_p300_delay",
        session_index=2,
        source_page=7,
        value=290,
    )
    normalized = _DataPackMixin._normalize_facts_for_page_session_aliases(
        [fact], page_session_aliases={7: {1: 2, 2: 3}}, input_namespace="global"
    )
    assert normalized[0]["session_index"] == 2
    assert normalized[0]["local_session_index"] == 1
    assert normalized[0]["source_page"] == 7
    assert normalized[0]["value"] == 290


_NUMERIC_DETAIL_CASES = [
    pytest.param(
        dict(
            fact_type="performance_metric", metric="physical_reaction_time", value=280
        ),
        "sd_plus_minus",
        20,
        id="reaction-sd",
    ),
    pytest.param(
        dict(fact_type="n100_central_frontal_average", uv=-4.4, ms=120),
        "yield",
        36,
        id="n100-yield",
    ),
    pytest.param(
        dict(fact_type="p300_cp_site", site="C3", uv=8.1, ms=285),
        "yield",
        40,
        id="p300-yield",
    ),
    pytest.param(
        dict(
            fact_type="n100_central_frontal_average", uv=None, ms=None, shown_as="N/A"
        ),
        "yield",
        36,
        id="n100-na-yield",
    ),
    pytest.param(
        dict(fact_type="p300_cp_site", site="C3", uv=None, ms=None, shown_as="N/A"),
        "yield",
        40,
        id="p300-na-yield",
    ),
]


@pytest.mark.parametrize("base,field,value", _NUMERIC_DETAIL_CASES)
@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("different", [False, True])
def test_semantic_duplicates_compare_sd_and_yield(
    base, field, value, reverse, different
):
    facts = [
        dict(
            base,
            session_index=1,
            session_index_namespace="local",
            source_page=page,
            **{field: number},
        )
        for page, number in [(1, value), (2, value + 1 if different else str(value))]
    ]
    if reverse:
        facts.reverse()
    aliases = {1: {1: 2}, 2: {1: 2}}
    normalize = _DataPackMixin._normalize_facts_for_page_session_aliases
    canonical = normalize(facts, page_session_aliases=aliases)
    assert [f[field] for f in canonical] == [
        f[field] for f in (facts if different else facts[:1])
    ]
    assert len(_DataPackMixin._find_fact_conflicts(canonical)) == int(different)
    assert normalize(canonical, page_session_aliases=aliases) == canonical
    assert all(f["session_index"] == 2 for f in canonical)


class _NumericDetailPackHarness(_TimeoutCapturingDataPack):
    def __init__(self, facts):
        super().__init__()
        self.facts = facts

    async def _call_model_multimodal(self, **kwargs):
        return json.dumps(
            {
                "schema_version": DATA_PACK_SCHEMA_VERSION,
                "pages_seen": [1, 2],
                "facts": self.facts,
            }
        )


@pytest.mark.parametrize("base,field,value", _NUMERIC_DETAIL_CASES)
@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("different", [False, True])
@pytest.mark.asyncio
async def test_sd_yield_preservation_in_fresh_and_cached_packs(
    temp_data_dir,
    base,
    field,
    value,
    reverse,
    different,
):
    # The known-global model boundary supplies facts from two sources. The real
    # pack/cache pipeline decides equality and strict conflict handling.
    facts = [
        dict(base, source_page=page, session_index=2, **{field: number})
        for page, number in [(1, value), (2, value + int(different))]
    ]
    if reverse:
        facts.reverse()
    text = (
        "=== PAGE 1 / 2 ===\n[[QEEG_SESSION_ALIAS local=1 global=2]]\n"
        "=== PAGE 2 / 2 ===\n[[QEEG_SESSION_ALIAS local=1 global=2]]\n"
    )
    run_id, report = _create_run_with_report(text)
    workflow = _NumericDetailPackHarness(facts)
    args = dict(
        run_id=run_id,
        report=report,
        report_text=text,
        page_images=[
            PageImage(page=1, base64_png="ZmFrZQ=="),
            PageImage(page=2, base64_png="ZmFrZQ=="),
        ],
        candidate_extractor_model_ids=["vision-model"],
    )
    pack = await workflow._ensure_data_pack(**args, strict=False)
    assert [f[field] for f in pack["facts"]] == [
        f[field] for f in (facts if different else facts[:1])
    ]
    for _ in range(2):
        assert await workflow._ensure_data_pack(**args, strict=False) == pack
    if different:
        with pytest.raises(RuntimeError, match="(?i)conflict"):
            await workflow._ensure_data_pack(**args, strict=True)
    else:
        assert await workflow._ensure_data_pack(**args, strict=True) == pack


@pytest.mark.parametrize("base,field,value", _NUMERIC_DETAIL_CASES)
def test_deterministic_same_page_precedence_keeps_sd_yield_conflicts_out(
    base, field, value
):
    deterministic = dict(
        base,
        session_index=2,
        source_page=1,
        extraction_method="deterministic_report_text",
        **{field: value},
    )
    model = dict(deterministic, extraction_method="vision_llm", **{field: value + 1})
    assert len(_DataPackMixin._find_fact_conflicts([deterministic, model])) == 1
    kept = _DataPackMixin._filter_shadowed_facts([deterministic], [model])
    assert kept == []
    assert _DataPackMixin._find_fact_conflicts([deterministic] + kept) == []


@pytest.mark.parametrize("base,field,value", _NUMERIC_DETAIL_CASES)
@pytest.mark.parametrize("reverse", [False, True])
def test_missing_optional_detail_retains_recorded_value_without_inventing_conflict(
    base, field, value, reverse
):
    facts = [
        dict(base, session_index=2, source_page=1),
        dict(base, session_index=2, source_page=2, **{field: value}),
    ]
    if reverse:
        facts.reverse()
    canonical = _DataPackMixin._normalize_facts_for_page_session_aliases(
        facts, page_session_aliases={1: {1: 2}, 2: {1: 2}}, input_namespace="global"
    )
    assert len(canonical) == 2
    assert [f[field] for f in canonical if field in f] == [value]
    assert _DataPackMixin._find_fact_conflicts(canonical) == []
