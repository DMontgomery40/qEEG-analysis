from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic_ai.models.function import FunctionModel
from pydantic_ai.models.test import ModelResponse, TestModel, ToolCallPart

from backend.council.ai_review_agents import (
    _cliproxy_openai_base_url,
    run_stage2_peer_review,
    run_stage5_final_review,
)


FIXTURE_DIR = Path(__file__).parent / "fixtures" / "ai_review"


def _fixture(name: str) -> dict:
    return json.loads((FIXTURE_DIR / name).read_text(encoding="utf-8"))


def _tool_output_response(info, payload: dict, *, call_id: str) -> ModelResponse:
    output_tool = info.output_tools[0]
    return ModelResponse(
        parts=[
            ToolCallPart(
                output_tool.name,
                payload,
                tool_call_id=call_id,
            )
        ],
        model_name="function-test",
    )


@pytest.mark.asyncio
async def test_stage2_peer_review_accepts_valid_fixture():
    payload = await run_stage2_peer_review(
        llm_client=None,
        model_id="mock-council-a",
        prompt_text="Review these analyses.",
        expected_labels=["A", "B"],
        model_override=TestModel(custom_output_args=_fixture("stage2_valid.json")),
    )

    assert payload.model_dump(mode="json") == _fixture("stage2_valid.json")


@pytest.mark.asyncio
async def test_stage2_peer_review_retries_when_expected_label_is_missing():
    attempts = {"count": 0}
    invalid = _fixture("stage2_missing_b.json")
    valid = _fixture("stage2_valid.json")

    async def function(messages, info):
        del messages
        attempts["count"] += 1
        payload = invalid if attempts["count"] == 1 else valid
        return _tool_output_response(
            info, payload, call_id=f"stage2-{attempts['count']}"
        )

    payload = await run_stage2_peer_review(
        llm_client=None,
        model_id="mock-council-a",
        prompt_text="Review these analyses.",
        expected_labels=["A", "B"],
        model_override=FunctionModel(function=function),
    )

    assert attempts["count"] == 2
    assert payload.ranking_best_to_worst == ["A", "B"]


@pytest.mark.asyncio
async def test_stage5_final_review_accepts_valid_fixture():
    payload = await run_stage5_final_review(
        llm_client=None,
        model_id="mock-council-a",
        prompt_text="Review this consolidated report.",
        model_override=TestModel(
            custom_output_args=_fixture("stage5_approve_valid.json")
        ),
    )

    assert payload.model_dump(mode="json") == _fixture("stage5_approve_valid.json")


@pytest.mark.asyncio
async def test_stage5_final_review_allows_revise_with_empty_required_changes():
    payload = await run_stage5_final_review(
        llm_client=None,
        model_id="mock-council-a",
        prompt_text="Review this consolidated report.",
        model_override=TestModel(
            custom_output_args={
                "vote": "REVISE",
                "required_changes": [],
                "optional_changes": ["Clarify the summary wording."],
                "quality_score_1to10": 6,
            }
        ),
    )

    assert payload.vote == "REVISE"
    assert payload.required_changes == []


@pytest.mark.asyncio
async def test_stage5_final_review_retries_when_vote_conflicts_with_required_changes():
    attempts = {"count": 0}
    invalid = _fixture("stage5_approve_with_required_changes.json")
    valid = _fixture("stage5_revise_valid.json")

    async def function(messages, info):
        del messages
        attempts["count"] += 1
        payload = invalid if attempts["count"] == 1 else valid
        return _tool_output_response(
            info, payload, call_id=f"stage5-{attempts['count']}"
        )

    payload = await run_stage5_final_review(
        llm_client=None,
        model_id="mock-council-a",
        prompt_text="Review this consolidated report.",
        model_override=FunctionModel(function=function),
    )

    assert attempts["count"] == 2
    assert payload.vote == "REVISE"
    assert payload.required_changes == valid["required_changes"]


def test_cliproxy_openai_base_url_appends_v1_once():
    assert (
        _cliproxy_openai_base_url("http://127.0.0.1:8317") == "http://127.0.0.1:8317/v1"
    )
    assert (
        _cliproxy_openai_base_url("http://127.0.0.1:8317/v1")
        == "http://127.0.0.1:8317/v1"
    )

@pytest.mark.asyncio
@pytest.mark.parametrize('model_id,expected_limit', [('z-ai/glm-5.3-flash',8192),('openai/gpt-6.1-sol',2500),('mock-council-a',2500)])
async def test_stage5_keeps_room_for_glm_reasoning_and_valid_final_vote(model_id,expected_limit):
    observed=[]
    async def function(messages,info):
        observed.append(info.model_settings['max_tokens'])
        # Model refuses to provide a usable tool result if reasoning consumes
        # the entire allowance; the live GLM receipt exhausted 2500 tokens.
        assert info.model_settings['max_tokens']==expected_limit
        return _tool_output_response(info,_fixture('stage5_approve_valid.json'),call_id='bounded-stage5')
    payload=await run_stage5_final_review(llm_client=None,model_id=model_id,prompt_text='Synthetic final review.',model_override=FunctionModel(function=function))
    assert payload.vote=='APPROVE'
    assert observed==[expected_limit]
