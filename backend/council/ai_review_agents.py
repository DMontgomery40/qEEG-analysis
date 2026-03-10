from __future__ import annotations

import json
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any, AsyncIterator, Literal

import httpx
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from pydantic_ai import Agent, ModelRetry, RunContext
from pydantic_ai.exceptions import ModelAPIError, ModelHTTPError
from pydantic_ai.models.openai import OpenAIChatModel, OpenAIResponsesModel
from pydantic_ai.providers.openai import OpenAIProvider

from ..config import CLIPROXY_API_KEY, CLIPROXY_BASE_URL
from ..llm_client import (
    AsyncOpenAICompatClient,
    _is_openai_gpt5_model,
    _openai_reasoning_effort,
)
from .utils import _sleep_backoff


def _strip_text(value: Any, *, field_name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string")
    text = value.strip()
    if not text:
        raise ValueError(f"{field_name} must not be empty")
    return text


def _strip_string_list(value: Any, *, field_name: str) -> list[str]:
    if not isinstance(value, list):
        raise TypeError(f"{field_name} must be a list")
    cleaned: list[str] = []
    for item in value:
        text = _strip_text(item, field_name=field_name)
        cleaned.append(text)
    return cleaned


def _normalize_label(value: Any) -> str:
    return _strip_text(value, field_name="label").upper()


class Stage2ReviewItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    analysis_label: str
    strengths: list[str]
    weaknesses: list[str]
    missing_or_unclear: list[str]
    accuracy_issues: list[str]
    risk_of_overreach: str

    @field_validator("analysis_label", mode="before")
    @classmethod
    def _validate_analysis_label(cls, value: Any) -> str:
        return _normalize_label(value)

    @field_validator(
        "strengths",
        "weaknesses",
        "missing_or_unclear",
        "accuracy_issues",
        mode="before",
    )
    @classmethod
    def _validate_string_lists(cls, value: Any, info) -> list[str]:
        return _strip_string_list(value, field_name=info.field_name)

    @field_validator("risk_of_overreach", mode="before")
    @classmethod
    def _validate_risk_of_overreach(cls, value: Any) -> str:
        return _strip_text(value, field_name="risk_of_overreach")


class Stage2PeerReviewPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    reviews: list[Stage2ReviewItem]
    ranking_best_to_worst: list[str]
    overall_notes: str

    @field_validator("ranking_best_to_worst", mode="before")
    @classmethod
    def _validate_ranking(cls, value: Any) -> list[str]:
        if not isinstance(value, list):
            raise TypeError("ranking_best_to_worst must be a list")
        return [_normalize_label(item) for item in value]

    @field_validator("overall_notes", mode="before")
    @classmethod
    def _validate_overall_notes(cls, value: Any) -> str:
        return _strip_text(value, field_name="overall_notes")

    @model_validator(mode="after")
    def _check_unique_review_labels(self) -> "Stage2PeerReviewPayload":
        review_labels = [item.analysis_label for item in self.reviews]
        if len(set(review_labels)) != len(review_labels):
            raise ValueError("reviews must not repeat analysis_label values")
        if len(set(self.ranking_best_to_worst)) != len(self.ranking_best_to_worst):
            raise ValueError("ranking_best_to_worst must not repeat labels")
        return self


class Stage5FinalReviewPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    vote: Literal["APPROVE", "REVISE"]
    required_changes: list[str]
    optional_changes: list[str]
    quality_score_1to10: int = Field(ge=1, le=10)

    @field_validator("required_changes", "optional_changes", mode="before")
    @classmethod
    def _validate_change_lists(cls, value: Any, info) -> list[str]:
        return _strip_string_list(value, field_name=info.field_name)

    @model_validator(mode="after")
    def _check_vote_consistency(self) -> "Stage5FinalReviewPayload":
        if self.vote == "APPROVE" and self.required_changes:
            raise ValueError("APPROVE outputs must not include required_changes")
        if self.vote == "REVISE" and not self.required_changes:
            raise ValueError("REVISE outputs must include at least one required change")
        return self


@dataclass(frozen=True)
class Stage2ReviewDeps:
    expected_labels: tuple[str, ...]


_STAGE2_REVIEW_AGENT = Agent(
    output_type=Stage2PeerReviewPayload,
    deps_type=Stage2ReviewDeps,
    system_prompt=(
        "You are the qEEG Council blinded peer review agent. "
        "Follow the user prompt exactly, stay evidence-grounded, and return only structured output."
    ),
    output_retries=2,
    defer_model_check=True,
)

_STAGE5_REVIEW_AGENT = Agent(
    output_type=Stage5FinalReviewPayload,
    system_prompt=(
        "You are the qEEG Council final review agent. "
        "Follow the user prompt exactly, stay evidence-grounded, and return only structured output."
    ),
    output_retries=2,
    defer_model_check=True,
)


@_STAGE2_REVIEW_AGENT.output_validator
def _validate_stage2_expected_labels(
    ctx: RunContext[Stage2ReviewDeps], output: Stage2PeerReviewPayload
) -> Stage2PeerReviewPayload:
    expected = tuple(_normalize_label(label) for label in ctx.deps.expected_labels)
    expected_set = set(expected)
    review_labels = [item.analysis_label for item in output.reviews]
    ranking_labels = output.ranking_best_to_worst

    if len(review_labels) != len(expected) or set(review_labels) != expected_set:
        expected_text = ", ".join(expected)
        raise ModelRetry(
            "Return exactly one review item for each expected analysis label and no extras. "
            f"Expected labels: {expected_text}."
        )
    if len(ranking_labels) != len(expected) or set(ranking_labels) != expected_set:
        expected_text = ", ".join(expected)
        raise ModelRetry(
            "ranking_best_to_worst must contain every expected analysis label exactly once. "
            f"Expected labels: {expected_text}."
        )

    return output


def _cliproxy_openai_base_url(base_url: str | None) -> str:
    root = (base_url or CLIPROXY_BASE_URL).strip().rstrip("/")
    if root.endswith("/v1"):
        return root
    return f"{root}/v1"


def _coerce_reasoning_effort(model_id: str) -> Literal["low", "medium", "high"] | None:
    effort = _openai_reasoning_effort(model_id)
    if effort is None:
        return None
    return {
        "minimal": "low",
        "low": "low",
        "medium": "medium",
        "high": "high",
        "xhigh": "high",
    }.get(effort)


def _model_settings(
    model_id: str, *, temperature: float, max_tokens: int
) -> dict[str, Any]:
    settings: dict[str, Any] = {
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    effort = _coerce_reasoning_effort(model_id)
    if effort is not None:
        settings["openai_reasoning_effort"] = effort
    return settings


def _chat_unsupported_http_error(err: ModelHTTPError) -> bool:
    if err.status_code not in {400, 404, 405}:
        return False
    text = str(err).lower()
    return (
        "responses" in text
        or "response endpoint" in text
        or "not support chat" in text
        or ("chat completions" in text and "not supported" in text)
    )


@asynccontextmanager
async def _provider_for_call(
    llm_client: AsyncOpenAICompatClient | None,
) -> AsyncIterator[OpenAIProvider]:
    http_client: httpx.AsyncClient | None = None

    if llm_client is not None:
        transport = getattr(llm_client, "_transport", None)
        timeout_s = getattr(llm_client, "_timeout_s", 120.0)
        base_url = _cliproxy_openai_base_url(getattr(llm_client, "_base_url", None))
        api_key = getattr(llm_client, "_api_key", "") or CLIPROXY_API_KEY
        if transport is not None:
            http_client = httpx.AsyncClient(
                timeout=httpx.Timeout(timeout_s),
                transport=transport,
            )
        provider = OpenAIProvider(
            base_url=base_url,
            api_key=api_key or None,
            http_client=http_client,
        )
    else:
        provider = OpenAIProvider(
            base_url=_cliproxy_openai_base_url(None),
            api_key=CLIPROXY_API_KEY or None,
        )

    try:
        yield provider
    finally:
        if http_client is not None:
            await http_client.aclose()


async def _run_agent_with_backoff(
    *,
    agent: Agent[Any, Any],
    prompt_text: str,
    model: Any,
    deps: Any = None,
    model_settings: dict[str, Any] | None = None,
) -> Any:
    attempts = 0
    while True:
        try:
            result = await agent.run(
                prompt_text,
                deps=deps,
                model=model,
                model_settings=model_settings,
            )
            return result.output
        except ModelHTTPError as err:
            if err.status_code in {429, 500, 502, 503, 504} and attempts < 4:
                await _sleep_backoff(attempts)
                attempts += 1
                continue
            raise
        except ModelAPIError:
            if attempts < 4:
                await _sleep_backoff(attempts)
                attempts += 1
                continue
            raise


async def run_stage2_peer_review(
    *,
    llm_client: AsyncOpenAICompatClient | None,
    model_id: str,
    prompt_text: str,
    expected_labels: list[str],
    model_override: Any = None,
) -> Stage2PeerReviewPayload:
    deps = Stage2ReviewDeps(expected_labels=tuple(expected_labels))
    if model_override is not None:
        return await _run_agent_with_backoff(
            agent=_STAGE2_REVIEW_AGENT,
            prompt_text=prompt_text,
            model=model_override,
            deps=deps,
            model_settings=_model_settings(model_id, temperature=0.1, max_tokens=4000),
        )

    async with _provider_for_call(llm_client) as provider:
        chat_model = OpenAIChatModel(model_id, provider=provider)
        try:
            return await _run_agent_with_backoff(
                agent=_STAGE2_REVIEW_AGENT,
                prompt_text=prompt_text,
                model=chat_model,
                deps=deps,
                model_settings=_model_settings(
                    model_id, temperature=0.1, max_tokens=4000
                ),
            )
        except ModelHTTPError as err:
            if not _chat_unsupported_http_error(err):
                raise
            responses_model = OpenAIResponsesModel(model_id, provider=provider)
            return await _run_agent_with_backoff(
                agent=_STAGE2_REVIEW_AGENT,
                prompt_text=prompt_text,
                model=responses_model,
                deps=deps,
                model_settings=_model_settings(
                    model_id, temperature=0.1, max_tokens=4000
                ),
            )


async def run_stage2_peer_review_json(
    *,
    llm_client: AsyncOpenAICompatClient | None,
    model_id: str,
    prompt_text: str,
    expected_labels: list[str],
    model_override: Any = None,
) -> str:
    payload = await run_stage2_peer_review(
        llm_client=llm_client,
        model_id=model_id,
        prompt_text=prompt_text,
        expected_labels=expected_labels,
        model_override=model_override,
    )
    return json.dumps(payload.model_dump(mode="json"), indent=2, sort_keys=True)


async def run_stage5_final_review(
    *,
    llm_client: AsyncOpenAICompatClient | None,
    model_id: str,
    prompt_text: str,
    model_override: Any = None,
) -> Stage5FinalReviewPayload:
    if model_override is not None:
        return await _run_agent_with_backoff(
            agent=_STAGE5_REVIEW_AGENT,
            prompt_text=prompt_text,
            model=model_override,
            model_settings=_model_settings(model_id, temperature=0.1, max_tokens=2500),
        )

    async with _provider_for_call(llm_client) as provider:
        chat_model = OpenAIChatModel(model_id, provider=provider)
        try:
            return await _run_agent_with_backoff(
                agent=_STAGE5_REVIEW_AGENT,
                prompt_text=prompt_text,
                model=chat_model,
                model_settings=_model_settings(
                    model_id, temperature=0.1, max_tokens=2500
                ),
            )
        except ModelHTTPError as err:
            if not (
                _is_openai_gpt5_model(model_id) or _chat_unsupported_http_error(err)
            ):
                raise
            responses_model = OpenAIResponsesModel(model_id, provider=provider)
            return await _run_agent_with_backoff(
                agent=_STAGE5_REVIEW_AGENT,
                prompt_text=prompt_text,
                model=responses_model,
                model_settings=_model_settings(
                    model_id, temperature=0.1, max_tokens=2500
                ),
            )


async def run_stage5_final_review_json(
    *,
    llm_client: AsyncOpenAICompatClient | None,
    model_id: str,
    prompt_text: str,
    model_override: Any = None,
) -> str:
    payload = await run_stage5_final_review(
        llm_client=llm_client,
        model_id=model_id,
        prompt_text=prompt_text,
        model_override=model_override,
    )
    return json.dumps(payload.model_dump(mode="json"), indent=2, sort_keys=True)
