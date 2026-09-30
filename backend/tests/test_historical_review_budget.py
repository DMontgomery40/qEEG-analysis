"""REG-21/26/27: final-review adapter budgets with provider-free transport seams.

These prove the shipping adapter's bounded request choices, not that a real
provider returns a usable review within that bound or that publication is ready.
"""
from contextlib import asynccontextmanager
import json
from pathlib import Path

import pytest
from pydantic_ai.exceptions import ModelHTTPError

from backend.council import ai_review_agents as review


@pytest.mark.asyncio
@pytest.mark.parametrize('model_id,limit', [
    ('z-ai/glm-5.3-flash', 8192), ('openai/gpt-6.1-sol', 2500), ('mock-council-a', 2500),
])
@pytest.mark.parametrize('responses_fallback', [False, True])
async def test_REG_21_final_review_budget_survives_chat_and_responses_fallback(monkeypatch, model_id, limit, responses_fallback):
    calls = []

    @asynccontextmanager
    async def provider(_):
        yield object()

    async def agent_call(**kwargs):
        kind = kwargs['model']
        calls.append((kind, dict(kwargs['model_settings']), kwargs['prompt_text']))
        if kind == 'chat' and responses_fallback:
            raise ModelHTTPError(400, model_id, {'error': 'Chat completions not supported; use Responses'})
        data = json.loads((Path(__file__).parent / 'fixtures/ai_review/stage5_approve_valid.json').read_text())
        return review.Stage5FinalReviewPayload.model_validate(data)

    monkeypatch.setattr(review, '_provider_for_call', provider)
    monkeypatch.setattr(review, 'OpenAIChatModel', lambda *args, **kwargs: 'chat')
    monkeypatch.setattr(review, 'OpenAIResponsesModel', lambda *args, **kwargs: 'responses')
    monkeypatch.setattr(review._ReviewRequests, 'wrap', lambda self, model: model)
    monkeypatch.setattr(review, '_run_agent_with_backoff', agent_call)
    result = await review.run_stage5_final_review(llm_client=None, model_id=model_id,
                                                prompt_text='Synthetic full analysis; retain all sessions.')
    assert result.vote == 'APPROVE'
    assert [c[0] for c in calls] == (['chat', 'responses'] if responses_fallback else ['chat'])
    assert all(c[1]['max_tokens'] == limit for c in calls)
    assert all(c[1]['temperature'] == 0.1 for c in calls)
    assert all(c[2] == 'Synthetic full analysis; retain all sessions.' for c in calls)


@pytest.mark.asyncio
async def test_REG_27_busy_final_review_does_not_invent_unsupported_transport_fallback(monkeypatch):
    calls = []

    @asynccontextmanager
    async def provider(_):
        yield object()

    async def agent_call(**kwargs):
        calls.append(kwargs['model'])
        raise ModelHTTPError(503, 'z-ai/glm-5.3-flash', {'error': 'Temporarily busy'})

    monkeypatch.setattr(review, '_provider_for_call', provider)
    monkeypatch.setattr(review, 'OpenAIChatModel', lambda *args, **kwargs: 'chat')
    monkeypatch.setattr(review, 'OpenAIResponsesModel', lambda *args, **kwargs: 'responses')
    monkeypatch.setattr(review._ReviewRequests, 'wrap', lambda self, model: model)
    monkeypatch.setattr(review, '_run_agent_with_backoff', agent_call)
    with pytest.raises(ModelHTTPError) as error:
        await review.run_stage5_final_review(llm_client=None, model_id='z-ai/glm-5.3-flash',
                                            prompt_text='Synthetic full analysis.')
    assert error.value.status_code == 503
    assert calls == ['chat']
