import json

import httpx
import pytest

from backend.llm_client import AsyncOpenAICompatClient


@pytest.mark.asyncio
async def test_list_models_parses_openai_shape():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/models"
        return httpx.Response(200, json={"data": [{"id": "a"}, {"id": "b"}]})

    transport = httpx.MockTransport(handler)
    client = AsyncOpenAICompatClient(
        base_url="http://test", api_key="", timeout_s=5.0, transport=transport
    )
    try:
        ids = await client.list_models()
    finally:
        await client.aclose()
    assert ids == ["a", "b"]


@pytest.mark.asyncio
async def test_chat_completions_falls_back_to_responses():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/chat/completions":
            return httpx.Response(
                400,
                json={
                    "error": {
                        "message": "chat completions not supported; use /v1/responses"
                    }
                },
            )
        if request.url.path == "/v1/responses":
            return httpx.Response(200, json={"output_text": "ok"})
        raise AssertionError(f"Unexpected request path: {request.url.path}")

    transport = httpx.MockTransport(handler)
    client = AsyncOpenAICompatClient(
        base_url="http://test", api_key="", timeout_s=5.0, transport=transport
    )
    try:
        out = await client.chat_completions(
            model_id="m",
            messages=[{"role": "user", "content": "hi"}],
            temperature=0.2,
            max_tokens=20,
            stream=False,
        )
    finally:
        await client.aclose()
    assert out == "ok"


@pytest.mark.asyncio
async def test_chat_completions_prefers_responses_for_gpt5_and_sets_max_output_tokens():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/responses":
            body = json.loads(request.content)
            assert body.get("model") == "gpt-5.2"
            assert body.get("max_output_tokens") == 20
            assert body.get("reasoning") == {"effort": "medium"}
            assert body.get("input") == [
                {"role": "user", "content": [{"type": "input_text", "text": "hi"}]}
            ]
            return httpx.Response(200, json={"output_text": "ok"})
        raise AssertionError(f"Unexpected request path: {request.url.path}")

    transport = httpx.MockTransport(handler)
    client = AsyncOpenAICompatClient(
        base_url="http://test", api_key="", timeout_s=5.0, transport=transport
    )
    try:
        out = await client.chat_completions(
            model_id="gpt-5.2",
            messages=[{"role": "user", "content": "hi"}],
            temperature=0.2,
            max_tokens=20,
            stream=False,
        )
    finally:
        await client.aclose()
    assert out == "ok"


@pytest.mark.asyncio
async def test_chat_completions_maps_xhigh_reasoning_for_responses():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/responses":
            body = json.loads(request.content)
            assert body.get("model") == "gpt-5.3-codex-xhigh"
            assert body.get("reasoning") == {"effort": "xhigh"}
            assert body.get("max_output_tokens") == 20
            return httpx.Response(200, json={"output_text": "ok"})
        raise AssertionError(f"Unexpected request path: {request.url.path}")

    transport = httpx.MockTransport(handler)
    client = AsyncOpenAICompatClient(
        base_url="http://test", api_key="", timeout_s=5.0, transport=transport
    )
    try:
        out = await client.chat_completions(
            model_id="gpt-5.3-codex-xhigh",
            messages=[{"role": "user", "content": "hi"}],
            temperature=0.2,
            max_tokens=20,
            stream=False,
        )
    finally:
        await client.aclose()
    assert out == "ok"


@pytest.mark.asyncio
async def test_chat_completions_honors_reasoning_override_env(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("QEEG_OPENAI_REASONING_EFFORT", "xhigh")

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/responses":
            body = json.loads(request.content)
            assert body.get("model") == "gpt-5.4"
            assert body.get("reasoning") == {"effort": "xhigh"}
            return httpx.Response(200, json={"output_text": "ok"})
        raise AssertionError(f"Unexpected request path: {request.url.path}")

    transport = httpx.MockTransport(handler)
    client = AsyncOpenAICompatClient(
        base_url="http://test", api_key="", timeout_s=5.0, transport=transport
    )
    try:
        out = await client.chat_completions(
            model_id="gpt-5.4",
            messages=[{"role": "user", "content": "hi"}],
            temperature=0.2,
            max_tokens=20,
            stream=False,
        )
    finally:
        await client.aclose()
    assert out == "ok"


def test_messages_to_responses_input_preserves_multimodal_blocks():
    converted = AsyncOpenAICompatClient._messages_to_responses_input(
        [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "look at this"},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "data:image/png;base64,abc",
                            "detail": "high",
                        },
                    },
                ],
            }
        ]
    )

    assert converted == [
        {
            "role": "user",
            "content": [
                {"type": "input_text", "text": "look at this"},
                {"type": "input_image", "image_url": "data:image/png;base64,abc"},
            ],
        }
    ]
