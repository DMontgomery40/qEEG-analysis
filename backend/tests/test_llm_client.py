import json

import httpx
import pytest

from backend.llm_client import AsyncOpenAICompatClient, UpstreamError


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
async def test_chat_completions_omits_temperature_for_claude_models():
    seen_payloads: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/chat/completions"
        body = json.loads(request.content)
        seen_payloads.append(body)
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": "ok"}}]},
        )

    transport = httpx.MockTransport(handler)
    client = AsyncOpenAICompatClient(
        base_url="http://test", api_key="", timeout_s=5.0, transport=transport
    )
    try:
        out = await client.chat_completions(
            model_id="claude-sonnet-4-6",
            messages=[{"role": "user", "content": "hi"}],
            temperature=0.2,
            max_tokens=20,
            stream=False,
        )
    finally:
        await client.aclose()

    assert out == "ok"
    assert seen_payloads
    assert "temperature" not in seen_payloads[0]


@pytest.mark.asyncio
async def test_chat_completions_rejects_empty_text_content():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/chat/completions"
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": ""}}]},
        )

    transport = httpx.MockTransport(handler)
    client = AsyncOpenAICompatClient(
        base_url="http://test", api_key="", timeout_s=5.0, transport=transport
    )
    try:
        with pytest.raises(UpstreamError, match="empty text content"):
            await client.chat_completions(
                model_id="claude-sonnet-4-6",
                messages=[{"role": "user", "content": "hi"}],
                max_tokens=20,
            )
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_chat_completions_prefers_responses_for_gpt5_and_sets_max_output_tokens(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.delenv("QEEG_OPENAI_REASONING_EFFORT", raising=False)
    monkeypatch.delenv("OPENAI_REASONING_EFFORT", raising=False)

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


@pytest.mark.asyncio
async def test_responses_reconstructs_output_blocks_when_output_text_is_empty():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/responses"
        return httpx.Response(
            200,
            json={
                "output_text": "",
                "output": [
                    {
                        "content": [
                            {"type": "output_text", "text": "recovered"},
                        ]
                    }
                ],
            },
        )

    transport = httpx.MockTransport(handler)
    client = AsyncOpenAICompatClient(
        base_url="http://test", api_key="", timeout_s=5.0, transport=transport
    )
    try:
        out = await client.responses(model_id="gpt-5.4", input_data="hi")
    finally:
        await client.aclose()

    assert out == "recovered"


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


@pytest.mark.asyncio
async def test_list_models_request_failure_sets_operator_hint():
    def handler(_request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    transport = httpx.MockTransport(handler)
    client = AsyncOpenAICompatClient(
        base_url="http://test", api_key="", timeout_s=5.0, transport=transport
    )
    try:
        with pytest.raises(UpstreamError) as exc_info:
            await client.list_models()
    finally:
        await client.aclose()

    assert "CLIProxyAPI request failed" in str(exc_info.value)
    assert exc_info.value.operator_hint is not None
    assert "/v1/models" in exc_info.value.operator_hint


@pytest.mark.asyncio
async def test_responses_unexpected_shape_sets_operator_hint():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/responses"
        return httpx.Response(200, json={"wrong": "shape"})

    transport = httpx.MockTransport(handler)
    client = AsyncOpenAICompatClient(
        base_url="http://test", api_key="", timeout_s=5.0, transport=transport
    )
    try:
        with pytest.raises(UpstreamError) as exc_info:
            await client.responses(
                model_id="gpt-5.4",
                input_data=[{"role": "user", "content": [{"type": "input_text", "text": "hi"}]}],
                stream=False,
                max_output_tokens=20,
            )
    finally:
        await client.aclose()

    assert "unexpected shape" in str(exc_info.value)
    assert exc_info.value.operator_hint is not None
    assert "/v1/responses" in exc_info.value.operator_hint
