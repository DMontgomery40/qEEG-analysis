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
async def test_list_models_appends_explicit_openrouter_extras(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("QEEG_OPENROUTER_EXTRA_MODELS", "z-ai/glm-5.1, a")
    monkeypatch.setenv("QEEG_ROUTE_OPENROUTER_EXTRAS_DIRECT", "1")
    monkeypatch.setenv("OPENROUTER_API_KEY", "or-key")

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
    assert ids == ["a", "b", "z-ai/glm-5.1"]


@pytest.mark.asyncio
async def test_list_models_omits_openrouter_extras_when_direct_provider_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("QEEG_OPENROUTER_EXTRA_MODELS", "z-ai/glm-5.2")
    monkeypatch.delenv("QEEG_ROUTE_OPENROUTER_EXTRAS_DIRECT", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/models"
        return httpx.Response(200, json={"data": [{"id": "a"}, {"id": "b"}]})

    client = AsyncOpenAICompatClient(
        base_url="http://test",
        api_key="",
        timeout_s=5.0,
        transport=httpx.MockTransport(handler),
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
async def test_chat_completions_routes_explicit_openrouter_extra(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("QEEG_OPENROUTER_EXTRA_MODELS", "z-ai/glm-5.1")
    monkeypatch.setenv("QEEG_ROUTE_OPENROUTER_EXTRAS_DIRECT", "1")
    monkeypatch.setenv("OPENROUTER_API_KEY", "or-key")
    monkeypatch.setenv("OPENROUTER_BASE_URL", "https://openrouter.test/api")

    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        assert request.url.host == "openrouter.test"
        assert request.url.path == "/api/v1/chat/completions"
        assert request.headers["Authorization"] == "Bearer or-key"
        body = json.loads(request.content)
        assert body["model"] == "z-ai/glm-5.1"
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": "ok"}}]},
        )

    transport = httpx.MockTransport(handler)
    client = AsyncOpenAICompatClient(
        base_url="http://cliproxy.test", api_key="", timeout_s=5.0, transport=transport
    )
    try:
        out = await client.chat_completions(
            model_id="z-ai/glm-5.1",
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=20,
        )
    finally:
        await client.aclose()

    assert out == "ok"
    assert seen


@pytest.mark.asyncio
async def test_chat_completions_routes_openrouter_extra_through_cliproxy_by_default(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("QEEG_OPENROUTER_EXTRA_MODELS", "z-ai/glm-5.2")
    monkeypatch.delenv("QEEG_ROUTE_OPENROUTER_EXTRAS_DIRECT", raising=False)
    monkeypatch.setenv("OPENROUTER_API_KEY", "or-key")
    monkeypatch.setenv("OPENROUTER_BASE_URL", "https://openrouter.test/api")

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.host == "cliproxy.test"
        assert request.url.path == "/v1/chat/completions"
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": "writer output"}}]},
        )

    client = AsyncOpenAICompatClient(
        base_url="http://cliproxy.test",
        api_key="",
        timeout_s=5.0,
        transport=httpx.MockTransport(handler),
    )
    try:
        out = await client.chat_completions(
            model_id="z-ai/glm-5.2",
            messages=[{"role": "user", "content": "Write the report."}],
            max_tokens=100,
        )
    finally:
        await client.aclose()

    assert out == "writer output"


@pytest.mark.asyncio
@pytest.mark.parametrize("model_id", ["z-ai/glm-5.2", "z-ai/glm-5.3-flash"])
async def test_glm_writer_uses_high_reasoning_and_returns_only_final_content(
    monkeypatch: pytest.MonkeyPatch,
    model_id: str,
):
    monkeypatch.delenv("QEEG_OPENROUTER_EXTRA_MODELS", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    requests: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.host == "cliproxy.test"
        assert request.url.path == "/v1/chat/completions"
        body = json.loads(request.content)
        requests.append(body)
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": "# Your Brain Assessment Summary\nFinal report.",
                            "reasoning": "private chain of thought",
                        }
                    }
                ]
            },
        )

    client = AsyncOpenAICompatClient(
        base_url="http://cliproxy.test",
        api_key="",
        timeout_s=5.0,
        transport=httpx.MockTransport(handler),
    )
    try:
        output = await client.chat_completions(
            model_id=model_id,
            messages=[{"role": "user", "content": "Write the report."}],
            max_tokens=100,
        )
    finally:
        await client.aclose()

    assert output == "# Your Brain Assessment Summary\nFinal report."
    assert "private chain of thought" not in output
    assert requests[0]["reasoning"] == {"effort": "high", "exclude": True}


@pytest.mark.asyncio
@pytest.mark.parametrize("model_id", ["z-ai/glm-5.2", "z-ai/glm-5.3-flash"])
async def test_glm_empty_content_retries_once_without_reasoning(
    monkeypatch: pytest.MonkeyPatch,
    model_id: str,
):
    monkeypatch.delenv("QEEG_OPENROUTER_EXTRA_MODELS", raising=False)
    monkeypatch.setenv("OPENROUTER_API_KEY", "or-key")
    requests: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        requests.append(body)
        if len(requests) == 1:
            return httpx.Response(
                200,
                json={
                    "choices": [
                        {
                            "message": {
                                "content": "",
                                "reasoning": "The unpublished draft reasoning.",
                            }
                        }
                    ]
                },
            )
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": "# Your Brain Assessment Summary\nRecovered final."
                        }
                    }
                ]
            },
        )

    client = AsyncOpenAICompatClient(
        base_url="http://cliproxy.test",
        api_key="",
        timeout_s=5.0,
        transport=httpx.MockTransport(handler),
    )
    try:
        output = await client.chat_completions(
            model_id=model_id,
            messages=[{"role": "user", "content": "Write the report."}],
            max_tokens=100,
        )
    finally:
        await client.aclose()

    assert output.endswith("Recovered final.")
    assert len(requests) == 2
    assert requests[0]["reasoning"] == {"effort": "high", "exclude": True}
    assert requests[1]["reasoning"] == {"effort": "none", "exclude": True}
    assert "final draft only" in requests[1]["messages"][-1]["content"].lower()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "empty_message",
    [
        {"content": None, "reasoning": "Unpublished reasoning."},
        {"reasoning": "Unpublished reasoning."},
    ],
)
@pytest.mark.parametrize("model_id", ["z-ai/glm-5.2", "z-ai/glm-5.3-flash"])
async def test_glm_null_or_missing_content_retries_once(
    monkeypatch: pytest.MonkeyPatch,
    empty_message: dict,
    model_id: str,
):
    monkeypatch.delenv("QEEG_OPENROUTER_EXTRA_MODELS", raising=False)
    monkeypatch.setenv("OPENROUTER_API_KEY", "or-key")
    calls = 0

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        message = (
            empty_message
            if calls == 1
            else {"content": "# Your Brain Assessment Summary\nRecovered final."}
        )
        return httpx.Response(200, json={"choices": [{"message": message}]})

    client = AsyncOpenAICompatClient(
        base_url="http://cliproxy.test",
        api_key="",
        timeout_s=5.0,
        transport=httpx.MockTransport(handler),
    )
    try:
        output = await client.chat_completions(
            model_id=model_id,
            messages=[{"role": "user", "content": "Write the report."}],
            max_tokens=100,
        )
    finally:
        await client.aclose()

    assert output.endswith("Recovered final.")
    assert calls == 2


@pytest.mark.asyncio
async def test_glm52_rejects_reasoning_leak_after_single_retry(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.delenv("QEEG_OPENROUTER_EXTRA_MODELS", raising=False)
    monkeypatch.setenv("OPENROUTER_API_KEY", "or-key")
    calls = 0

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": "private deliberation\n# Your Brain Assessment Summary",
                            "reasoning": "private deliberation",
                        }
                    }
                ]
            },
        )

    client = AsyncOpenAICompatClient(
        base_url="http://cliproxy.test",
        api_key="",
        timeout_s=5.0,
        transport=httpx.MockTransport(handler),
    )
    try:
        with pytest.raises(UpstreamError, match="publishable final content"):
            await client.chat_completions(
                model_id="z-ai/glm-5.2",
                messages=[{"role": "user", "content": "Write the report."}],
                max_tokens=100,
            )
    finally:
        await client.aclose()

    assert calls == 2


@pytest.mark.asyncio
async def test_chat_completions_reconstructs_text_block_content():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/chat/completions"
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": [
                                {"type": "text", "text": "hello"},
                                {"type": "text", "text": " world"},
                            ]
                        }
                    }
                ]
            },
        )

    transport = httpx.MockTransport(handler)
    client = AsyncOpenAICompatClient(
        base_url="http://test", api_key="", timeout_s=5.0, transport=transport
    )
    try:
        out = await client.chat_completions(
            model_id="block-content-model",
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=20,
        )
    finally:
        await client.aclose()

    assert out == "hello world"


@pytest.mark.asyncio
async def test_chat_completions_adds_openrouter_reasoning_override(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("QEEG_OPENROUTER_EXTRA_MODELS", "z-ai/glm-5.1")
    monkeypatch.setenv("QEEG_ROUTE_OPENROUTER_EXTRAS_DIRECT", "1")
    monkeypatch.setenv("OPENROUTER_API_KEY", "or-key")
    monkeypatch.setenv("QEEG_OPENROUTER_REASONING_EFFORT_Z_AI_GLM_5_1", "none")
    monkeypatch.setenv("QEEG_OPENROUTER_REASONING_EXCLUDE_Z_AI_GLM_5_1", "1")
    monkeypatch.setenv("QEEG_OPENAI_REASONING_EFFORT", "high")

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.host == "openrouter.ai"
        body = json.loads(request.content)
        assert body["reasoning"] == {"effort": "none", "exclude": True}
        assert "reasoning_effort" not in body
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": "ok"}}]},
        )

    transport = httpx.MockTransport(handler)
    client = AsyncOpenAICompatClient(
        base_url="http://cliproxy.test", api_key="", timeout_s=5.0, transport=transport
    )
    try:
        out = await client.chat_completions(
            model_id="z-ai/glm-5.1",
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=20,
        )
    finally:
        await client.aclose()

    assert out == "ok"


@pytest.mark.asyncio
async def test_chat_completions_adds_env_gated_non_openai_reasoning(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("QEEG_REASONING_MODEL_IDS", "deepseek-v4-pro")
    monkeypatch.setenv("QEEG_REASONING_EFFORT", "high")
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
            model_id="deepseek-v4-pro",
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=20,
        )
    finally:
        await client.aclose()

    assert out == "ok"
    assert seen_payloads[0]["reasoning"] == {"effort": "high"}


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
async def test_chat_completions_defaults_gpt55_to_low_reasoning(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.delenv("QEEG_OPENAI_REASONING_EFFORT", raising=False)
    monkeypatch.delenv("OPENAI_REASONING_EFFORT", raising=False)

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/responses":
            body = json.loads(request.content)
            assert body.get("model") == "gpt-5.5"
            assert body.get("reasoning") == {"effort": "low"}
            return httpx.Response(200, json={"output_text": "ok"})
        raise AssertionError(f"Unexpected request path: {request.url.path}")

    transport = httpx.MockTransport(handler)
    client = AsyncOpenAICompatClient(
        base_url="http://test", api_key="", timeout_s=5.0, transport=transport
    )
    try:
        out = await client.chat_completions(
            model_id="gpt-5.5",
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
                input_data=[
                    {"role": "user", "content": [{"type": "input_text", "text": "hi"}]}
                ],
                stream=False,
                max_output_tokens=20,
            )
    finally:
        await client.aclose()

    assert "unexpected shape" in str(exc_info.value)
    assert exc_info.value.operator_hint is not None
    assert "/v1/responses" in exc_info.value.operator_hint


@pytest.mark.asyncio
@pytest.mark.parametrize("direct", [False, True])
@pytest.mark.parametrize("shape", ["text", "blocks"])
async def test_gpt5_extra_responses_keeps_selected_upstream(monkeypatch, direct, shape):
    monkeypatch.setenv("QEEG_OPENROUTER_EXTRA_MODELS", "openai/gpt-5.6-terra")
    monkeypatch.setenv("QEEG_ROUTE_OPENROUTER_EXTRAS_DIRECT", "1" if direct else "0")
    monkeypatch.setenv("OPENROUTER_API_KEY", "synthetic")
    monkeypatch.setenv("OPENROUTER_BASE_URL", "https://openrouter.test/api")
    seen = []
    usage = []

    def handler(request):
        seen.append(request)
        return httpx.Response(
            200,
            json={
                **(
                    {"output_text": "ok"}
                    if shape == "text"
                    else {
                        "output": [{"content": [{"type": "output_text", "text": "ok"}]}]
                    }
                ),
                "usage": {"input_tokens": 2, "output_tokens": 3},
            },
        )

    client = AsyncOpenAICompatClient(
        base_url="http://cliproxy.test",
        api_key="",
        timeout_s=5,
        transport=httpx.MockTransport(handler),
    )
    try:
        assert (
            await client.chat_completions(
                model_id="openai/gpt-5.6-terra",
                messages=[{"role": "user", "content": "hi"}],
                max_tokens=37,
                usage_callback=usage.append,
            )
            == "ok"
        )
    finally:
        await client.aclose()
    assert seen[0].url.host == ("openrouter.test" if direct else "cliproxy.test")
    assert seen[0].url.path == ("/api/v1/responses" if direct else "/v1/responses")
    assert json.loads(seen[0].content)["max_output_tokens"] == 37
    assert usage[-1]["provider"] == ("openrouter" if direct else "cliproxy")


@pytest.mark.asyncio
@pytest.mark.parametrize("direct", [False, True])
async def test_glm_retry_usage_matches_actual_route(monkeypatch, direct):
    monkeypatch.setenv("QEEG_OPENROUTER_EXTRA_MODELS", "z-ai/glm-5.2")
    monkeypatch.setenv("QEEG_ROUTE_OPENROUTER_EXTRAS_DIRECT", "1" if direct else "0")
    monkeypatch.setenv("OPENROUTER_API_KEY", "synthetic")
    usage = []
    seen = []

    def handler(request):
        seen.append(request)
        return httpx.Response(
            200,
            json={
                "choices": [
                    {"message": {"content": "" if len(seen) == 1 else "final report"}}
                ],
                "usage": {"prompt_tokens": 1, "completion_tokens": 2},
            },
        )

    client = AsyncOpenAICompatClient(
        base_url="http://cliproxy.test",
        api_key="",
        timeout_s=5,
        transport=httpx.MockTransport(handler),
    )
    try:
        assert (
            await client.chat_completions(
                model_id="z-ai/glm-5.2", messages=[], usage_callback=usage.append
            )
            == "final report"
        )
    finally:
        await client.aclose()
    assert len(usage) == 2
    assert {row["provider"] for row in usage} == {
        "openrouter" if direct else "cliproxy"
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("direct", [False, True])
@pytest.mark.parametrize("mode", ["responses", "glm-retry"])
@pytest.mark.parametrize("failure", ["http", "transport", "shape"])
async def test_selected_route_failure_attribution(monkeypatch, direct, mode, failure):
    model = "openai/gpt-5.6-terra" if mode == "responses" else "z-ai/glm-5.2"
    monkeypatch.setenv("QEEG_OPENROUTER_EXTRA_MODELS", model)
    monkeypatch.setenv("QEEG_ROUTE_OPENROUTER_EXTRAS_DIRECT", "1" if direct else "0")
    monkeypatch.setenv("OPENROUTER_API_KEY", "synthetic")
    calls = 0

    def handler(request):
        nonlocal calls
        calls += 1
        if mode == "glm-retry" and calls == 1:
            return httpx.Response(200, json={"choices": [{"message": {"content": ""}}]})
        if failure == "http":
            return httpx.Response(503, json={"error": {"message": "unavailable"}})
        if failure == "transport":
            raise httpx.ConnectError("offline", request=request)
        return httpx.Response(200, json={"wrong": "shape"})

    client = AsyncOpenAICompatClient(
        base_url="http://cliproxy.test",
        api_key="",
        timeout_s=5,
        transport=httpx.MockTransport(handler),
    )
    try:
        with pytest.raises(UpstreamError) as exc:
            await client.chat_completions(model_id=model, messages=[])
    finally:
        await client.aclose()
    expected = "OpenRouter" if direct else "CLIProxyAPI"
    wrong = "CLIProxy" if direct else "OpenRouter"
    assert expected in str(exc.value)
    assert wrong not in str(exc.value)
    assert wrong not in exc.value.operator_hint
