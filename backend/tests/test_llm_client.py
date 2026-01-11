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
                json={"error": {"message": "chat completions not supported; use /v1/responses"}},
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

