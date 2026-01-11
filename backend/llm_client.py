from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import httpx


class UpstreamError(RuntimeError):
    def __init__(self, message: str, *, status_code: int | None = None):
        super().__init__(message)
        self.status_code = status_code


@dataclass(frozen=True)
class _OpenAICompatError:
    message: str
    type: str | None = None
    code: str | None = None


def _parse_openai_error(payload: Any) -> _OpenAICompatError | None:
    if not isinstance(payload, dict):
        return None
    err = payload.get("error")
    if not isinstance(err, dict):
        return None
    message = err.get("message")
    if not isinstance(message, str) or not message.strip():
        return None
    return _OpenAICompatError(
        message=message.strip(),
        type=err.get("type") if isinstance(err.get("type"), str) else None,
        code=err.get("code") if isinstance(err.get("code"), str) else None,
    )


def _format_http_error(
    response: httpx.Response, *, prefix: str, fallback_message: str
) -> UpstreamError:
    try:
        payload = response.json()
    except Exception:
        payload = None

    parsed = _parse_openai_error(payload)
    if parsed is not None:
        msg = f"{prefix}: {parsed.message}"
        return UpstreamError(msg, status_code=response.status_code)

    body_preview: str | None = None
    try:
        body_preview = response.text
        if len(body_preview) > 5000:
            body_preview = body_preview[:5000] + "…"
    except Exception:
        body_preview = None

    msg = f"{prefix}: {fallback_message}"
    if body_preview:
        msg = f"{msg}\n\nUpstream response body:\n{body_preview}"
    return UpstreamError(msg, status_code=response.status_code)


def _chat_unsupported(err: UpstreamError) -> bool:
    if err.status_code is None:
        return False
    if err.status_code not in {400, 404, 405}:
        return False
    text = str(err).lower()
    return (
        "responses" in text
        or "response endpoint" in text
        or "not support chat" in text
        or "chat completions" in text and "not supported" in text
    )


class AsyncOpenAICompatClient:
    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        timeout_s: float = 120.0,
        transport: httpx.AsyncBaseTransport | None = None,
    ):
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key.strip()
        self._timeout_s = timeout_s
        self._transport = transport
        self._client: httpx.AsyncClient | None = None

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            headers: dict[str, str] = {"Content-Type": "application/json"}
            if self._api_key:
                headers["Authorization"] = f"Bearer {self._api_key}"
            self._client = httpx.AsyncClient(
                base_url=self._base_url,
                headers=headers,
                timeout=httpx.Timeout(self._timeout_s),
                transport=self._transport,
            )
        return self._client

    async def list_models(self) -> list[str]:
        client = self._get_client()
        try:
            resp = await client.get("/v1/models")
        except Exception as e:
            raise UpstreamError(f"CLIProxyAPI request failed: {e}") from e

        if resp.status_code >= 400:
            raise _format_http_error(
                resp,
                prefix="CLIProxyAPI /v1/models failed",
                fallback_message=f"HTTP {resp.status_code}",
            )

        try:
            payload = resp.json()
        except Exception as e:
            raise UpstreamError(f"CLIProxyAPI /v1/models returned invalid JSON: {e}") from e

        data = payload.get("data") if isinstance(payload, dict) else None
        if not isinstance(data, list):
            raise UpstreamError("CLIProxyAPI /v1/models returned unexpected shape")

        ids: list[str] = []
        for item in data:
            if isinstance(item, dict) and isinstance(item.get("id"), str):
                ids.append(item["id"])
        return ids

    async def chat_completions(
        self,
        *,
        model_id: str,
        messages: list[dict],
        temperature: float = 0.2,
        max_tokens: int = 1800,
        stream: bool = False,
    ) -> str:
        client = self._get_client()
        payload = {
            "model": model_id,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": stream,
        }

        try:
            resp = await client.post("/v1/chat/completions", json=payload)
        except Exception as e:
            raise UpstreamError(f"CLIProxyAPI request failed: {e}") from e

        if resp.status_code >= 400:
            err = _format_http_error(
                resp,
                prefix="CLIProxyAPI /v1/chat/completions failed",
                fallback_message=f"HTTP {resp.status_code}",
            )
            if _chat_unsupported(err):
                input_text = self._messages_to_input_text(messages)
                return await self.responses(
                    model_id=model_id, input_text=input_text, stream=stream
                )
            raise err

        try:
            data = resp.json()
        except Exception as e:
            raise UpstreamError(
                f"CLIProxyAPI /v1/chat/completions returned invalid JSON: {e}"
            ) from e

        try:
            content = data["choices"][0]["message"]["content"]
        except Exception as e:
            raise UpstreamError(
                f"CLIProxyAPI /v1/chat/completions returned unexpected shape: {e}"
            ) from e

        if not isinstance(content, str):
            raise UpstreamError("CLIProxyAPI /v1/chat/completions returned non-text content")
        return content

    async def responses(self, *, model_id: str, input_text: str, stream: bool = False) -> str:
        client = self._get_client()
        payload = {"model": model_id, "input": input_text, "stream": stream}

        try:
            resp = await client.post("/v1/responses", json=payload)
        except Exception as e:
            raise UpstreamError(f"CLIProxyAPI request failed: {e}") from e

        if resp.status_code >= 400:
            raise _format_http_error(
                resp,
                prefix="CLIProxyAPI /v1/responses failed",
                fallback_message=f"HTTP {resp.status_code}",
            )

        try:
            data = resp.json()
        except Exception as e:
            raise UpstreamError(
                f"CLIProxyAPI /v1/responses returned invalid JSON: {e}"
            ) from e

        # OpenAI Responses API: output_text may be present; otherwise reconstruct from output blocks.
        output_text = data.get("output_text") if isinstance(data, dict) else None
        if isinstance(output_text, str):
            return output_text

        if not isinstance(data, dict) or not isinstance(data.get("output"), list):
            raise UpstreamError("CLIProxyAPI /v1/responses returned unexpected shape")

        chunks: list[str] = []
        for item in data["output"]:
            if not isinstance(item, dict):
                continue
            content = item.get("content")
            if not isinstance(content, list):
                continue
            for block in content:
                if isinstance(block, dict) and block.get("type") == "output_text":
                    text = block.get("text")
                    if isinstance(text, str):
                        chunks.append(text)
        return "".join(chunks).strip()

    @staticmethod
    def _messages_to_input_text(messages: list[dict]) -> str:
        lines: list[str] = []
        for m in messages:
            role = m.get("role", "user")
            content = m.get("content", "")
            if not isinstance(role, str):
                role = "user"
            if not isinstance(content, str):
                try:
                    content = json.dumps(content)
                except Exception:
                    content = str(content)
            lines.append(f"{role.upper()}:\n{content}".strip())
        return "\n\n".join(lines).strip()
