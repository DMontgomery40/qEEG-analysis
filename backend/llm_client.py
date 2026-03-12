from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any

import httpx


class UpstreamError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        operator_hint: str | None = None,
    ):
        super().__init__(message)
        self.status_code = status_code
        self.operator_hint = operator_hint


def _operator_hint(endpoint: str, issue: str) -> str:
    if endpoint == "/v1/models":
        if issue == "request_failed":
            return "AsyncOpenAICompatClient.list_models calls CLIProxyAPI /v1/models before model refresh; inspect CLIProxy reachability or auth at that endpoint."
        if issue == "invalid_json":
            return "list_models expects /v1/models to return JSON with a top-level data array; HTML or truncated proxy output usually means CLIProxy failed before serialization."
        if issue == "unexpected_shape":
            return "list_models expects /v1/models -> {data:[{id: ...}]}; inspect CLIProxy model listing output for a shape drift at that boundary."

    if endpoint == "/v1/chat/completions":
        if issue == "request_failed":
            return "chat_completions posts to CLIProxyAPI /v1/chat/completions for non-GPT-5 models; inspect provider auth or upstream reachability for the selected model."
        if issue == "invalid_json":
            return "chat_completions expects JSON choices[0].message.content from /v1/chat/completions; inspect CLIProxy output for HTML, truncation, or gateway error pages."
        if issue == "unexpected_shape":
            return "chat_completions expects /v1/chat/completions -> choices[0].message.content as a string; inspect the provider response shape before the OpenAI-compat projection."
        if issue == "non_text":
            return "chat_completions expects /v1/chat/completions to yield text content after the OpenAI-compat projection; inspect whether the provider returned tool or multimodal blocks instead of plain text."
        if issue == "http_error":
            return "chat_completions reached CLIProxyAPI but got an HTTP error; inspect provider auth, model availability, or endpoint compatibility for the requested model."

    if endpoint == "/v1/responses":
        if issue == "request_failed":
            return "responses posts to CLIProxyAPI /v1/responses for GPT-5-style calls; inspect provider auth or upstream reachability for the selected model."
        if issue == "invalid_json":
            return "responses expects JSON with output_text or output blocks; inspect CLIProxy output for HTML, truncation, or gateway error pages."
        if issue == "unexpected_shape":
            return "responses expects /v1/responses to return output_text or an output array with output_text blocks; inspect the OpenAI Responses projection before text reconstruction."
        if issue == "http_error":
            return "responses reached CLIProxyAPI but got an HTTP error; inspect provider auth, model availability, or endpoint compatibility for the requested model."

    return f"AsyncOpenAICompatClient hit {endpoint} and failed during {issue}; inspect the CLIProxy boundary and the expected OpenAI-compatible response contract."


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
    response: httpx.Response, *, endpoint: str, prefix: str, fallback_message: str
) -> UpstreamError:
    try:
        payload = response.json()
    except Exception:
        payload = None

    parsed = _parse_openai_error(payload)
    if parsed is not None:
        msg = f"{prefix}: {parsed.message}"
        return UpstreamError(
            msg,
            status_code=response.status_code,
            operator_hint=_operator_hint(endpoint, "http_error"),
        )

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
    return UpstreamError(
        msg,
        status_code=response.status_code,
        operator_hint=_operator_hint(endpoint, "http_error"),
    )


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
        or "chat completions" in text
        and "not supported" in text
    )


def _is_openai_gpt5_model(model_id: str) -> bool:
    mid = (model_id or "").strip().lower()
    if not mid:
        return False
    mid = mid.removeprefix("openai/")
    return mid.startswith("gpt-5")


def _openai_reasoning_effort(model_id: str) -> str | None:
    """Infer a reasoning effort from GPT-5 model ids.

    Defaults to medium for GPT-5.* ids when no explicit tier is encoded.
    """
    for env_name in ("QEEG_OPENAI_REASONING_EFFORT", "OPENAI_REASONING_EFFORT"):
        override = (os.getenv(env_name) or "").strip().lower()
        if override in {"minimal", "low", "medium", "high", "xhigh"}:
            return override

    mid = (model_id or "").strip().lower()
    if not mid:
        return None
    mid = mid.removeprefix("openai/")
    if not mid.startswith("gpt-5."):
        return None

    # Prefer explicit effort suffix when present (e.g. "...-high", "...-xhigh").
    for token in reversed([t for t in mid.split("-") if t]):
        if token in {"minimal", "low", "medium", "high", "xhigh"}:
            return token

    return "medium"


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
            raise UpstreamError(
                f"CLIProxyAPI request failed: {e}",
                operator_hint=_operator_hint("/v1/models", "request_failed"),
            ) from e

        if resp.status_code >= 400:
            raise _format_http_error(
                resp,
                endpoint="/v1/models",
                prefix="CLIProxyAPI /v1/models failed",
                fallback_message=f"HTTP {resp.status_code}",
            )

        try:
            payload = resp.json()
        except Exception as e:
            raise UpstreamError(
                f"CLIProxyAPI /v1/models returned invalid JSON: {e}",
                operator_hint=_operator_hint("/v1/models", "invalid_json"),
            ) from e

        data = payload.get("data") if isinstance(payload, dict) else None
        if not isinstance(data, list):
            raise UpstreamError(
                "CLIProxyAPI /v1/models returned unexpected shape",
                operator_hint=_operator_hint("/v1/models", "unexpected_shape"),
            )

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
        reasoning_effort = _openai_reasoning_effort(model_id)
        if _is_openai_gpt5_model(model_id):
            return await self.responses(
                model_id=model_id,
                input_data=self._messages_to_responses_input(messages),
                stream=stream,
                reasoning_effort=reasoning_effort,
                max_output_tokens=max_tokens,
            )

        payload = {
            "model": model_id,
            "messages": messages,
            "temperature": temperature,
            "stream": stream,
        }
        if _is_openai_gpt5_model(model_id):
            payload["max_completion_tokens"] = max_tokens
        else:
            payload["max_tokens"] = max_tokens
        if reasoning_effort:
            payload["reasoning_effort"] = reasoning_effort

        try:
            resp = await client.post("/v1/chat/completions", json=payload)
        except Exception as e:
            raise UpstreamError(
                f"CLIProxyAPI request failed: {e}",
                operator_hint=_operator_hint(
                    "/v1/chat/completions", "request_failed"
                ),
            ) from e

        if resp.status_code >= 400:
            err = _format_http_error(
                resp,
                endpoint="/v1/chat/completions",
                prefix="CLIProxyAPI /v1/chat/completions failed",
                fallback_message=f"HTTP {resp.status_code}",
            )
            if _chat_unsupported(err):
                return await self.responses(
                    model_id=model_id,
                    input_data=self._messages_to_responses_input(messages),
                    stream=stream,
                    reasoning_effort=reasoning_effort,
                    max_output_tokens=max_tokens,
                )
            raise err

        try:
            data = resp.json()
        except Exception as e:
            raise UpstreamError(
                f"CLIProxyAPI /v1/chat/completions returned invalid JSON: {e}",
                operator_hint=_operator_hint(
                    "/v1/chat/completions", "invalid_json"
                ),
            ) from e

        try:
            content = data["choices"][0]["message"]["content"]
        except Exception as e:
            raise UpstreamError(
                f"CLIProxyAPI /v1/chat/completions returned unexpected shape: {e}",
                operator_hint=_operator_hint(
                    "/v1/chat/completions", "unexpected_shape"
                ),
            ) from e

        if not isinstance(content, str):
            raise UpstreamError(
                "CLIProxyAPI /v1/chat/completions returned non-text content",
                operator_hint=_operator_hint("/v1/chat/completions", "non_text"),
            )
        return content

    async def responses(
        self,
        *,
        model_id: str,
        input_data: Any,
        stream: bool = False,
        reasoning_effort: str | None = None,
        max_output_tokens: int | None = None,
    ) -> str:
        client = self._get_client()
        payload = {"model": model_id, "input": input_data, "stream": stream}
        if reasoning_effort:
            payload["reasoning"] = {"effort": reasoning_effort}
        if isinstance(max_output_tokens, int) and max_output_tokens > 0:
            payload["max_output_tokens"] = max_output_tokens

        try:
            resp = await client.post("/v1/responses", json=payload)
        except Exception as e:
            raise UpstreamError(
                f"CLIProxyAPI request failed: {e}",
                operator_hint=_operator_hint("/v1/responses", "request_failed"),
            ) from e

        if resp.status_code >= 400:
            raise _format_http_error(
                resp,
                endpoint="/v1/responses",
                prefix="CLIProxyAPI /v1/responses failed",
                fallback_message=f"HTTP {resp.status_code}",
            )

        try:
            data = resp.json()
        except Exception as e:
            raise UpstreamError(
                f"CLIProxyAPI /v1/responses returned invalid JSON: {e}",
                operator_hint=_operator_hint("/v1/responses", "invalid_json"),
            ) from e

        # OpenAI Responses API: output_text may be present; otherwise reconstruct from output blocks.
        output_text = data.get("output_text") if isinstance(data, dict) else None
        if isinstance(output_text, str):
            return output_text

        if not isinstance(data, dict) or not isinstance(data.get("output"), list):
            raise UpstreamError(
                "CLIProxyAPI /v1/responses returned unexpected shape",
                operator_hint=_operator_hint("/v1/responses", "unexpected_shape"),
            )

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

    @staticmethod
    def _messages_to_responses_input(messages: list[dict]) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for message in messages:
            role = message.get("role", "user")
            if not isinstance(role, str):
                role = "user"
            content = message.get("content", "")
            blocks: list[dict[str, Any]] = []

            if isinstance(content, str):
                blocks.append({"type": "input_text", "text": content})
            elif isinstance(content, list):
                for item in content:
                    if not isinstance(item, dict):
                        blocks.append({"type": "input_text", "text": str(item)})
                        continue
                    item_type = item.get("type")
                    if item_type == "text" and isinstance(item.get("text"), str):
                        blocks.append({"type": "input_text", "text": item["text"]})
                        continue
                    if item_type == "image_url":
                        image_url = item.get("image_url")
                        if isinstance(image_url, dict) and isinstance(
                            image_url.get("url"), str
                        ):
                            blocks.append(
                                {"type": "input_image", "image_url": image_url["url"]}
                            )
                        continue
                    try:
                        blocks.append({"type": "input_text", "text": json.dumps(item)})
                    except Exception:
                        blocks.append({"type": "input_text", "text": str(item)})
            else:
                try:
                    blocks.append({"type": "input_text", "text": json.dumps(content)})
                except Exception:
                    blocks.append({"type": "input_text", "text": str(content)})

            out.append({"role": role, "content": blocks})
        return out
