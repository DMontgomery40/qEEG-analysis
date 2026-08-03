from __future__ import annotations

import asyncio
import json
import os
from dataclasses import dataclass
from typing import Any, Callable

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


def _chat_content_text(content: Any) -> str | None:
    if isinstance(content, str):
        return content
    if isinstance(content, dict):
        text = content.get("text")
        if isinstance(text, str):
            return text
        nested = content.get("content")
        if nested is not content:
            return _chat_content_text(nested)
        return None
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            text = _chat_content_text(block)
            if text:
                parts.append(text)
        joined = "".join(parts)
        return joined if joined.strip() else None
    return None


def _is_openai_gpt5_model(model_id: str) -> bool:
    mid = (model_id or "").strip().lower()
    if not mid:
        return False
    mid = mid.removeprefix("openai/")
    return mid.startswith("gpt-5")


def _is_anthropic_claude_model(model_id: str) -> bool:
    mid = (model_id or "").strip().lower()
    if not mid:
        return False
    mid = mid.removeprefix("anthropic/")
    return mid.startswith("claude-")


def _openai_reasoning_effort(model_id: str) -> str | None:
    """Infer a reasoning effort from GPT-5 model ids.

    Defaults to low for GPT-5.5 ids and medium for older GPT-5.* ids when
    no explicit tier is encoded.
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

    if mid == "gpt-5.5" or mid.startswith("gpt-5.5-"):
        return "low"

    return "medium"


def _split_env_list(name: str) -> list[str]:
    raw = os.getenv(name) or ""
    items: list[str] = []
    for part in raw.replace("\n", ",").split(","):
        item = part.strip()
        if item:
            items.append(item)
    return items


def _extra_openrouter_model_ids() -> list[str]:
    configured = os.getenv("QEEG_OPENROUTER_EXTRA_MODELS")
    if configured is None:
        configured = "z-ai/glm-5.2"
    items: list[str] = []
    for part in configured.replace("\n", ",").split(","):
        item = part.strip()
        if item:
            items.append(item)
    return items


def _is_openrouter_extra_model(model_id: str) -> bool:
    mid = (model_id or "").strip().lower()
    if not mid:
        return False
    return mid in {m.lower() for m in _extra_openrouter_model_ids()}


def _env_bool(name: str) -> bool | None:
    raw = os.getenv(name)
    if raw is None:
        return None
    value = raw.strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    return None


def _route_openrouter_extras_direct() -> bool:
    """Require an explicit opt-in before bypassing CLIProxyAPI."""
    return _env_bool("QEEG_ROUTE_OPENROUTER_EXTRAS_DIRECT") is True


def _model_env_suffix(model_id: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in (model_id or "").upper()).strip(
        "_"
    )


def _openrouter_reasoning_config(model_id: str) -> dict[str, Any] | None:
    suffix = _model_env_suffix(model_id)
    effort = ""
    for name in (
        f"QEEG_OPENROUTER_REASONING_EFFORT_{suffix}",
        "QEEG_OPENROUTER_REASONING_EFFORT",
    ):
        effort = (os.getenv(name) or "").strip().lower()
        if effort:
            break

    exclude: bool | None = None
    for name in (
        f"QEEG_OPENROUTER_REASONING_EXCLUDE_{suffix}",
        "QEEG_OPENROUTER_REASONING_EXCLUDE",
    ):
        exclude = _env_bool(name)
        if exclude is not None:
            break

    config: dict[str, Any] = {}
    is_glm52_writer = (model_id or "").strip().lower() == "z-ai/glm-5.2"
    if not effort and is_glm52_writer:
        effort = "high"
    if exclude is None and is_glm52_writer:
        exclude = True
    if effort in {"none", "minimal", "low", "medium", "high", "xhigh"}:
        config["effort"] = effort
    if exclude is not None:
        config["exclude"] = exclude
    return config or None


def _is_glm52_writer(model_id: str) -> bool:
    return (model_id or "").strip().lower() == "z-ai/glm-5.2"


def _looks_like_reasoning_leak(content: str, message: dict[str, Any]) -> bool:
    text = (content or "").strip()
    lower = text.lower()
    if "<think>" in lower or "</think>" in lower:
        return True
    reasoning = message.get("reasoning")
    if isinstance(reasoning, str) and reasoning.strip():
        reasoning_text = reasoning.strip()
        if text == reasoning_text or reasoning_text in text:
            return True
    return False


def _non_openai_reasoning_effort(model_id: str) -> str | None:
    mid = (model_id or "").strip().lower()
    if not mid:
        return None
    reasoning_models = {m.lower() for m in _split_env_list("QEEG_REASONING_MODEL_IDS")}
    if mid not in reasoning_models:
        return None
    effort = (os.getenv("QEEG_REASONING_EFFORT") or "high").strip().lower()
    return effort if effort in {"minimal", "low", "medium", "high", "xhigh"} else "high"


def _usage_int(value: Any) -> int | None:
    try:
        if value is None:
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _usage_number(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _nested_dict(root: dict[str, Any], *keys: str) -> dict[str, Any]:
    for key in keys:
        value = root.get(key)
        if isinstance(value, dict):
            return value
    return {}


def _response_usage_metadata(
    data: dict[str, Any],
    *,
    requested_model_id: str,
    endpoint: str,
    provider: str,
) -> dict[str, Any] | None:
    usage = data.get("usage")
    if not isinstance(usage, dict):
        return None

    input_details = _nested_dict(usage, "input_tokens_details", "prompt_tokens_details")
    output_details = _nested_dict(usage, "output_tokens_details", "completion_tokens_details")
    input_tokens = (
        _usage_int(usage.get("input_tokens"))
        or _usage_int(usage.get("prompt_tokens"))
        or _usage_int(usage.get("input_token_count"))
    )
    output_tokens = (
        _usage_int(usage.get("output_tokens"))
        or _usage_int(usage.get("completion_tokens"))
        or _usage_int(usage.get("candidates_token_count"))
    )
    cache_read_tokens = (
        _usage_int(usage.get("cache_read_tokens"))
        or _usage_int(usage.get("cached_tokens"))
        or _usage_int(input_details.get("cached_tokens"))
    )
    total_tokens = (
        _usage_int(usage.get("total_tokens"))
        or _usage_int(usage.get("total_token_count"))
        or ((input_tokens or 0) + (output_tokens or 0) if input_tokens or output_tokens else None)
    )

    cost_usd = None
    for key in ("cost", "cost_usd", "total_cost", "total_cost_usd", "price", "price_usd"):
        cost_usd = _usage_number(usage.get(key))
        if cost_usd is not None:
            break
    if cost_usd is None and isinstance(data.get("cost"), (int, float, str)):
        cost_usd = _usage_number(data.get("cost"))

    if cost_usd is None and (input_tokens is not None or output_tokens is not None):
        try:
            from genai_prices import Usage, calc_price  # type: ignore

            price = calc_price(
                Usage(
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    cache_read_tokens=cache_read_tokens,
                ),
                model_ref=str(data.get("model") or requested_model_id),
            )
            cost_usd = float(price.total_price)
        except Exception:
            cost_usd = None

    return {
        "requested_model_id": requested_model_id,
        "api_model_id": data.get("model") if isinstance(data.get("model"), str) else requested_model_id,
        "endpoint": endpoint,
        "provider": provider,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cache_read_tokens": cache_read_tokens,
        "output_reasoning_tokens": _usage_int(output_details.get("reasoning_tokens")),
        "total_tokens": total_tokens,
        "cost_usd": cost_usd,
        "raw_usage": usage,
    }


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
        self._openrouter_client: httpx.AsyncClient | None = None
        self.last_response_metadata: dict[str, Any] | None = None

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None
        if self._openrouter_client is not None:
            await self._openrouter_client.aclose()
            self._openrouter_client = None

    def _remember_response_usage(
        self,
        data: dict[str, Any],
        *,
        requested_model_id: str,
        endpoint: str,
        provider: str,
        usage_callback: Callable[[dict[str, Any]], None] | None = None,
    ) -> dict[str, Any] | None:
        metadata = _response_usage_metadata(
            data,
            requested_model_id=requested_model_id,
            endpoint=endpoint,
            provider=provider,
        )
        self.last_response_metadata = metadata
        if metadata is not None and usage_callback is not None:
            usage_callback(metadata)
        return metadata

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

    def _get_openrouter_client(self) -> httpx.AsyncClient:
        if self._openrouter_client is None:
            api_key = (os.getenv("OPENROUTER_API_KEY") or "").strip()
            if not api_key:
                raise UpstreamError(
                    "OpenRouter model requested but OPENROUTER_API_KEY is not set",
                    operator_hint="Set OPENROUTER_API_KEY when QEEG_OPENROUTER_EXTRA_MODELS contains the selected model.",
                )
            base_url = (
                os.getenv("OPENROUTER_BASE_URL") or "https://openrouter.ai/api"
            ).rstrip("/")
            headers: dict[str, str] = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            }
            referer = (os.getenv("OPENROUTER_HTTP_REFERER") or "").strip()
            title = (os.getenv("OPENROUTER_APP_TITLE") or "").strip()
            if referer:
                headers["HTTP-Referer"] = referer
            if title:
                headers["X-Title"] = title
            self._openrouter_client = httpx.AsyncClient(
                base_url=base_url,
                headers=headers,
                timeout=httpx.Timeout(self._timeout_s),
                transport=self._transport,
            )
        return self._openrouter_client

    def _chat_completions_sync(self, payload: dict[str, Any]) -> httpx.Response:
        headers: dict[str, str] = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        with httpx.Client(
            base_url=self._base_url,
            headers=headers,
            timeout=httpx.Timeout(self._timeout_s),
        ) as client:
            return client.post("/v1/chat/completions", json=payload)

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
        # CLIProxy's advertised inventory is authoritative during normal
        # operation. Only advertise direct-provider extras when that emergency
        # bypass is explicitly enabled and has the credential it requires.
        if _route_openrouter_extras_direct() and (
            os.getenv("OPENROUTER_API_KEY") or ""
        ).strip():
            for model_id in _extra_openrouter_model_ids():
                if model_id not in ids:
                    ids.append(model_id)
        return ids

    async def chat_completions(
        self,
        *,
        model_id: str,
        messages: list[dict],
        temperature: float = 0.2,
        max_tokens: int = 1800,
        stream: bool = False,
        usage_callback: Callable[[dict[str, Any]], None] | None = None,
    ) -> str:
        self.last_response_metadata = None
        use_openrouter = _route_openrouter_extras_direct() and _is_openrouter_extra_model(
            model_id
        )
        client = self._get_openrouter_client() if use_openrouter else self._get_client()
        reasoning_effort = _openai_reasoning_effort(model_id)
        if _is_openai_gpt5_model(model_id):
            return await self.responses(
                model_id=model_id,
                input_data=self._messages_to_responses_input(messages),
                stream=stream,
                reasoning_effort=reasoning_effort,
                max_output_tokens=max_tokens,
                usage_callback=usage_callback,
            )

        payload = {
            "model": model_id,
            "messages": messages,
            "stream": stream,
        }
        if not _is_anthropic_claude_model(model_id):
            payload["temperature"] = temperature
        if _is_openai_gpt5_model(model_id):
            payload["max_completion_tokens"] = max_tokens
        else:
            payload["max_tokens"] = max_tokens
        if reasoning_effort and _is_openai_gpt5_model(model_id):
            payload["reasoning_effort"] = reasoning_effort
        openrouter_reasoning = (
            _openrouter_reasoning_config(model_id)
            if use_openrouter or _is_glm52_writer(model_id)
            else None
        )
        if openrouter_reasoning:
            payload["reasoning"] = openrouter_reasoning
        else:
            non_openai_reasoning_effort = _non_openai_reasoning_effort(model_id)
            if non_openai_reasoning_effort:
                payload["reasoning"] = {"effort": non_openai_reasoning_effort}

        try:
            if (
                _is_anthropic_claude_model(model_id)
                and self._transport is None
                and not use_openrouter
            ):
                resp = await asyncio.to_thread(self._chat_completions_sync, payload)
            else:
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
                    usage_callback=usage_callback,
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

        self._remember_response_usage(
            data,
            requested_model_id=model_id,
            endpoint="/v1/chat/completions",
            provider="openrouter" if use_openrouter else "cliproxy",
            usage_callback=usage_callback,
        )

        try:
            message = data["choices"][0]["message"]
            if not isinstance(message, dict):
                raise TypeError("message is not an object")
        except Exception as e:
            raise UpstreamError(
                f"CLIProxyAPI /v1/chat/completions returned unexpected shape: {e}",
                operator_hint=_operator_hint(
                    "/v1/chat/completions", "unexpected_shape"
                ),
            ) from e

        content_text = _chat_content_text(message.get("content"))
        if not isinstance(content_text, str) and not _is_glm52_writer(model_id):
            raise UpstreamError(
                "CLIProxyAPI /v1/chat/completions returned non-text content",
                operator_hint=_operator_hint("/v1/chat/completions", "non_text"),
            )
        if not isinstance(content_text, str):
            content_text = ""
        needs_glm_retry = _is_glm52_writer(model_id) and (
            not content_text.strip()
            or _looks_like_reasoning_leak(content_text, message)
        )
        if needs_glm_retry:
            retry_payload = dict(payload)
            retry_payload["messages"] = [
                *messages,
                {
                    "role": "user",
                    "content": (
                        "Return the complete publishable final draft only. "
                        "Do not include analysis, planning, chain-of-thought, or reasoning text."
                    ),
                },
            ]
            retry_payload["reasoning"] = {"effort": "none", "exclude": True}
            try:
                retry_response = await client.post(
                    "/v1/chat/completions", json=retry_payload
                )
            except Exception as e:
                raise UpstreamError(
                    f"OpenRouter GLM final-content retry failed: {e}",
                    operator_hint=_operator_hint(
                        "/v1/chat/completions", "request_failed"
                    ),
                ) from e
            if retry_response.status_code >= 400:
                raise _format_http_error(
                    retry_response,
                    endpoint="/v1/chat/completions",
                    prefix="OpenRouter GLM final-content retry failed",
                    fallback_message=f"HTTP {retry_response.status_code}",
                )
            try:
                retry_data = retry_response.json()
                retry_message = retry_data["choices"][0]["message"]
                retry_content = _chat_content_text(retry_message.get("content"))
            except Exception as e:
                raise UpstreamError(
                    f"OpenRouter GLM final-content retry returned unexpected shape: {e}",
                    operator_hint=_operator_hint(
                        "/v1/chat/completions", "unexpected_shape"
                    ),
                ) from e
            self._remember_response_usage(
                retry_data,
                requested_model_id=model_id,
                endpoint="/v1/chat/completions",
                provider="openrouter",
                usage_callback=usage_callback,
            )
            if (
                not isinstance(retry_content, str)
                or not retry_content.strip()
                or _looks_like_reasoning_leak(retry_content, retry_message)
            ):
                raise UpstreamError(
                    "OpenRouter GLM did not return publishable final content after one reasoning-disabled retry",
                    operator_hint=(
                        "The GLM writer returned empty or reasoning-like text twice; "
                        "do not publish it and retry the patient-facing generation later."
                    ),
                )
            return retry_content
        if not content_text.strip():
            await self.aclose()
            raise UpstreamError(
                "CLIProxyAPI /v1/chat/completions returned empty text content",
                operator_hint=_operator_hint(
                    "/v1/chat/completions", "unexpected_shape"
                ),
            )
        return content_text

    async def responses(
        self,
        *,
        model_id: str,
        input_data: Any,
        stream: bool = False,
        reasoning_effort: str | None = None,
        max_output_tokens: int | None = None,
        usage_callback: Callable[[dict[str, Any]], None] | None = None,
    ) -> str:
        self.last_response_metadata = None
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
        if isinstance(output_text, str) and output_text.strip():
            self._remember_response_usage(
                data,
                requested_model_id=model_id,
                endpoint="/v1/responses",
                provider="cliproxy",
                usage_callback=usage_callback,
            )
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
        text = "".join(chunks).strip()
        if not text:
            await self.aclose()
            raise UpstreamError(
                "CLIProxyAPI /v1/responses returned empty text content",
                operator_hint=_operator_hint("/v1/responses", "unexpected_shape"),
            )
        self._remember_response_usage(
            data,
            requested_model_id=model_id,
            endpoint="/v1/responses",
            provider="cliproxy",
            usage_callback=usage_callback,
        )
        return text

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
