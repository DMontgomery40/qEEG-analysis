"""Mock transport for CLIProxyAPI endpoints.

This module provides a mock httpx transport that simulates CLIProxyAPI responses
for the endpoints used by AsyncOpenAICompatClient:
- GET /v1/models
- POST /v1/chat/completions
- POST /v1/responses (fallback)

The mock detects which pipeline stage is calling based on prompt content
and returns appropriate stage-specific responses.
"""

from __future__ import annotations

import json
from typing import Callable

import httpx

from backend.tests.fixtures.mock_responses import (
    STAGE1_MARKDOWN,
    STAGE2_JSON,
    STAGE3_MARKDOWN,
    STAGE4_MARKDOWN,
    STAGE5_JSON,
    STAGE6_MARKDOWN,
)

# Model IDs that won't trigger vision/multimodal path
MOCK_MODEL_IDS = ["mock-council-a", "mock-council-b", "mock-consolidator"]


def detect_stage(messages: list[dict]) -> int:
    """Detect which pipeline stage is calling based on prompt content.

    The detection relies on keywords from the stage prompt templates:
    - Stage 2: "blinded peer review", "Analysis A" (anonymized labels)
    - Stage 5: "performing final review", "Before voting"
    - Stage 3: "revising YOUR prior", "revision"
    - Stage 4: "consolidator", "synthesize all revised"
    - Stage 6: "final polished draft", "Apply ALL required changes"
    - Stage 1: default (initial analysis)
    """
    # Combine all message content for keyword search
    text = ""
    for msg in messages:
        content = msg.get("content", "")
        if isinstance(content, str):
            text += content.lower()
        elif isinstance(content, list):
            # Handle multimodal messages (text + images)
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    text += part.get("text", "").lower()

    # Order matters: check most specific patterns first
    # Stage 6: final draft with required changes from Stage 5
    if "final polished draft" in text or ("final draft" in text and "apply all required changes" in text):
        return 6

    # Stage 5: final review with voting
    if "performing final review" in text or "before voting" in text:
        return 5

    # Stage 4: consolidation
    if "consolidator" in text or "synthesize all revised" in text:
        return 4

    # Stage 3: revision based on peer feedback
    if "revising your prior" in text or ("revision" in text and "peer review feedback" in text):
        return 3

    # Stage 2: blinded peer review
    if "blinded peer review" in text or "analysis a" in text or "analysis_label" in text:
        return 2

    # Default to Stage 1 (initial analysis)
    return 1


def create_mock_transport(
    on_request: Callable[[httpx.Request, int], None] | None = None
) -> httpx.MockTransport:
    """Create a mock HTTP transport for CLIProxyAPI endpoints.

    Args:
        on_request: Optional callback called with (request, detected_stage) for each request.
                    Useful for test assertions about what was sent.

    Returns:
        httpx.MockTransport configured to handle CLIProxyAPI endpoints.
    """

    def handler(request: httpx.Request) -> httpx.Response:
        # GET /v1/models - return mock model list
        if request.url.path == "/v1/models" and request.method == "GET":
            return httpx.Response(
                200,
                json={
                    "object": "list",
                    "data": [{"id": mid, "object": "model"} for mid in MOCK_MODEL_IDS],
                },
            )

        # POST /v1/chat/completions - main LLM endpoint
        if request.url.path == "/v1/chat/completions" and request.method == "POST":
            try:
                body = json.loads(request.content)
            except (json.JSONDecodeError, TypeError):
                return httpx.Response(400, json={"error": {"message": "Invalid JSON"}})

            messages = body.get("messages", [])
            stage = detect_stage(messages)

            if on_request is not None:
                on_request(request, stage)

            # Select response based on detected stage
            responses = {
                1: STAGE1_MARKDOWN,
                2: STAGE2_JSON,
                3: STAGE3_MARKDOWN,
                4: STAGE4_MARKDOWN,
                5: STAGE5_JSON,
                6: STAGE6_MARKDOWN,
            }
            content = responses.get(stage, STAGE1_MARKDOWN)

            return httpx.Response(
                200,
                json={
                    "id": f"mock-completion-stage-{stage}",
                    "object": "chat.completion",
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": content},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {"prompt_tokens": 100, "completion_tokens": 500, "total_tokens": 600},
                },
            )

        # POST /v1/responses - fallback endpoint for some models
        if request.url.path == "/v1/responses" and request.method == "POST":
            try:
                body = json.loads(request.content)
            except (json.JSONDecodeError, TypeError):
                return httpx.Response(400, json={"error": {"message": "Invalid JSON"}})

            input_text = body.get("input", "")
            # Simple stage detection from input text
            stage = 1
            input_lower = input_text.lower()
            if "peer review" in input_lower:
                stage = 2
            elif "final review" in input_lower:
                stage = 5

            if on_request is not None:
                on_request(request, stage)

            responses = {
                1: STAGE1_MARKDOWN,
                2: STAGE2_JSON,
                3: STAGE3_MARKDOWN,
                4: STAGE4_MARKDOWN,
                5: STAGE5_JSON,
                6: STAGE6_MARKDOWN,
            }
            content = responses.get(stage, STAGE1_MARKDOWN)

            return httpx.Response(
                200,
                json={
                    "id": f"mock-response-stage-{stage}",
                    "output_text": content,
                },
            )

        # Unknown endpoint
        return httpx.Response(
            404, json={"error": {"message": f"Unknown endpoint: {request.url.path}"}}
        )

    return httpx.MockTransport(handler)
