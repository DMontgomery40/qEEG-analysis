from __future__ import annotations

import json
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime, timezone
from typing import Any

from ...config import ARTIFACTS_DIR
from ...llm_client import UpstreamError
from ...storage import Artifact, create_artifact
from ...storage import session_scope
from ..execution import (
    raise_if_execution_blocked,
    require_semantic_scope,
    execution_llm,
)
from ..paths import _artifact_path, _stage_dir
from ..types import PageImage, StageDef
from ..utils import _sleep_backoff
from .exceptions import _NeedsAuth


_USAGE_RUN_ID: ContextVar[str | None] = ContextVar("qeeg_usage_run_id", default=None)

# A Stage 1 member could be sent ten times on 429/5xx: five sends in the
# primary unit, then five more in the reduced-budget pass (EN-H12). Every send
# the member's primary and reduced-budget units make is now charged to one
# budget; a retry or the reduced pass is refused once six are spent. A repair
# continuation's first send is new work, bounded by QEEG_LONGFORM_REPAIR_CALLS,
# and is charged without being refused.
MEMBER_SEND_LIMIT = 6
_MEMBER_SENDS: ContextVar["MemberSendBudget | None"] = ContextVar(
    "qeeg_member_sends", default=None
)
_RETRYABLE_STATUS = {429, 500, 502, 503, 504}
# A 429 that names a usage window (a weekly or monthly allowance, a quota)
# will not clear in seconds; re-sending only repeats the refusal.
_WINDOW_LIMIT_WORDS = ("usage limit", "usage_limit", "quota", "weekly", "monthly")


class MemberSendBudget:
    def __init__(self, limit: int = MEMBER_SEND_LIMIT):
        self.limit = limit
        self.sent = 0

    @property
    def spent(self) -> bool:
        return self.sent >= self.limit


@contextmanager
def member_send_budget(limit: int = MEMBER_SEND_LIMIT):
    """Charge every send in this context (and tasks started inside it) to one budget."""
    budget = MemberSendBudget(limit)
    token = _MEMBER_SENDS.set(budget)
    try:
        yield budget
    finally:
        _MEMBER_SENDS.reset(token)


def member_budget_spent() -> bool:
    budget = _MEMBER_SENDS.get()
    return budget is not None and budget.spent


def is_window_limit(error: BaseException | None) -> bool:
    """True for a 429 whose body says a usage window or quota is spent."""
    seen = set()
    while error is not None and id(error) not in seen:
        seen.add(id(error))
        if isinstance(error, UpstreamError) and error.status_code == 429:
            text = " ".join(
                str(part)
                for part in (
                    error,
                    getattr(error, "error_type", None),
                    getattr(error, "error_code", None),
                )
                if part
            ).lower()
            if any(word in text for word in _WINDOW_LIMIT_WORDS):
                return True
        error = error.__cause__ or error.__context__
    return False


def _charge_send() -> None:
    budget = _MEMBER_SENDS.get()
    if budget is not None:
        budget.sent += 1


def _may_resend(error: UpstreamError, attempts: int) -> bool:
    if not (error.status_code in _RETRYABLE_STATUS or error.status_code is None):
        return False
    if attempts >= 4 or is_window_limit(error):
        return False
    return not member_budget_spent()


class _LLMCallsMixin:
    def _record_model_usage(
        self,
        *,
        model_id: str,
        call_kind: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        run_id = _USAGE_RUN_ID.get()
        if metadata is None:
            metadata = getattr(self._llm, "last_response_metadata", None)
        if not run_id or not isinstance(metadata, dict):
            return
        if not metadata.get("raw_usage"):
            return

        ledger = ARTIFACTS_DIR / run_id / "usage.jsonl"
        ledger.parent.mkdir(parents=True, exist_ok=True)
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "run_id": run_id,
            "call_kind": call_kind,
            "model_id": model_id,
            "requested_model_id": metadata.get("requested_model_id"),
            "api_model_id": metadata.get("api_model_id"),
            "provider": metadata.get("provider"),
            "endpoint": metadata.get("endpoint"),
            "input_tokens": metadata.get("input_tokens"),
            "output_tokens": metadata.get("output_tokens"),
            "cache_read_tokens": metadata.get("cache_read_tokens"),
            "output_reasoning_tokens": metadata.get("output_reasoning_tokens"),
            "total_tokens": metadata.get("total_tokens"),
            "cost_usd": metadata.get("cost_usd"),
            "raw_usage": metadata.get("raw_usage"),
        }
        with ledger.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, sort_keys=True) + "\n")

    async def _call_model_chat(
        self,
        *,
        model_id: str,
        prompt_text: str,
        temperature: float,
        max_tokens: int,
    ) -> str:
        require_semantic_scope()
        attempts = 0
        while True:
            _charge_send()
            try:
                text = await execution_llm(self._llm).chat_completions(
                    model_id=model_id,
                    messages=[{"role": "user", "content": prompt_text}],
                    temperature=temperature,
                    max_tokens=max_tokens,
                    stream=False,
                    usage_callback=lambda metadata: self._record_model_usage(
                        model_id=model_id,
                        call_kind="chat",
                        metadata=metadata,
                    ),
                )
                return text
            except UpstreamError as e:
                raise_if_execution_blocked(e)
                if e.status_code == 401:
                    raise _NeedsAuth(str(e)) from e
                if _may_resend(e, attempts):
                    await _sleep_backoff(attempts)
                    attempts += 1
                    continue
                raise

    async def _call_model_multimodal(
        self,
        *,
        model_id: str,
        prompt_text: str,
        images: list[PageImage],
        temperature: float,
        max_tokens: int,
        allow_text_fallback: bool = True,
    ) -> str:
        """Call a vision-capable model with text and images."""
        # Build multimodal content array
        content: list[dict] = [{"type": "text", "text": prompt_text}]

        # Add images (page-tagged, in order).
        for img in images:
            tag = f"[PAGE {img.page}]"
            if img.label:
                tag = f"{tag} [{img.label}]"
            content.append({"type": "text", "text": tag})
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{img.base64_png}",
                    "detail": "high"  # Use high detail for clinical data
                }
            })

        messages = [{"role": "user", "content": content}]

        require_semantic_scope()
        attempts = 0
        while True:
            _charge_send()
            try:
                text = await execution_llm(self._llm).chat_completions(
                    model_id=model_id,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    stream=False,
                    usage_callback=lambda metadata: self._record_model_usage(
                        model_id=model_id,
                        call_kind="multimodal",
                        metadata=metadata,
                    ),
                )
                return text
            except UpstreamError as e:
                raise_if_execution_blocked(e)
                if e.status_code == 401:
                    raise _NeedsAuth(str(e)) from e
                if _may_resend(e, attempts):
                    await _sleep_backoff(attempts)
                    attempts += 1
                    continue
                # If multimodal fails, optionally fall back to text-only (NOT suitable for strict data capture).
                if allow_text_fallback and attempts == 0 and not is_window_limit(e):
                    return await self._call_model_chat(
                        model_id=model_id,
                        prompt_text=prompt_text,
                        temperature=temperature,
                        max_tokens=max_tokens,
                    )
                raise

    async def _write_artifact(
        self,
        *,
        run_id: str,
        stage: StageDef,
        model_id: str,
        text: str,
    ) -> Artifact:
        out_dir = _stage_dir(run_id, stage.num)
        out_dir.mkdir(parents=True, exist_ok=True)
        path = _artifact_path(run_id, stage.num, model_id, stage.ext)
        path.write_text(text, encoding="utf-8")
        with session_scope() as session:
            artifact = create_artifact(
                session,
                run_id=run_id,
                stage_num=stage.num,
                stage_name=stage.name,
                model_id=model_id,
                kind=stage.kind,
                content_path=path,
                content_type=stage.content_type,
            )
        return artifact
