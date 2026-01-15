from __future__ import annotations

from typing import Any

from ...llm_client import UpstreamError
from ...storage import Artifact, create_artifact
from ...storage import session_scope
from ..paths import _artifact_path, _stage_dir
from ..types import PageImage, StageDef
from ..utils import _sleep_backoff
from .exceptions import _NeedsAuth


class _LLMCallsMixin:
    async def _call_model_chat(
        self,
        *,
        model_id: str,
        prompt_text: str,
        temperature: float,
        max_tokens: int,
    ) -> str:
        attempts = 0
        while True:
            try:
                return await self._llm.chat_completions(
                    model_id=model_id,
                    messages=[{"role": "user", "content": prompt_text}],
                    temperature=temperature,
                    max_tokens=max_tokens,
                    stream=False,
                )
            except UpstreamError as e:
                if e.status_code == 401:
                    raise _NeedsAuth(str(e)) from e
                if (e.status_code in {429, 500, 502, 503, 504} or e.status_code is None) and attempts < 4:
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

        attempts = 0
        while True:
            try:
                return await self._llm.chat_completions(
                    model_id=model_id,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    stream=False,
                )
            except UpstreamError as e:
                if e.status_code == 401:
                    raise _NeedsAuth(str(e)) from e
                if (e.status_code in {429, 500, 502, 503, 504} or e.status_code is None) and attempts < 4:
                    await _sleep_backoff(attempts)
                    attempts += 1
                    continue
                # If multimodal fails, optionally fall back to text-only (NOT suitable for strict data capture).
                if allow_text_fallback and attempts == 0:
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


