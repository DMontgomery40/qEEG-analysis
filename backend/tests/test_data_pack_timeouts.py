from __future__ import annotations

import json
import os
import uuid
from pathlib import Path
from typing import Any, Awaitable

import pytest

from backend.council.constants import DATA_PACK_SCHEMA_VERSION
from backend.council.types import PageImage
from backend.council.workflow.data_pack import _DataPackMixin


def _create_run_with_report(report_text: str) -> tuple[str, Any]:
    from backend.config import REPORTS_DIR
    from backend.storage import (
        create_patient,
        create_report,
        create_run,
        get_report,
        session_scope,
    )

    report_id = str(uuid.uuid4())
    report_dir = REPORTS_DIR / "timeout-test" / report_id
    report_dir.mkdir(parents=True, exist_ok=True)
    stored_path = report_dir / "original.pdf"
    extracted_path = report_dir / "extracted.txt"
    stored_path.write_bytes(b"%PDF-1.4\n")
    extracted_path.write_text(report_text, encoding="utf-8")

    with session_scope() as session:
        patient = create_patient(session, label="SL_01-01-1990", notes="")
        report = create_report(
            session,
            report_id=report_id,
            patient_id=patient.id,
            filename="original.pdf",
            mime_type="application/pdf",
            stored_path=stored_path,
            extracted_text_path=extracted_path,
        )
        run = create_run(
            session,
            patient_id=patient.id,
            report_id=report.id,
            council_model_ids=["vision-model"],
            consolidator_model_id="vision-model",
        )
        return run.id, get_report(session, report.id)


class _TimeoutCapturingDataPack(_DataPackMixin):
    def __init__(self) -> None:
        self.calls: list[tuple[str | None, int | None]] = []

    @staticmethod
    def _int_env(name: str, default: int) -> int:
        try:
            return int(os.getenv(name, str(default)) or str(default))
        except Exception:
            return default

    async def _await_with_heartbeat(
        self,
        awaitable: Awaitable[Any],
        *,
        emit,
        payload: dict[str, Any],
        timeout_s: int | None = None,
    ) -> Any:
        self.calls.append((payload.get("task"), timeout_s))
        return await awaitable

    async def _call_model_multimodal(
        self,
        *,
        model_id: str,
        prompt_text: str,
        images: list[PageImage],
        temperature: float,
        max_tokens: int,
        allow_text_fallback: bool,
    ) -> str:
        if "LOSSLESS MULTIMODAL TRANSCRIPTION" in prompt_text:
            return "## Page 1\nVisible table text"
        return json.dumps(
            {
                "schema_version": DATA_PACK_SCHEMA_VERSION,
                "pages_seen": [img.page for img in images],
                "page_inventory": [],
                "facts": [],
                "unparsed_required": [],
            }
        )

    @staticmethod
    def _missing_required_fields(
        data_pack: dict[str, Any], *, expected_sessions: list[int]
    ) -> set[str]:
        return set()


@pytest.mark.asyncio
async def test_stage1_multimodal_extraction_passes_configured_timeouts(
    temp_data_dir: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setenv("QEEG_STAGE1_MULTIMODAL_TIMEOUT_S", "17")
    monkeypatch.setenv("QEEG_STAGE1_DATA_PACK_TIMEOUT_S", "23")
    monkeypatch.setenv("QEEG_STAGE1_VISION_TRANSCRIPT_TIMEOUT_S", "19")

    run_id, report = _create_run_with_report("=== PAGE 1 / 1 ===\nSession 1\n")
    workflow = _TimeoutCapturingDataPack()
    images = [PageImage(page=1, base64_png="ZmFrZQ==")]

    await workflow._ensure_data_pack(
        run_id=run_id,
        report=report,
        report_text="=== PAGE 1 / 1 ===\nSession 1\n",
        page_images=images,
        candidate_extractor_model_ids=["vision-model"],
        strict=False,
    )
    await workflow._ensure_vision_transcript(
        run_id=run_id,
        report=report,
        page_images=images,
        transcript_model_id="vision-model",
        strict=False,
    )

    assert ("data_pack_chunk", 23) in workflow.calls
    assert ("vision_transcript_chunk", 19) in workflow.calls
