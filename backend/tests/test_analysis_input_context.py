"""Recorded model traffic verifies saved operator context at its real boundaries."""

from __future__ import annotations

import json
import uuid

import httpx
import pytest

from backend import storage
from backend.tests.test_analysis_inputs import admission as admission, source


def test_normal_start_carries_instructions_through_all_stages_and_reviews(
    admission,  # noqa: F811 - imported shared pytest fixture
    monkeypatch,
):
    from backend.council import QEEGCouncilWorkflow
    from backend.llm_client import AsyncOpenAICompatClient
    from backend.tests.fixtures.mock_llm import (
        create_mock_transport,
        detect_stage,
    )

    client, payload, tmp, main = admission
    instructions = "  Investigate return-to-work fatigue.\nOperator says café noise is difficult.  \n"
    originals = [
        source(tmp, payload["patient_id"], date=date, pages=6)
        for date in ["2026-02-02", "2026-01-02"]
    ]
    payload = {
        **payload,
        "council_model_ids": ["mock-council-a", "mock-council-b"],
        "report_ids": [r.id for r in originals],
        "special_instructions": instructions,
        "operation_id": "whole-pipeline",
    }
    created = client.post("/api/runs", json=payload)
    assert created.status_code == 200, created.text
    run_id = created.json()["id"]
    requests = []
    ordinary = create_mock_transport()
    invalid_sent = set()

    def handle(request):
        if request.method == "POST":
            body = json.loads(request.content)
            stage = detect_stage(body.get("messages", []))
            requests.append((stage, body))
            response = ordinary.handle_request(request)
            # Pydantic review response repair must retain the original operator context.
            if stage in (2, 5) and stage not in invalid_sent:
                invalid_sent.add(stage)
                result = response.json()
                result["choices"][0]["message"]["content"] = "{}"
                return httpx.Response(200, json=result)
            return response
        return ordinary.handle_request(request)

    llm = AsyncOpenAICompatClient(
        base_url="http://mock-cliproxy:8317",
        api_key="",
        transport=httpx.MockTransport(handle),
    )
    from functools import partial
    from backend.run_runtime import continue_owned_run
    from backend.tests.test_patient_postprocessing import text as patient_text
    from scripts import generate_patient_facing_writeups as writer
    from backend.paid_transport import current_paid_scope

    generated = []
    original_handle = handle

    def owned_handle(request):
        scope = current_paid_scope()
        if scope is not None and scope.semantic_key.startswith("post/"):
            generated.append(request.content)
            return httpx.Response(
                200, json={"choices": [{"message": {"content": patient_text()}}]}
            )
        return original_handle(request)

    llm._transport = httpx.MockTransport(owned_handle)
    main.app.state.run_runtime.continuation = partial(
        continue_owned_run,
        llm=llm,
        workflow=QEEGCouncilWorkflow(llm=llm),
        sync=lambda _: True,
    )
    monkeypatch.setenv("QEEG_AUTO_CATHODE_VIDEO", "0")
    monkeypatch.setenv("QEEG_STAGE6_FINAL_DRAFT_MODEL", "mock-council-a")
    monkeypatch.setenv("QEEG_PATIENT_FACING_MODEL", "mock-council-a")
    monkeypatch.setenv("QEEG_AUTO_PATIENT_FACING", "1")
    monkeypatch.setenv("QEEG_ROUTE_OPENROUTER_EXTRAS_DIRECT", "0")
    monkeypatch.setattr(
        writer,
        "render_patient_facing_markdown_to_pdf",
        lambda md, path, **kw: path.write_bytes(b"%PDF synthetic owned downstream"),
    )

    async def forbidden(*args, **kwargs):
        raise AssertionError("API start used the old unowned post subprocess")

    monkeypatch.setattr(main.asyncio, "create_subprocess_exec", forbidden)
    response = client.post(f"/api/runs/{run_id}/start")
    assert response.status_code == 200, response.text
    client.portal.call(main.app.state.run_runtime._work, run_id)
    client.portal.call(llm.aclose)
    with storage.session_scope() as s:
        run = storage.get_run(s, run_id)
        assert run.status == "complete", run.error_message
        assert storage.get_patient(s, payload["patient_id"]).notes == ""
    assert len(generated) == 1
    assert (
        client.get(f"/api/runs/{run_id}").json()["patient_facing"]["verified"] is True
    )
    assert {stage for stage, _ in requests} == {1, 2, 3, 4, 5, 6}
    for stage, body in requests:
        text = "\n".join(
            m["content"] for m in body["messages"] if isinstance(m.get("content"), str)
        )
        assert instructions in text, stage
    assert sum(stage == 2 for stage, _ in requests) >= 3
    assert sum(stage == 5 for stage, _ in requests) >= 3
    for original in originals:
        assert instructions not in open(original.extracted_text_path).read()


@pytest.mark.asyncio
@pytest.mark.parametrize("stage", [1, 3, 6])
async def test_narrative_response_repair_retains_exact_context(
    temp_data_dir, mock_llm_client, monkeypatch, stage
):
    from backend.analysis_inputs import saved_operator_instructions
    from backend.council import QEEGCouncilWorkflow
    from backend.council.prompts import _workflow_context_block
    from backend.tests.test_data_pack_timeouts import _create_run_with_report

    run_id, _ = _create_run_with_report("=== PAGE 1 / 1 ===\nSession 1\n")
    instructions = " \nExplain sensory workload.  "
    with storage.session_scope() as s:
        storage.get_run(s, run_id).special_instructions = instructions
        s.commit()
    prompts = []

    async def call(**kwargs):
        prompts.append(kwargs["prompt_text"])
        return (
            "# Findings\npartial"
            if len(prompts) == 1
            else "# Findings\ncomplete\n<!-- DONE -->"
        )

    workflow = QEEGCouncilWorkflow(llm=mock_llm_client)
    monkeypatch.setattr(workflow, "_call_model_chat", call)
    prompt = _workflow_context_block(
        stage_num=stage,
        stage_name="narrative",
        special_instructions=saved_operator_instructions(run_id),
    )
    result = await workflow._call_longform_chat_with_repairs(
        model_id="test-model",
        prompt_text=prompt,
        temperature=0.1,
        max_tokens=100,
        end_sentinel="<!-- DONE -->",
        required_headings=["# Findings"],
    )
    assert "<!-- DONE -->" in result
    assert len(prompts) == 2
    assert all(instructions in p for p in prompts)


@pytest.mark.asyncio
async def test_consolidation_response_repair_loads_saved_instructions(
    temp_data_dir, mock_llm_client, monkeypatch
):
    from backend.council import QEEGCouncilWorkflow
    from backend.tests.test_stage4_consolidation_repair import _create_stage4_ready_run

    run_id = _create_stage4_ready_run(
        report_id=str(uuid.uuid4()), consolidator_model_id="test-model"
    )
    instructions = "\n  Focus on job demands, exactly.\n"
    with storage.session_scope() as s:
        storage.get_run(s, run_id).special_instructions = instructions
        s.commit()
    prompts = []

    async def call(**kwargs):
        prompts.append(kwargs["prompt_text"])
        return (
            "# Dataset and Sessions\npartial"
            if len(prompts) == 1
            else "# Dataset and Sessions\ncomplete\n<!-- END CONSOLIDATED REPORT -->"
        )

    async def emit(payload):
        pass

    workflow = QEEGCouncilWorkflow(llm=mock_llm_client)
    monkeypatch.setattr(workflow, "_call_model_chat", call)
    await workflow._stage4(run_id, emit)
    assert len(prompts) == 2
    assert all(instructions in p for p in prompts)


@pytest.mark.asyncio
async def test_saved_operator_context_never_enters_strict_transcription(
    temp_data_dir, monkeypatch
):
    from backend.council.types import PageImage
    from backend.tests.test_data_pack_timeouts import (
        _TimeoutCapturingDataPack,
        _create_run_with_report,
    )

    run_id, report = _create_run_with_report("=== PAGE 1 / 1 ===\nSession 1\n")
    instructions = "Operator-only context: measure nothing from this sentence."
    with storage.session_scope() as s:
        storage.get_run(s, run_id).special_instructions = instructions
        s.commit()
    captured = []
    workflow = _TimeoutCapturingDataPack()
    ordinary = workflow._call_model_multimodal

    async def record(**kwargs):
        captured.append(kwargs["prompt_text"])
        return await ordinary(**kwargs)

    monkeypatch.setattr(workflow, "_call_model_multimodal", record)
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
    assert len(captured) >= 2
    assert all(instructions not in prompt for prompt in captured)
