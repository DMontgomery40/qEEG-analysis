"""Opt-in council execution contracts, with real SDK and synthetic HTTP only."""

import asyncio
import importlib
import json
import re
from pathlib import Path

import httpx
import pytest
from sqlalchemy import select

from backend import storage
from backend.run_execution import ExecutionStore, ExecutionConflict
from backend.paid_transport import PaidOutcomeUnknown
from backend.llm_client import AsyncOpenAICompatClient


@pytest.fixture(autouse=True)
def expected_sdk_reasoning_profile_warning(request):
    """The unchanged SDK settings intentionally request temperature with reasoning.

    Assert that specific installed-SDK warning in the real generation tests;
    unrelated warnings retain their normal pytest handling.
    """
    sdk_tests = {
        "test_real_sdk_validation_sequence_replays",
        "test_wrapped_sdk_unknown_never_retries_or_falls_back",
        "test_sdk_eligible_endpoint_fallback_replays_shared_request_sequence",
        "test_direct_none_sdk_client_uses_same_paid_boundary",
        "test_sdk_sticky_unknown_survives_failed_metadata_write",
        "test_sdk_deliberate_backoff_keeps_request_sequence_across_agent_runs",
    }
    if request.node.originalname in sdk_tests:
        with pytest.warns(
            UserWarning,
            match=re.escape(
                "Sampling parameters ['temperature'] are not supported when reasoning is enabled. "
                "These settings will be ignored."
            ),
        ):
            yield
    else:
        yield


def execution():
    assert importlib.util.find_spec(
        "backend.council.execution"
    ), "council execution context absent"
    return importlib.import_module("backend.council.execution")


@pytest.fixture
def owner(tmp_path):
    storage.reset_engine(f'sqlite:///{tmp_path / "app.db"}')
    storage.init_db()
    with storage.session_scope() as session:
        session.add(
            storage.Run(
                id="r",
                patient_id="p",
                report_id="q",
                status="created",
                council_model_ids_json='["gpt-5.5", "gpt-5.5"]',
                consolidator_model_id="gpt-5.5",
                analysis_input_fingerprint="original-input",
                source_manifest_json=' {"source": "original"} ',
            )
        )
        session.commit()
    store = ExecutionStore(storage.engine)
    store.request_run_start("r")
    handle = store.claim_run_owner("r")
    yield handle
    handle.close()


def client(send):
    return AsyncOpenAICompatClient(
        base_url="http://synthetic",
        api_key="test",
        timeout_s=31,
        transport=httpx.MockTransport(send),
    )


def fixture(name):
    return json.loads((Path(__file__).parent / "fixtures/ai_review" / name).read_text())


def completion(request, payload):
    body = json.loads(request.content)
    tool = body["tools"][0]["function"]["name"]
    return httpx.Response(
        200,
        json={
            "id": "stable",
            "model": body["model"],
            "object": "chat.completion",
            "created": 1,
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "tool_calls",
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "stable-tool",
                                "type": "function",
                                "function": {
                                    "name": tool,
                                    "arguments": json.dumps(payload),
                                },
                            }
                        ],
                    },
                }
            ],
        },
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "stage,invalid,valid",
    [
        (2, "stage2_missing_b.json", "stage2_valid.json"),
        (5, "stage5_approve_with_required_changes.json", "stage5_revise_valid.json"),
    ],
)
async def test_real_sdk_validation_sequence_replays(owner, stage, invalid, valid):
    from backend.council.ai_review_agents import (
        run_stage2_peer_review,
        run_stage5_final_review,
    )

    e = execution()
    sent = []

    def send(req):
        sent.append(req.content)
        return completion(req, fixture(invalid if len(sent) == 1 else valid))

    llm = client(send)
    ctx = e.prepare_execution(owner, llm_client=llm)
    for _ in range(2):
        with e.execution_context(ctx):
            fn = run_stage2_peer_review if stage == 2 else run_stage5_final_review
            kw = {"expected_labels": ["A", "B"]} if stage == 2 else {}
            result = await e.execute_unit(
                f"s{stage}/reviewer/0/gpt-5.5",
                fn(llm_client=llm, model_id="gpt-5.5", prompt_text="Review.", **kw),
            )
            assert result.model_dump(mode="json") == fixture(valid)
    assert len(sent) == 2
    assert len(json.loads(sent[1])["messages"]) > len(json.loads(sent[0])["messages"])


@pytest.mark.asyncio
@pytest.mark.parametrize("stage", [2, 5])
@pytest.mark.parametrize("failure", ["reset", "429", "503"])
async def test_wrapped_sdk_unknown_never_retries_or_falls_back(
    owner, monkeypatch, stage, failure
):
    from backend.council import ai_review_agents as agents

    e = execution()
    sent = []

    def send(req):
        sent.append(req.content)
        if failure == "reset":
            raise httpx.ReadError("lost response")
        return httpx.Response(int(failure), json={"error": {"message": "unavailable"}})

    llm = client(send)
    ctx = e.prepare_execution(owner, llm_client=llm)
    with e.execution_context(ctx), pytest.raises(PaidOutcomeUnknown):
        fn = (
            agents.run_stage2_peer_review
            if stage == 2
            else agents.run_stage5_final_review
        )
        kw = {"expected_labels": ["A", "B"]} if stage == 2 else {}
        await e.execute_unit(
            f"s{stage}/reviewer/0/gpt-5.5",
            fn(llm_client=llm, model_id="gpt-5.5", prompt_text="Review.", **kw),
        )
    assert len(sent) == 1


def test_manifest_freezes_settings_prompts_roles_and_original_admission(
    owner, monkeypatch
):
    from backend.council.workflow import stages
    from backend.council.prompts import _load_prompt

    e = execution()
    monkeypatch.setenv("QEEG_STAGE1_MAX_TOKENS", "321")
    monkeypatch.setattr(stages, "DISCOVERED_MODEL_IDS", {"gpt-5.5", "z-ai/glm-5.2"})
    llm = client(lambda _: None)
    ctx = e.prepare_execution(owner, llm_client=llm)
    old = ctx.manifest_hash
    original = _load_prompt("stage1_analysis.md")
    monkeypatch.setenv("QEEG_STAGE1_MAX_TOKENS", "999")
    monkeypatch.setenv("QEEG_STAGE6_FINAL_DRAFT_MODEL", "reappeared-model")
    monkeypatch.setattr(stages, "DISCOVERED_MODEL_IDS", {"reappeared-model"})
    recovered = e.prepare_execution(owner, llm_client=llm)
    with e.execution_context(recovered):
        assert stages._StagesMixin._int_env("QEEG_STAGE1_MAX_TOKENS", 12000) == 321
        assert _load_prompt("stage1_analysis.md") == original
        assert recovered.manifest_hash == old
        assert "reappeared-model" not in recovered.roles["writers"]
        e.bind_source("prepared-report", {"text": "original bytes", "images": []})
        with pytest.raises(ExecutionConflict):
            e.bind_source("prepared-report", {"text": "changed bytes", "images": []})
    with owner.transaction() as session:
        run = session.get(storage.Run, "r")
        assert run.analysis_input_fingerprint == "original-input"
        assert run.source_manifest_json == ' {"source": "original"} '
        assert run.council_model_ids_json == '["gpt-5.5", "gpt-5.5"]'


@pytest.mark.asyncio
@pytest.mark.parametrize("ending", ["cancel", "emit", "deadline"])
async def test_heartbeat_scopes_actual_child_and_drains_before_owner_exit(
    owner, monkeypatch, ending
):
    from backend.council.workflow.stages import _StagesMixin

    e = execution()
    entered = asyncio.Event()
    released = asyncio.Event()
    drained = asyncio.Event()

    async def send(req):
        entered.set()
        try:
            await released.wait()
        except asyncio.CancelledError:
            await released.wait()
        drained.set()
        return httpx.Response(
            200, json={"choices": [{"message": {"content": "acknowledged"}}]}
        )

    llm = client(send)
    monkeypatch.setenv("QEEG_PROGRESS_HEARTBEAT_S", "1")
    ctx = e.prepare_execution(owner, llm_client=llm)

    async def emit(_):
        if ending == "emit":
            raise RuntimeError("emit failed")

    async def run():
        with e.execution_context(ctx):
            return await _StagesMixin()._await_with_heartbeat(
                e.execute_unit(
                    "s1/member/0/gpt-5.5/primary",
                    llm.chat_completions(
                        model_id="gpt-5.5",
                        messages=[{"role": "user", "content": "hello"}],
                    ),
                ),
                emit=emit,
                payload={"task": "test"},
                timeout_s=0.02 if ending == "deadline" else None,
            )

    task = asyncio.create_task(run())
    await asyncio.wait_for(entered.wait(), 2)
    if ending == "cancel":
        task.cancel()
    await asyncio.sleep(1.1 if ending == "emit" else 0.06)
    assert not task.done()
    assert owner.store.claim_run_owner("r") is None
    released.set()
    with pytest.raises(BaseException):
        await task
    assert drained.is_set()


@pytest.mark.asyncio
async def test_gather_drains_and_retains_successful_siblings(owner):
    e = execution()
    done = asyncio.Event()

    async def bad():
        raise PaidOutcomeUnknown(("r", "a", 0))

    async def good():
        await asyncio.sleep(0.03)
        done.set()
        return ("model", "saved")

    with pytest.raises(PaidOutcomeUnknown) as err:
        await e.gather_units(bad(), good())
    assert done.is_set()
    assert err.value.completed_results == [("model", "saved")]


@pytest.mark.asyncio
async def test_sdk_preserves_default_and_borrowed_timeout_and_transport_lifetime():
    from backend.council.ai_review_agents import _provider_for_call

    class Shared(httpx.AsyncBaseTransport):
        closed = False

        async def handle_async_request(self, request):
            return httpx.Response(200, json={})

        async def aclose(self):
            self.closed = True

    transport = Shared()
    llm = AsyncOpenAICompatClient(
        base_url="http://synthetic", api_key="", timeout_s=37, transport=transport
    )
    async with _provider_for_call(llm) as first:
        async with _provider_for_call(llm):
            assert first.client.max_retries == 0
            assert first.client._client.timeout.read == 37
        assert not transport.closed
    assert not transport.closed
    for direct in [
        None,
        AsyncOpenAICompatClient(base_url="http://synthetic", api_key="", timeout_s=37),
    ]:
        async with _provider_for_call(direct) as provider:
            assert provider.client.max_retries == 0
            assert provider.client._client.timeout.read == 600
            assert provider.client._client.timeout.connect == 5


@pytest.mark.asyncio
async def test_owned_low_level_call_requires_semantic_identity(owner):
    from backend.council import QEEGCouncilWorkflow

    e = execution()
    sent = []
    llm = client(
        lambda r: sent.append(r)
        or httpx.Response(200, json={"choices": [{"message": {"content": "oops"}}]})
    )
    ctx = e.prepare_execution(owner, llm_client=llm)
    with e.execution_context(ctx), pytest.raises(ExecutionConflict):
        await QEEGCouncilWorkflow(llm=llm)._call_model_chat(
            model_id="gpt-5.5", prompt_text="missing unit", temperature=0, max_tokens=50
        )
    assert not sent


@pytest.mark.asyncio
async def test_direct_data_pack_source_candidate_drift_blocks_before_new_key(
    owner, tmp_path, monkeypatch
):
    from backend.council import QEEGCouncilWorkflow

    e = execution()
    llm = client(
        lambda r: httpx.Response(
            200, json={"choices": [{"message": {"content": "{}"}}]}
        )
    )
    workflow = QEEGCouncilWorkflow(llm=llm)
    # Both calls have no images and spend nothing. Input binding still prevents
    # recovery from silently adding a new candidate before its first new scope.
    report = storage.Report(
        id="q",
        patient_id="p",
        filename="synthetic.txt",
        stored_path=str(tmp_path / "original.txt"),
        extracted_text_path=str(tmp_path / "extracted.txt"),
        mime_type="text/plain",
    )
    Path(report.stored_path).write_text("original")
    Path(report.extracted_text_path).write_text("original")
    with owner.transaction() as session:
        session.add(report)
    ctx = e.prepare_execution(owner, llm_client=llm)
    with e.execution_context(ctx):
        await workflow._ensure_data_pack(
            run_id="r",
            report=report,
            report_text="original",
            page_images=[],
            candidate_extractor_model_ids=["first"],
            strict=False,
        )
        with pytest.raises(ExecutionConflict):
            await workflow._ensure_data_pack(
                run_id="r",
                report=report,
                report_text="changed",
                page_images=[],
                candidate_extractor_model_ids=["new"],
                strict=False,
            )


@pytest.mark.asyncio
async def test_missing_consumed_source_binding_cannot_be_recreated(owner):
    e = execution()
    llm = client(
        lambda r: httpx.Response(
            200,
            json={
                "output_text": "saved",
                "choices": [{"message": {"content": "saved"}}],
            },
        )
    )
    ctx = e.prepare_execution(owner, llm_client=llm)
    with e.execution_context(ctx):
        e.bind_source("s1/prepared-report", {"text": "original"}, consumers="s1/")
        await e.execute_unit(
            "s1/member/0/gpt-5.5/primary",
            llm.chat_completions(
                model_id="gpt-5.5", messages=[{"role": "user", "content": "original"}]
            ),
        )
        for path in (ctx.manifest_path.parent / "sources").glob("*.json"):
            path.unlink()
        with pytest.raises(ExecutionConflict):
            e.bind_source("s1/prepared-report", {"text": "changed"}, consumers="s1/")


@pytest.mark.asyncio
async def test_stage_argument_identity_cannot_add_a_new_council_member(owner):
    from backend.council import QEEGCouncilWorkflow

    e = execution()
    llm = client(lambda _: None)
    ctx = e.prepare_execution(owner, llm_client=llm)

    async def emit(_):
        pass

    with e.execution_context(ctx), pytest.raises(ExecutionConflict):
        await QEEGCouncilWorkflow(llm=llm)._stage5("r", ["unexpected"], emit)


@pytest.mark.asyncio
async def test_independent_run_contexts_share_workflow_without_sharing_saved_routes(
    owner, monkeypatch
):
    from backend.council import QEEGCouncilWorkflow
    from backend.council import execution as e

    monkeypatch.setenv("OPENROUTER_API_KEY", "synthetic-key")
    monkeypatch.setenv("QEEG_ROUTE_OPENROUTER_EXTRAS_DIRECT", "1")
    monkeypatch.setenv("OPENROUTER_BASE_URL", "http://route-one/api")
    sent = []

    async def send(req):
        sent.append(str(req.url))
        await asyncio.sleep(0.01)
        return httpx.Response(
            200, json={"choices": [{"message": {"content": "answer"}}]}
        )

    llm = client(send)
    ctx1 = e.prepare_execution(owner, llm_client=llm)
    with storage.session_scope() as session:
        session.add(
            storage.Run(
                id="r2",
                patient_id="p",
                report_id="q",
                status="created",
                council_model_ids_json='["z-ai/glm-5.2"]',
                analysis_input_fingerprint="second",
            )
        )
        session.commit()
    owner.store.request_run_start("r2")
    owner2 = owner.store.claim_run_owner("r2")
    monkeypatch.setenv("OPENROUTER_BASE_URL", "http://route-two/api")
    ctx2 = e.prepare_execution(owner2, llm_client=llm)
    workflow = QEEGCouncilWorkflow(llm=llm)

    async def run(ctx):
        with e.execution_context(ctx):
            return await e.execute_unit(
                "s6/writer/0/z-ai/glm-5.2",
                workflow._call_model_chat(
                    model_id="z-ai/glm-5.2",
                    prompt_text="write",
                    temperature=0.2,
                    max_tokens=50,
                ),
            )

    try:
        assert await asyncio.gather(run(ctx1), run(ctx2)) == ["answer", "answer"]
        assert sorted(sent) == [
            "http://route-one/api/v1/chat/completions",
            "http://route-two/api/v1/chat/completions",
        ]
    finally:
        owner2.close()


def seed_stages(owner, tmp_path, monkeypatch, models=("model-a", "model-b")):
    from backend import config
    from backend.council.workflow import stages

    monkeypatch.setattr(config, "ARTIFACTS_DIR", tmp_path / "artifacts")
    monkeypatch.setattr(stages, "DISCOVERED_MODEL_IDS", set(models))
    monkeypatch.setenv("QEEG_STAGE6_FINAL_DRAFT_MODEL", models[0])
    monkeypatch.setenv("QEEG_STAGE6_FINAL_DRAFT_FALLBACK_MODEL", models[-1])
    stored = tmp_path / "original.txt"
    stored.write_text("Source facts.\n")
    extracted = tmp_path / "extracted.txt"
    extracted.write_text("Source facts.\n")
    with storage.session_scope() as session:
        report = storage.Report(
            id="q",
            patient_id="p",
            filename="original.txt",
            stored_path=str(stored),
            extracted_text_path=str(extracted),
            mime_type="text/plain",
        )
        session.add(report)
        run = session.get(storage.Run, "r")
        run.council_model_ids_json = json.dumps(list(models))
        run.consolidator_model_id = models[0]
        for num, kind in [
            (1, "analysis"),
            (2, "peer_review"),
            (3, "revision"),
            (4, "consolidation"),
            (5, "final_review"),
        ]:
            for i, m in enumerate(dict.fromkeys(models)):
                path = tmp_path / f"input-{num}-{i}.txt"
                path.write_text(
                    json.dumps(fixture("stage5_revise_valid.json"))
                    if num == 5
                    else "Prior accepted clinical text."
                )
                storage.create_artifact(
                    session,
                    run_id="r",
                    stage_num=num,
                    stage_name=kind,
                    model_id=m,
                    kind=kind,
                    content_path=path,
                    content_type="text/plain",
                )
        session.commit()
    return report


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "stage,repair",
    [(s, r) for s in range(1, 7) for r in (False, True) if not (r and s in (2, 5))],
)
async def test_every_council_stage_propagates_unknown_before_next_policy(
    owner, tmp_path, monkeypatch, stage, repair
):
    from backend.council import QEEGCouncilWorkflow

    e = execution()
    report = seed_stages(owner, tmp_path, monkeypatch)
    sent = []

    def send(req):
        model = json.loads(req.content)["model"]
        sent.append(model)
        if repair and len(sent) == 1:
            return httpx.Response(
                200,
                json={
                    "choices": [
                        {"message": {"content": "Incomplete accepted response."}}
                    ]
                },
            )
        raise httpx.ReadError("response was lost")

    llm = client(send)
    ctx = e.prepare_execution(owner, llm_client=llm)
    workflow = QEEGCouncilWorkflow(llm=llm)

    async def emit(_):
        pass

    kwargs = (
        ("r", ["model-a", "model-b"], report, emit)
        if stage == 1
        else (("r", emit) if stage == 4 else ("r", ["model-a", "model-b"], emit))
    )
    with e.execution_context(ctx), pytest.raises(PaidOutcomeUnknown):
        await getattr(workflow, f"_stage{stage}")(*kwargs)
    # A synchronous synthetic transport makes the initial unknown authoritative
    # before another member starts; no budget retry, writer fallback, or next repair.
    assert len(sent) == (2 if repair else 1)
    with owner.transaction() as session:
        rows = list(session.scalars(select(storage.PaidRequest)))
    assert any(r.state == "unknown" for r in rows)
    assert all(
        "reduced-budget" not in r.scope_key and "writer/1/" not in r.scope_key
        for r in rows
    )


@pytest.mark.asyncio
async def test_stage1_interleaved_late_sibling_is_saved_before_unknown_returns(
    owner, tmp_path, monkeypatch
):
    from backend.council import QEEGCouncilWorkflow

    e = execution()
    report = seed_stages(owner, tmp_path, monkeypatch)
    second = asyncio.Event()
    sent = []

    async def send(req):
        model = json.loads(req.content)["model"]
        sent.append(model)
        if model == "model-a":
            await second.wait()
            raise httpx.ReadError("lost")
        second.set()
        await asyncio.sleep(0.05)
        return httpx.Response(
            200,
            json={
                "choices": [
                    {"message": {"content": "Complete\n<!-- END STAGE1 ANALYSIS -->"}}
                ]
            },
        )

    llm = client(send)
    ctx = e.prepare_execution(owner, llm_client=llm)

    async def emit(_):
        pass

    with e.execution_context(ctx), pytest.raises(PaidOutcomeUnknown) as error:
        await QEEGCouncilWorkflow(llm=llm)._stage1(
            "r", ["model-a", "model-b"], report, emit
        )
    assert error.value.completed_results == [
        ("model-b", "Complete\n<!-- END STAGE1 ANALYSIS -->")
    ]
    with owner.transaction() as session:
        rows = list(session.scalars(select(storage.PaidRequest)))
    assert sorted(r.state for r in rows) == ["response_saved", "unknown"]
    assert len(sent) == 2


@pytest.mark.asyncio
async def test_duplicate_council_ids_replay_distinct_ordered_member_scopes(
    owner, tmp_path, monkeypatch
):
    from backend.council import QEEGCouncilWorkflow

    e = execution()
    report = seed_stages(owner, tmp_path, monkeypatch, models=("model-a", "model-a"))
    sent = []

    async def send(req):
        sent.append(req.content)
        await asyncio.sleep(0.01)
        return httpx.Response(
            200,
            json={
                "choices": [
                    {"message": {"content": "Complete\n<!-- END STAGE1 ANALYSIS -->"}}
                ]
            },
        )

    llm = client(send)
    ctx = e.prepare_execution(owner, llm_client=llm)

    async def emit(_):
        pass

    for _ in range(2):
        with e.execution_context(ctx):
            await QEEGCouncilWorkflow(llm=llm)._stage1(
                "r", ["model-a", "model-a"], report, emit
            )
    assert len(sent) == 2
    with owner.transaction() as session:
        keys = {r.scope_key for r in session.scalars(select(storage.PaidRequest))}
    assert keys == {"s1/member/0/model-a/primary", "s1/member/1/model-a/primary"}


@pytest.mark.asyncio
@pytest.mark.parametrize("stage", [2, 5])
async def test_sdk_eligible_endpoint_fallback_replays_shared_request_sequence(
    owner, stage
):
    from backend.council import ai_review_agents as agents

    e = execution()
    sent = []

    def send(req):
        sent.append(str(req.url))
        if req.url.path.endswith("/chat/completions"):
            return httpx.Response(
                400,
                json={
                    "error": {
                        "message": "chat completions not supported; use responses"
                    }
                },
            )
        body = json.loads(req.content)
        tool = body["tools"][0]["name"]
        value = fixture(
            "stage2_valid.json" if stage == 2 else "stage5_revise_valid.json"
        )
        return httpx.Response(
            200,
            json={
                "id": "response-stable",
                "created_at": 1,
                "object": "response",
                "model": body["model"],
                "output": [
                    {
                        "type": "function_call",
                        "id": "fc-stable",
                        "call_id": "call-stable",
                        "name": tool,
                        "arguments": json.dumps(value),
                    }
                ],
            },
        )

    llm = client(send)
    ctx = e.prepare_execution(owner, llm_client=llm)
    fn = agents.run_stage2_peer_review if stage == 2 else agents.run_stage5_final_review
    for _ in range(2):
        with e.execution_context(ctx):
            await e.execute_unit(
                f"s{stage}/reviewer/0/gpt-5.5",
                fn(
                    llm_client=llm,
                    model_id="gpt-5.5",
                    prompt_text="review",
                    **({"expected_labels": ["A", "B"]} if stage == 2 else {}),
                ),
            )
    assert len(sent) == 2
    with owner.transaction() as session:
        rows = list(
            session.scalars(
                select(storage.PaidRequest).order_by(storage.PaidRequest.scope_key)
            )
        )
    assert [r.state for r in rows] == ["rejected", "response_saved"]
    assert [r.scope_key.rsplit("/", 1)[-1] for r in rows] == ["0", "1"]


@pytest.mark.asyncio
async def test_direct_none_sdk_client_uses_same_paid_boundary(owner, monkeypatch):
    from backend.council import ai_review_agents as agents
    from backend.paid_transport import PaidAsyncClient

    e = execution()
    sent = []

    def send(req):
        sent.append(req.content)
        return completion(req, fixture("stage5_approve_valid.json"))

    monkeypatch.setattr(
        agents,
        "PaidAsyncClient",
        lambda **kw: PaidAsyncClient(**{**kw, "transport": httpx.MockTransport(send)}),
    )
    ctx = e.prepare_execution(owner)
    for _ in range(2):
        with e.execution_context(ctx):
            await e.execute_unit(
                "s5/reviewer/0/gpt-5.5",
                agents.run_stage5_final_review(
                    llm_client=None, model_id="gpt-5.5", prompt_text="review"
                ),
            )
    assert len(sent) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "unit,strict,unknown_at",
    [
        ("data-pack", False, 1),
        ("data-pack", True, 1),
        ("p300", False, 2),
        ("p300", True, 2),
        ("summary", False, 3),
        ("summary", True, 3),
        ("transcript", False, 1),
        ("transcript", True, 1),
    ],
)
async def test_extraction_repairs_and_transcript_never_advance_after_unknown(
    owner, tmp_path, monkeypatch, unit, strict, unknown_at
):
    from backend.council import QEEGCouncilWorkflow
    from backend.council.types import PageImage
    from backend.council.constants import DATA_PACK_SCHEMA_VERSION
    from backend import config

    e = execution()
    monkeypatch.setattr(config, "ARTIFACTS_DIR", tmp_path / "artifacts")
    monkeypatch.setenv("QEEG_VISION_TRANSCRIPT_PAGES_PER_CALL", "1")
    sent = []

    def send(req):
        sent.append(req.content)
        if len(sent) == unknown_at:
            raise httpx.ReadError("response lost")
        pack = {
            "schema_version": DATA_PACK_SCHEMA_VERSION,
            "facts": [],
            "pages_seen": [1, 2],
            "page_inventory": [],
        }
        return httpx.Response(
            200, json={"choices": [{"message": {"content": json.dumps(pack)}}]}
        )

    llm = client(send)
    workflow = QEEGCouncilWorkflow(llm=llm)
    report = storage.Report(
        id="q",
        patient_id="p",
        filename="synthetic.txt",
        stored_path=str(tmp_path / "synthetic.txt"),
        extracted_text_path=str(tmp_path / "extracted.txt"),
        mime_type="text/plain",
    )
    Path(report.stored_path).write_text("Session 1")
    Path(report.extracted_text_path).write_text("Session 1")
    (tmp_path / "pages").mkdir()
    for page in (1, 2):
        (tmp_path / "pages" / f"page-{page}.png").write_bytes(b"x")
    with owner.transaction() as session:
        session.add(report)
    ctx = e.prepare_execution(owner, llm_client=llm)
    images = [PageImage(page=n, base64_png="eA==") for n in (1, 2)]
    with e.execution_context(ctx), pytest.raises(PaidOutcomeUnknown):
        if unit == "transcript":
            await workflow._ensure_vision_transcript(
                run_id="r",
                report=report,
                page_images=images,
                transcript_model_id="vision-a",
                strict=strict,
            )
        else:
            await workflow._ensure_data_pack(
                run_id="r",
                report=report,
                report_text="Session 1",
                page_images=images,
                candidate_extractor_model_ids=["vision-a", "vision-b"],
                strict=strict,
            )
    assert len(sent) == unknown_at
    with owner.transaction() as session:
        rows = list(session.scalars(select(storage.PaidRequest)))
    assert sum(r.state == "unknown" for r in rows) == 1
    assert all("vision-b" not in r.scope_key for r in rows)


@pytest.mark.asyncio
async def test_per_member_vision_notes_unknown_does_not_start_primary(
    owner, tmp_path, monkeypatch
):
    from backend.council import QEEGCouncilWorkflow
    from backend.council.workflow import stages
    from backend.council.types import PageImage

    e = execution()
    report = seed_stages(owner, tmp_path, monkeypatch, models=("vision-a",))
    monkeypatch.setattr(stages, "is_vision_capable", lambda _: True)
    monkeypatch.setattr(
        stages, "_load_page_images", lambda *_: [PageImage(page=1, base64_png="eA==")]
    )

    class NotesOnly(QEEGCouncilWorkflow):
        async def _ensure_data_pack(self, **kw):
            return None

        async def _ensure_vision_transcript(self, **kw):
            return None

    sent = []

    def send(req):
        sent.append(req.content)
        raise httpx.ReadError("lost note")

    llm = client(send)
    ctx = e.prepare_execution(owner, llm_client=llm)

    async def emit(_):
        pass

    with e.execution_context(ctx), pytest.raises(PaidOutcomeUnknown):
        await NotesOnly(llm=llm)._stage1("r", ["vision-a"], report, emit)
    assert len(sent) == 1
    with owner.transaction() as session:
        rows = list(session.scalars(select(storage.PaidRequest)))
    assert rows[0].scope_key == "s1/member/0/vision-a/vision-notes/chunk/1"


@pytest.mark.asyncio
@pytest.mark.parametrize("drift", ["original", "extracted"])
async def test_original_admitted_source_drift_blocks_before_owned_dispatch(
    owner, tmp_path, monkeypatch, drift
):
    from backend.analysis_inputs import _source_snapshot

    e = execution()
    report = seed_stages(owner, tmp_path, monkeypatch)
    snapshot, _ = _source_snapshot(report)
    with owner.transaction() as session:
        run = session.get(storage.Run, "r")
        run.source_manifest_json = json.dumps({"legacy": False, "sources": [snapshot]})
    sent = []
    llm = client(lambda req: sent.append(req.content))
    ctx = e.prepare_execution(owner, llm_client=llm)
    Path(
        report.stored_path if drift == "original" else report.extracted_text_path
    ).write_text("Changed admitted source")
    with pytest.raises(ExecutionConflict):
        with e.execution_context(ctx):
            await e.execute_unit(
                "s1/member/0/model-a/primary",
                llm.chat_completions(model_id="model-a", messages=[]),
            )
    assert not sent


def test_manifest_file_missing_is_typed_conflict(owner):
    e = execution()
    ctx = e.prepare_execution(owner)
    ctx.manifest_path.unlink()
    with pytest.raises(ExecutionConflict):
        ctx.verify()
    with pytest.raises(ExecutionConflict):
        e.prepare_execution(owner)


@pytest.mark.asyncio
async def test_sdk_sticky_unknown_survives_failed_metadata_write(owner, monkeypatch):
    from contextlib import contextmanager
    from backend.council import ai_review_agents as agents

    e = execution()
    original = owner.transaction
    sent = []

    @contextmanager
    def broken():
        raise OSError("metadata store unavailable")
        yield

    def send(req):
        sent.append(req.content)
        monkeypatch.setattr(owner, "transaction", broken)
        raise httpx.ReadError("lost response")

    llm = client(send)
    ctx = e.prepare_execution(owner, llm_client=llm)
    try:
        with e.execution_context(ctx), pytest.raises(PaidOutcomeUnknown):
            await e.execute_unit(
                "s5/reviewer/0/gpt-5.5",
                agents.run_stage5_final_review(
                    llm_client=llm, model_id="gpt-5.5", prompt_text="Review."
                ),
            )
    finally:
        monkeypatch.setattr(owner, "transaction", original)
    assert len(sent) == 1
    with owner.transaction() as session:
        row = session.scalar(select(storage.PaidRequest))
    assert row.state == "dispatched"


@pytest.mark.asyncio
async def test_sdk_deliberate_backoff_keeps_request_sequence_across_agent_runs(
    owner, monkeypatch
):
    from backend.council import ai_review_agents as agents
    from pydantic_ai.exceptions import ModelAPIError

    e = execution()
    real_agent = agents._STAGE5_REVIEW_AGENT

    class LocalDecodeFailure:
        count = 0

        async def run(self, *args, **kwargs):
            result = await real_agent.run(*args, **kwargs)
            self.count += 1
            if self.count % 2:
                raise ModelAPIError(
                    "gpt-5.5", "local acknowledged-output interpretation failure"
                )
            return result

    async def no_wait(_):
        pass

    monkeypatch.setattr(agents, "_STAGE5_REVIEW_AGENT", LocalDecodeFailure())
    monkeypatch.setattr(agents, "_sleep_backoff", no_wait)
    sent = []

    def send(req):
        sent.append(req.content)
        return completion(req, fixture("stage5_approve_valid.json"))

    llm = client(send)
    ctx = e.prepare_execution(owner, llm_client=llm)
    for _ in range(2):
        with e.execution_context(ctx):
            await e.execute_unit(
                "s5/reviewer/0/gpt-5.5",
                agents.run_stage5_final_review(
                    llm_client=llm, model_id="gpt-5.5", prompt_text="Review."
                ),
            )
    assert len(sent) == 2
    with owner.transaction() as session:
        keys = {r.scope_key for r in session.scalars(select(storage.PaidRequest))}
    assert keys == {
        "s5/reviewer/0/gpt-5.5/sdk-request/0",
        "s5/reviewer/0/gpt-5.5/sdk-request/1",
    }


def test_saved_prompt_bytes_survive_live_prompt_and_recipe_drift_blocks(
    owner, monkeypatch
):
    from backend.council.prompts import _load_prompt

    e = execution()
    ctx = e.prepare_execution(owner)
    original = ctx.manifest["prompts"]["stage1_analysis.md"]
    read = Path.read_bytes

    def drift(path):
        return (
            b"Changed prompt content"
            if path.name == "stage1_analysis.md"
            else read(path)
        )

    monkeypatch.setattr(Path, "read_bytes", drift)
    recovered = e.prepare_execution(owner)
    with e.execution_context(recovered):
        assert _load_prompt("stage1_analysis.md") == original
    recipe = e._recipe()
    monkeypatch.setattr(e, "_recipe", lambda: {**recipe, "incompatible": True})
    with pytest.raises(ExecutionConflict):
        e.prepare_execution(owner)


@pytest.mark.asyncio
async def test_cancelled_gather_retains_child_unknown_authority(owner):
    e = execution()
    entered = asyncio.Event()

    async def send(req):
        entered.set()
        await asyncio.Event().wait()

    llm = client(send)
    ctx = e.prepare_execution(owner, llm_client=llm)

    async def run():
        with e.execution_context(ctx):
            await e.gather_units(
                e.execute_unit(
                    "s1/member/0/model-a/primary",
                    llm.chat_completions(
                        model_id="model-a",
                        messages=[{"role": "user", "content": "hello"}],
                    ),
                )
            )

    task = asyncio.create_task(run())
    await entered.wait()
    task.cancel()
    with pytest.raises(PaidOutcomeUnknown):
        await task


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "configured,environment,expected",
    [
        ("cliproxy-key", "openai-key", "cliproxy-key"),
        ("", "openai-key", "openai-key"),
        ("", "", ""),
    ],
)
async def test_sdk_preserves_existing_effective_auth_key_priority(
    monkeypatch, configured, environment, expected
):
    from backend.council import ai_review_agents as agents

    monkeypatch.setattr(agents, "CLIPROXY_API_KEY", configured)
    monkeypatch.setenv("OPENAI_API_KEY", environment)
    async with agents._provider_for_call(None) as provider:
        assert provider.client.api_key == expected


@pytest.mark.asyncio
async def test_nested_execution_context_cannot_inherit_another_unit_cursor(owner):
    from backend.council import QEEGCouncilWorkflow

    e = execution()
    sent = []
    llm = client(
        lambda r: sent.append(r.content)
        or httpx.Response(
            200, json={"choices": [{"message": {"content": "wrong scope"}}]}
        )
    )
    ctx = e.prepare_execution(owner, llm_client=llm)

    async def inner():
        # A new execution context must require explicit semantic entry even when
        # its caller is currently inside a different unit.
        with e.execution_context(ctx):
            await QEEGCouncilWorkflow(llm=llm)._call_model_chat(
                model_id="model-a", prompt_text="inner", temperature=0, max_tokens=50
            )

    with e.execution_context(ctx), pytest.raises(ExecutionConflict):
        await e.execute_unit("outer", inner())
    assert not sent


def bank_original_report(owner, report):
    from backend.analysis_inputs import _source_snapshot

    snapshot, _ = _source_snapshot(report)
    with owner.transaction() as session:
        run = session.get(storage.Run, owner.run_id)
        run.source_manifest_json = json.dumps(
            {
                "legacy": False,
                "execution_report_id": report.id,
                "source_report_ids": [report.id],
                "sources": [snapshot],
            }
        )


def supplied_report_copy(report, **changes):
    fields = {
        name: getattr(report, name)
        for name in (
            "id",
            "patient_id",
            "filename",
            "mime_type",
            "stored_path",
            "extracted_text_path",
        )
    }
    return storage.Report(**{**fields, **changes})


@pytest.mark.asyncio
@pytest.mark.parametrize("entry", ["stage1", "data_pack", "transcript"])
@pytest.mark.parametrize(
    "mismatch",
    [
        "report",
        "patient",
        "stored_path",
        "extracted_text_path",
        "filename",
        "mime_type",
        "stored_bytes",
        "text_bytes",
    ],
)
async def test_owned_report_mismatch_family_rejects_before_binding_or_sending(
    owner, tmp_path, monkeypatch, entry, mismatch
):
    from backend.council import QEEGCouncilWorkflow

    e = execution()
    report = seed_stages(owner, tmp_path, monkeypatch, models=("model-a",))
    bank_original_report(owner, report)
    wrong = tmp_path / "other-source.txt"
    wrong.write_text("OTHER PATIENT SOURCE FACTS")
    changes = {
        "report": {"id": "other-report"},
        "patient": {"patient_id": "other-patient"},
        "stored_path": {"stored_path": str(wrong)},
        "extracted_text_path": {"extracted_text_path": str(wrong)},
        "filename": {"filename": "other-source.txt"},
        "mime_type": {"mime_type": "application/pdf"},
        "stored_bytes": {},
        "text_bytes": {},
    }[mismatch]
    supplied = supplied_report_copy(report, **changes)
    sent = []

    def send(request):
        sent.append(request.content)
        return httpx.Response(
            200,
            json={
                "choices": [
                    {"message": {"content": "Complete\n<!-- END STAGE1 ANALYSIS -->"}}
                ]
            },
        )

    llm = client(send)
    context = e.prepare_execution(owner, llm_client=llm)
    if mismatch in {"stored_bytes", "text_bytes"}:
        path = (
            report.stored_path
            if mismatch == "stored_bytes"
            else report.extracted_text_path
        )
        Path(path).write_text("OTHER PATIENT SOURCE FACTS")
    workflow = QEEGCouncilWorkflow(llm=llm)

    async def emit(_):
        pass

    with pytest.raises(ExecutionConflict), e.execution_context(context):
        if entry == "stage1":
            await workflow._stage1("r", ["model-a"], supplied, emit)
        elif entry == "data_pack":
            await workflow._ensure_data_pack(
                run_id="r",
                report=supplied,
                report_text="Source facts.\n",
                page_images=[],
                candidate_extractor_model_ids=["model-a"],
                strict=False,
            )
        else:
            await workflow._ensure_vision_transcript(
                run_id="r",
                report=supplied,
                page_images=[],
                transcript_model_id="model-a",
                strict=False,
            )
    assert sent == []
    assert not list((context.manifest_path.parent / "sources").glob("*.json"))


@pytest.mark.asyncio
@pytest.mark.parametrize("entry", ["data_pack", "transcript"])
@pytest.mark.parametrize(
    "mismatch", ["run", "images", "image_order", "image_labels", "text"]
)
async def test_owned_direct_extraction_rejects_substituted_payload_before_binding(
    owner, tmp_path, monkeypatch, entry, mismatch
):
    if entry == "transcript" and mismatch == "text":
        # Transcript has no caller-supplied text; use a missing canonical report.
        mismatch = "missing_report"
    from backend.council import QEEGCouncilWorkflow
    from backend.council.report_assets import _load_page_images
    from backend.council.types import PageImage

    e = execution()
    report = seed_stages(owner, tmp_path, monkeypatch, models=("model-a",))
    pages = tmp_path / "pages"
    pages.mkdir()
    (pages / "page-1.png").write_bytes(b"first canonical image")
    (pages / "page-2.png").write_bytes(b"second canonical image")
    bank_original_report(owner, report)
    images = _load_page_images(report, tmp_path)
    if mismatch == "images":
        images = [PageImage(page=1, base64_png="b3RoZXI=")]
    if mismatch == "image_order":
        images = list(reversed(images))
    if mismatch == "image_labels":
        images = [
            PageImage(page=p.page, base64_png=p.base64_png, label="wrong")
            for p in images
        ]
    text = "OTHER PATIENT SOURCE FACTS" if mismatch == "text" else "Source facts.\n"
    sent = []
    llm = client(
        lambda request: sent.append(request.content)
        or httpx.Response(200, json={"choices": [{"message": {"content": "{}"}}]})
    )
    context = e.prepare_execution(owner, llm_client=llm)
    if mismatch == "missing_report":
        with storage.session_scope() as session:
            session.execute(
                storage.Report.__table__.delete().where(storage.Report.id == "q")
            )
            session.commit()
    workflow = QEEGCouncilWorkflow(llm=llm)
    with pytest.raises(ExecutionConflict), e.execution_context(context):
        kw = dict(
            run_id="another-run" if mismatch == "run" else "r",
            report=report,
            page_images=images,
            strict=False,
        )
        if entry == "data_pack":
            await workflow._ensure_data_pack(
                **kw, report_text=text, candidate_extractor_model_ids=["model-a"]
            )
        else:
            await workflow._ensure_vision_transcript(
                **kw, transcript_model_id="model-a"
            )
    assert sent == []
    assert not list((context.manifest_path.parent / "sources").glob("*.json"))


@pytest.mark.asyncio
@pytest.mark.parametrize("new_default", ["", "   ", "reappeared-writer"])
async def test_owned_writer_recovery_uses_saved_preference_before_current_default(
    owner, tmp_path, monkeypatch, new_default
):
    from dataclasses import replace
    from backend.council import QEEGCouncilWorkflow
    from backend.council.workflow import stages
    from backend.tests.test_stage6_final_draft_repair import _complete_stage6

    e = execution()
    seed_stages(owner, tmp_path, monkeypatch, models=("model-a",))
    monkeypatch.delenv("QEEG_STAGE6_FINAL_DRAFT_MODEL")
    monkeypatch.setattr(
        stages,
        "MODEL_ROLE_DEFAULTS",
        replace(stages.MODEL_ROLE_DEFAULTS, stage6_final_draft="model-a"),
    )
    sent = []
    llm = client(
        lambda request: sent.append(request.content)
        or httpx.Response(
            200, json={"choices": [{"message": {"content": _complete_stage6()}}]}
        )
    )
    context = e.prepare_execution(owner, llm_client=llm)

    async def emit(_):
        pass

    with e.execution_context(context):
        await QEEGCouncilWorkflow(llm=llm)._stage6("r", ["model-a"], emit)
    monkeypatch.setattr(
        stages,
        "MODEL_ROLE_DEFAULTS",
        replace(stages.MODEL_ROLE_DEFAULTS, stage6_final_draft=new_default),
    )
    recovered = e.prepare_execution(owner, llm_client=llm)
    with e.execution_context(recovered):
        await QEEGCouncilWorkflow(llm=llm)._stage6("r", ["model-a"], emit)
    assert len(sent) == 1
    assert json.loads(sent[0])["model"] == "model-a"


@pytest.mark.asyncio
@pytest.mark.parametrize("repair", [False, True])
async def test_owned_combined_report_uses_canonical_composition_and_repairs_without_resend(
    owner, tmp_path, monkeypatch, repair
):
    from backend import analysis_inputs, config, reports
    from backend.council import QEEGCouncilWorkflow
    from backend.council.workflow import stages
    from backend.tests.test_analysis_inputs import source

    monkeypatch.setattr(reports, "REPORTS_DIR", tmp_path / "reports")
    monkeypatch.setattr(config, "ARTIFACTS_DIR", tmp_path / "artifacts")
    monkeypatch.setattr(stages, "DISCOVERED_MODEL_IDS", {"mock-council-a"})
    with owner.transaction() as session:
        session.add(storage.Patient(id="p", label="ZZ_01-01-1900"))
    originals = [
        source(tmp_path, "p", date=f"2026-0{n}-02", value=270 + n) for n in (1, 2)
    ]
    models = dict(
        council_model_ids=["mock-council-a"],
        consolidator_model_id="mock-council-a",
        requested_model_ids=["mock-council-a"],
        resolved_model_ids=["mock-council-a"],
    )
    run = analysis_inputs.admit_run(
        patient_id="p",
        source_ids=[r.id for r in originals],
        special_instructions="",
        source_session_aliases={},
        operation_id=None,
        model_fields=lambda: models,
        immutable_request={"source_ids": [r.id for r in originals]},
    )
    store = ExecutionStore(storage.engine)
    store.request_run_start(run.id)
    combined_owner = store.claim_run_owner(run.id)
    with combined_owner.transaction() as session:
        report = session.get(storage.Report, run.report_id)
    assert report.id not in [r.id for r in originals]
    directory = Path(report.stored_path).parent
    original_text = Path(report.extracted_text_path).read_text()
    sent = []
    llm = client(
        lambda request: sent.append(request.content)
        or httpx.Response(
            200,
            json={
                "choices": [
                    {"message": {"content": "Complete\n<!-- END STAGE1 ANALYSIS -->"}}
                ]
            },
        )
    )
    e = execution()
    context = e.prepare_execution(combined_owner, llm_client=llm)

    async def emit(_):
        pass

    try:
        # Original component reports and forged combined IDs cannot substitute
        # for the execution report reserved by real multi-report admission.
        for supplied in [
            originals[0],
            supplied_report_copy(
                report,
                stored_path=originals[0].stored_path,
                extracted_text_path=originals[0].extracted_text_path,
            ),
        ]:
            with e.execution_context(context), pytest.raises(ExecutionConflict):
                await QEEGCouncilWorkflow(llm=llm)._stage1(
                    run.id, ["mock-council-a"], supplied, emit
                )
        assert sent == []
        assert not list((context.manifest_path.parent / "sources").glob("*.json"))
        # Direct extraction also resolves derived combined assets from the
        # banked originals, even if a caller loaded substituted on-disk content.
        from backend.council.report_assets import _load_page_images

        for entry in ("data_pack", "transcript"):
            (directory / "extracted_enhanced.txt").write_text(
                "OTHER PATIENT SOURCE FACTS"
            )
            (directory / "pages" / "page-1.png").write_bytes(b"substituted image")
            supplied_images = _load_page_images(report, directory)
            with e.execution_context(context), pytest.raises(ExecutionConflict):
                workflow = QEEGCouncilWorkflow(llm=llm)
                if entry == "data_pack":
                    await workflow._ensure_data_pack(
                        run_id=run.id,
                        report=report,
                        report_text="OTHER PATIENT SOURCE FACTS",
                        page_images=supplied_images,
                        candidate_extractor_model_ids=["mock-council-a"],
                        strict=False,
                    )
                else:
                    await workflow._ensure_vision_transcript(
                        run_id=run.id,
                        report=report,
                        page_images=supplied_images,
                        transcript_model_id="mock-council-a",
                        strict=False,
                    )
            assert sent == []
            assert not list((context.manifest_path.parent / "sources").glob("*.json"))
            assert (directory / "extracted_enhanced.txt").read_text() == original_text
        for attempt in range(2):
            if repair:
                # Restore from banked originals before both first use and replay.
                (directory / "extracted_enhanced.txt").unlink()
                (directory / "pages" / "page-1.png").unlink()
            recovered = (
                context
                if attempt == 0
                else e.prepare_execution(combined_owner, llm_client=llm)
            )
            with e.execution_context(recovered):
                await QEEGCouncilWorkflow(llm=llm)._stage1(
                    run.id, ["mock-council-a"], report, emit
                )
            assert Path(report.extracted_text_path).read_text() == original_text
            assert (directory / "pages" / "page-1.png").exists()
        assert len(sent) == 1
        assert all(str(value).encode() in sent[0] for value in (271, 272))
        from backend.council.report_text import _expected_session_indices

        assert _expected_session_indices(original_text) == [1, 2]
    finally:
        await context.aclose()
        combined_owner.close()
