"""Acknowledged council units and stage receipts, synthetic provider traffic only."""

import asyncio
import importlib
import json
from pathlib import Path

import httpx
import pytest
from sqlalchemy import select

from backend import storage
from backend.council import QEEGCouncilWorkflow
from backend.council import execution as e
from backend.paid_transport import PaidOutcomeUnknown
from backend.run_execution import ExecutionConflict
from backend.tests.test_council_execution import (
    owner as owner_fixture,
    seed_stages,
    client,
)


@pytest.fixture
def owner(tmp_path):
    yield from owner_fixture.__wrapped__(tmp_path)


def completion():
    return importlib.import_module("backend.council.completion")


async def silent(_):
    pass


def answer(text):
    return httpx.Response(200, json={"choices": [{"message": {"content": text}}]})


@pytest.mark.asyncio
@pytest.mark.parametrize("interruption", ["unknown", "cancel"])
async def test_acknowledged_member_persists_before_sibling_finishes(
    owner, tmp_path, monkeypatch, interruption
):
    report = seed_stages(owner, tmp_path, monkeypatch)
    started = asyncio.Event()
    release = asyncio.Event()
    sent = []

    async def send(req):
        model = json.loads(req.content)["model"]
        sent.append(model)
        if model == "model-b":
            started.set()
            await release.wait()
            raise httpx.ReadError("lost")
        await started.wait()
        return answer("Complete\n<!-- END STAGE1 ANALYSIS -->")

    llm = client(send)
    ctx = e.prepare_execution(owner, llm_client=llm)
    with e.execution_context(ctx):
        task = asyncio.create_task(
            QEEGCouncilWorkflow(llm=llm)._stage1(
                "r", ["model-a", "model-b"], report, silent
            )
        )
        await started.wait()
        for _ in range(30):
            await asyncio.sleep(0.01)
            with owner.transaction() as session:
                saved = list(
                    session.scalars(
                        select(storage.Artifact).where(
                            storage.Artifact.run_id == "r",
                            storage.Artifact.stage_num == 1,
                            storage.Artifact.content_path.contains("/artifacts/"),
                        )
                    )
                )
            if saved:
                break
        try:
            assert len(saved) == 1, "successful member remains only in gather memory"
            assert (
                Path(saved[0].content_path)
                .read_text()
                .endswith("<!-- END STAGE1 ANALYSIS -->")
            )
        finally:
            if interruption == "cancel":
                task.cancel()
            release.set()
            with pytest.raises((PaidOutcomeUnknown, asyncio.CancelledError)):
                await task
    assert len(sent) == 2


@pytest.mark.asyncio
async def test_receipt_precedes_progress_and_replays_without_dispatch(
    owner, tmp_path, monkeypatch
):
    report = seed_stages(owner, tmp_path, monkeypatch, models=("model-a", "model-a"))
    sent = []

    def send(req):
        sent.append(req.content)
        return answer(f"Output {len(sent)}\n<!-- END STAGE1 ANALYSIS -->")

    llm = client(send)
    ctx = e.prepare_execution(owner, llm_client=llm)

    async def broken_emit(event):
        if (
            event.get("stage_num") == 1
            and event.get("status") == "complete"
            and not event.get("task")
        ):
            with owner.transaction() as session:
                assert session.get(storage.StageReceipt, ("r", 1)) is not None
            raise RuntimeError("projection interrupted")

    with e.execution_context(ctx):
        with pytest.raises(RuntimeError, match="projection interrupted"):
            await QEEGCouncilWorkflow(llm=llm)._stage1(
                "r", ["model-a", "model-a"], report, broken_emit
            )
        events = []

        async def emit(event):
            events.append(event)

        await QEEGCouncilWorkflow(llm=llm)._stage1(
            "r", ["model-a", "model-a"], report, emit
        )
    assert len(sent) == 2
    assert events[-1]["success_count"] == events[-1]["requested_count"] == 2
    with owner.transaction() as session:
        rows = list(
            session.scalars(
                select(storage.Artifact).where(
                    storage.Artifact.operation_key.is_not(None)
                )
            )
        )
    assert len(rows) == len({a.content_path for a in rows}) == 2
    assert len({Path(a.content_path).read_text() for a in rows}) == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("damage", ["missing", "changed"])
async def test_completed_stage_rejects_damaged_artifact(
    owner, tmp_path, monkeypatch, damage
):
    report = seed_stages(owner, tmp_path, monkeypatch)
    sent = []

    def send(req):
        sent.append(req.content)
        return answer("Complete\n<!-- END STAGE1 ANALYSIS -->")

    llm = client(send)
    ctx = e.prepare_execution(owner, llm_client=llm)
    with e.execution_context(ctx):
        await QEEGCouncilWorkflow(llm=llm)._stage1(
            "r", ["model-a", "model-b"], report, silent
        )
        with owner.transaction() as session:
            row = session.scalar(
                select(storage.Artifact).where(
                    storage.Artifact.operation_key.is_not(None)
                )
            )
        path = Path(row.content_path)
        if damage == "missing":
            path.unlink()
        else:
            path.write_text("changed")
        with pytest.raises(ExecutionConflict):
            completion().verified_stage_prefix()
    assert len(sent) == 2


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "boundary", ["artifact_file", "artifact_row", "member", "stage_file"]
)
async def test_reconstructs_acknowledged_crash_boundaries_without_resend(
    owner, tmp_path, monkeypatch, boundary
):
    c = completion()
    report = seed_stages(owner, tmp_path, monkeypatch, models=("model-a",))
    sent = []
    llm = client(
        lambda req: sent.append(req.content)
        or answer("Complete\n<!-- END STAGE1 ANALYSIS -->")
    )
    ctx = e.prepare_execution(owner, llm_client=llm)
    publish = c._publish
    reconcile = c._reconcile_artifact
    fired = False

    def fail_publish(owner_arg, path, data):
        nonlocal fired
        publish(owner_arg, path, data)
        target = {
            "artifact_file": "/artifacts/",
            "member": "member/",
            "stage_file": "stage/1",
        }
        if boundary == "artifact_file":
            match = target[boundary] in str(path)
        elif boundary in target:
            try:
                match = (
                    json.loads(data)
                    .get("binding", {})
                    .get("key", "")
                    .startswith(target[boundary])
                )
            except ValueError:
                match = False
        else:
            match = False
        if match and not fired:
            fired = True
            raise ExecutionConflict("synthetic crash after publication")

    def fail_reconcile(record, **kwargs):
        nonlocal fired
        row = reconcile(record, **kwargs)
        if boundary == "artifact_row" and not fired:
            fired = True
            raise ExecutionConflict("synthetic crash after registration")
        return row

    with e.execution_context(ctx):
        monkeypatch.setattr(c, "_publish", fail_publish)
        monkeypatch.setattr(c, "_reconcile_artifact", fail_reconcile)
        with pytest.raises(ExecutionConflict, match="synthetic crash"):
            await QEEGCouncilWorkflow(llm=llm)._stage1("r", ["model-a"], report, silent)
        monkeypatch.setattr(c, "_publish", publish)
        monkeypatch.setattr(c, "_reconcile_artifact", reconcile)
        assert fired
        await QEEGCouncilWorkflow(llm=llm)._stage1("r", ["model-a"], report, silent)
        assert c.verified_stage_prefix() == 1
    assert len(sent) == 1
    with owner.transaction() as session:
        assert (
            len(
                list(
                    session.scalars(
                        select(storage.Artifact).where(
                            storage.Artifact.operation_key.is_not(None)
                        )
                    )
                )
            )
            == 1
        )


@pytest.mark.asyncio
async def test_definite_failures_and_successes_rejoin_unfinished_stage(
    owner, tmp_path, monkeypatch
):
    from backend.council.workflow import stages
    from backend.tests.test_council_execution import fixture

    c = completion()
    seed_stages(owner, tmp_path, monkeypatch)
    sent = []
    invoked = []
    llm = client(lambda req: sent.append(req.content) or answer("acknowledged"))

    async def reviewer(*, model_id, **kwargs):
        invoked.append(model_id)
        await llm.chat_completions(
            model_id=model_id, messages=[{"role": "user", "content": "review"}]
        )
        if model_id == "model-a":
            raise ValueError("definite invalid exhausted output")
        return json.dumps(fixture("stage5_approve_valid.json"))

    monkeypatch.setattr(stages, "run_stage5_final_review_json", reviewer)
    ctx = e.prepare_execution(owner, llm_client=llm)
    commit = c._commit_stage

    def crash(*args):
        raise ExecutionConflict("before stage receipt")

    with e.execution_context(ctx):
        monkeypatch.setattr(c, "_commit_stage", crash)
        with pytest.raises(ExecutionConflict):
            await QEEGCouncilWorkflow(llm=llm)._stage5(
                "r", ["model-a", "model-b"], silent
            )
        assert c.verified_stage_prefix() == 0  # Stage 5 rows cannot imply stages 1-4.
        monkeypatch.setattr(c, "_commit_stage", commit)
        events = []

        async def emit(event):
            events.append(event)

        await QEEGCouncilWorkflow(llm=llm)._stage5("r", ["model-a", "model-b"], emit)
        assert events[-1]["success_count"] == 1
        assert events[-1]["requested_count"] == 2
    assert invoked == ["model-a", "model-b"]
    assert len(sent) == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["data_pack", "vision_transcript"])
@pytest.mark.parametrize("missing", ["product", "product_and_intent"])
async def test_extraction_product_rebuild_uses_acknowledged_units(
    owner, tmp_path, monkeypatch, kind, missing
):
    from backend.council.types import PageImage
    from backend.council.constants import DATA_PACK_SCHEMA_VERSION
    from backend import config

    c = completion()
    monkeypatch.setattr(config, "ARTIFACTS_DIR", tmp_path / "artifacts")
    monkeypatch.setenv("QEEG_VISION_TRANSCRIPT_PAGES_PER_CALL", "1")
    monkeypatch.setenv("QEEG_VISION_PAGES_PER_CALL", "1")
    text = "Session 1"
    report = storage.Report(
        id="q",
        patient_id="p",
        filename="input.txt",
        stored_path=str(tmp_path / "input.txt"),
        extracted_text_path=str(tmp_path / "extracted.txt"),
        mime_type="text/plain",
    )
    Path(report.stored_path).write_text(text)
    Path(report.extracted_text_path).write_text(text)
    (tmp_path / "pages").mkdir()
    for i in (1, 2):
        (tmp_path / "pages" / f"page-{i}.png").write_bytes(b"x")
    with owner.transaction() as session:
        session.add(report)
    sent = []
    pack = {
        "schema_version": DATA_PACK_SCHEMA_VERSION,
        "facts": [],
        "pages_seen": [1, 2],
        "page_inventory": [],
    }
    llm = client(
        lambda req: sent.append(req.content)
        or answer(json.dumps(pack) if kind == "data_pack" else "## Page 1\nTable")
    )
    workflow = QEEGCouncilWorkflow(llm=llm)
    ctx = e.prepare_execution(owner, llm_client=llm)
    images = [PageImage(page=i, base64_png="eA==") for i in (1, 2)]

    async def run():
        if kind == "data_pack":
            return await workflow._ensure_data_pack(
                run_id="r",
                report=report,
                report_text=text,
                page_images=images,
                candidate_extractor_model_ids=["vision-a"],
                strict=False,
            )
        return await workflow._ensure_vision_transcript(
            run_id="r",
            report=report,
            page_images=images,
            transcript_model_id="vision-a",
            strict=False,
        )

    with e.execution_context(ctx):
        first = await run()
        assert first
        sends = len(sent)
        with owner.transaction() as session:
            row = session.scalar(
                select(storage.Artifact).where(storage.Artifact.kind == kind)
            )
        Path(row.content_path).unlink()
        if missing == "product_and_intent":
            # Simulate death before product intent: raw + semantic units survive.
            c._path("artifact/" + row.operation_key).unlink()
            with owner.transaction() as session:
                session.delete(session.get(storage.Artifact, row.id))
        second = await run()
        assert first == second
        assert len(sent) == sends
        assert Path(row.content_path).exists()
        assert any(
            json.loads(p.read_text())
            .get("binding", {})
            .get("key", "")
            .startswith("semantic/s1/")
            for p in (ctx.manifest_path.parent / "completion").glob("*.json")
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "stage_num,models,failures,expected,skip",
    [
        (1, ("mock-a", "mock-b"), 1, (1, 2), False),
        (2, ("mock-a",), 0, (0, 1), True),
        (2, ("mock-a", "mock-b"), 2, (0, 2), False),
        (3, ("mock-a", "mock-b"), 1, (1, 2), False),
        (4, ("mock-a", "mock-b"), 0, (1, 1), False),
        (5, ("mock-a", "mock-b"), 1, (1, 2), False),
        (6, ("mock-a", "mock-b"), 1, (1, 1), False),
    ],
)
async def test_six_stage_success_skip_and_count_policies(
    owner, tmp_path, monkeypatch, stage_num, models, failures, expected, skip
):
    from backend.council.workflow import stages
    from backend.tests.test_council_execution import fixture

    report = seed_stages(owner, tmp_path, monkeypatch, models=models)
    llm = client(lambda req: answer("unused"))
    calls = []

    async def generated(*args, model_id, **kwargs):
        calls.append(model_id)
        if model_id in models[:failures]:
            raise ValueError("definite failed generation")
        if stage_num == 5:
            return json.dumps(fixture("stage5_approve_valid.json"))
        return "Complete\n<!-- END CONSOLIDATED REPORT -->"

    monkeypatch.setattr(stages, "run_stage2_peer_review_json", generated)
    monkeypatch.setattr(stages, "run_stage5_final_review_json", generated)
    workflow = QEEGCouncilWorkflow(llm=llm)
    monkeypatch.setattr(workflow, "_call_longform_chat_with_repairs", generated)
    monkeypatch.setattr(workflow, "_call_model_chat", generated)
    ctx = e.prepare_execution(owner, llm_client=llm)
    events = []

    async def emit(event):
        events.append(event)

    args = (
        ("r", list(models), report, emit)
        if stage_num == 1
        else ("r", emit)
        if stage_num == 4
        else ("r", list(models), emit)
    )
    with e.execution_context(ctx):
        await getattr(workflow, f"_stage{stage_num}")(*args)
        count = len(calls)
        await getattr(workflow, f"_stage{stage_num}")(*args)
        assert len(calls) == count
    assert (events[-1]["success_count"], events[-1]["requested_count"]) == expected
    assert bool(events[-1].get("skipped")) == skip


def test_legacy_nullable_operation_keys_and_unique_owned_keys_migrate(tmp_path):
    import sqlite3
    from sqlalchemy.exc import IntegrityError

    path = tmp_path / "old.db"
    with sqlite3.connect(path) as db:
        db.execute(
            "CREATE TABLE artifacts(id VARCHAR PRIMARY KEY, run_id VARCHAR NOT NULL, stage_num INTEGER NOT NULL, stage_name VARCHAR NOT NULL, model_id VARCHAR NOT NULL, kind VARCHAR NOT NULL, content_path VARCHAR NOT NULL, content_type VARCHAR NOT NULL, created_at DATETIME)"
        )
        db.execute(
            "INSERT INTO artifacts VALUES('old','r',1,'analysis','m','analysis','old.md','text/markdown',NULL)"
        )
    storage.reset_engine(f"sqlite:///{path}")
    storage.init_db()
    storage.init_db()
    with storage.session_scope() as session:
        assert session.get(storage.Artifact, "old").operation_key is None
        for i, key in enumerate([None, None, "s1/member/0/m"]):
            session.add(
                storage.Artifact(
                    id=str(i),
                    run_id="r",
                    operation_key=key,
                    stage_num=1,
                    stage_name="analysis",
                    model_id="m",
                    kind="analysis",
                    content_path="x",
                    content_type="text/plain",
                )
            )
        session.commit()
        session.add(
            storage.Artifact(
                run_id="r",
                operation_key="s1/member/0/m",
                stage_num=1,
                stage_name="analysis",
                model_id="m",
                kind="analysis",
                content_path="x",
                content_type="text/plain",
            )
        )
        with pytest.raises(IntegrityError):
            session.commit()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "damage", ["semantic", "source", "member", "stage_row", "artifact_row"]
)
async def test_completed_receipt_validates_all_bindings(
    owner, tmp_path, monkeypatch, damage
):
    c = completion()
    report = seed_stages(owner, tmp_path, monkeypatch, models=("model-a",))
    sent = []
    llm = client(
        lambda req: sent.append(req.content)
        or answer("Complete\n<!-- END STAGE1 ANALYSIS -->")
    )
    ctx = e.prepare_execution(owner, llm_client=llm)
    with e.execution_context(ctx):
        await QEEGCouncilWorkflow(llm=llm)._stage1("r", ["model-a"], report, silent)
        if damage in ("semantic", "member"):
            path = next(
                p
                for p in (ctx.manifest_path.parent / "completion").glob("*.json")
                if json.loads(p.read_text())["binding"]["key"].startswith(damage + "/")
            )
            path.unlink()
        elif damage == "source":
            next((ctx.manifest_path.parent / "sources").glob("*.json")).unlink()
        else:
            with owner.transaction() as session:
                if damage == "stage_row":
                    session.get(storage.StageReceipt, ("r", 1)).receipt_hash = "0" * 64
                else:
                    session.scalar(
                        select(storage.Artifact).where(
                            storage.Artifact.operation_key.is_not(None)
                        )
                    ).model_id = "changed"
        with pytest.raises(ExecutionConflict):
            c.verified_stage_prefix()
    assert len(sent) == 1


@pytest.mark.asyncio
async def test_storage_failure_cannot_become_terminal_member_failure(
    owner, tmp_path, monkeypatch
):
    c = completion()
    report = seed_stages(owner, tmp_path, monkeypatch, models=("model-a",))
    sent = []
    llm = client(
        lambda req: sent.append(req.content)
        or answer("Complete\n<!-- END STAGE1 ANALYSIS -->")
    )
    ctx = e.prepare_execution(owner, llm_client=llm)
    publish = c._publish

    def broken(owner_arg, path, data):
        if "/artifacts/" in str(path):
            raise OSError("synthetic disk failure")
        publish(owner_arg, path, data)

    with e.execution_context(ctx):
        monkeypatch.setattr(c, "_publish", broken)
        with pytest.raises(ExecutionConflict):
            await QEEGCouncilWorkflow(llm=llm)._stage1("r", ["model-a"], report, silent)
        assert c._read("member/s1/member/0/model-a") is None
        monkeypatch.setattr(c, "_publish", publish)
        await QEEGCouncilWorkflow(llm=llm)._stage1("r", ["model-a"], report, silent)
    assert len(sent) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("boundary", ["artifact_file", "member", "stage_file"])
async def test_process_death_releases_owner_and_reconstructs_without_resend(
    owner, tmp_path, monkeypatch, boundary
):
    import os
    import subprocess
    import sys
    from backend.run_execution import ExecutionStore

    seed_stages(owner, tmp_path, monkeypatch, models=("model-a",))
    storage.init_db()  # Apply the existing source-id migration before admission.
    llm = client(lambda req: answer("unused"))
    ctx = e.prepare_execution(owner, llm_client=llm)
    await ctx.aclose()
    owner.release(state="pending")
    script = r"""
import asyncio, json, os, sys
from pathlib import Path
from backend import storage, config
from backend.council import QEEGCouncilWorkflow, execution as e, completion as c
from backend.run_execution import ExecutionStore
from backend.tests.test_council_execution import client
import httpx
root=Path(sys.argv[1]); boundary=sys.argv[2]
storage.reset_engine(f'sqlite:///{root / "app.db"}'); storage.init_db()
config.ARTIFACTS_DIR=root/'artifacts'
owner=ExecutionStore(storage.engine).claim_run_owner('r')
assert owner is not None
async def main():
    count=root/'sends.txt'
    def send(request):
        count.write_text(count.read_text()+'send\n' if count.exists() else 'send\n')
        return httpx.Response(200,json={'choices':[{'message':{'content':'Complete\n<!-- END STAGE1 ANALYSIS -->'}}]})
    llm=client(send)
    ctx=e.prepare_execution(owner,llm_client=llm)
    publish=c._publish
    def crash(handle,path,data):
        publish(handle,path,data)
        try: key=json.loads(data).get('binding',{}).get('key','')
        except ValueError: key=''
        if (boundary=='artifact_file' and '/artifacts/' in str(path)) or (boundary=='member' and key.startswith('member/')) or (boundary=='stage_file' and key=='stage/1'):
            os._exit(71)
    c._publish=crash
    with owner, e.execution_context(ctx):
        with owner.transaction() as session: report=session.get(storage.Report,'q')
        async def emit(event): pass
        await QEEGCouncilWorkflow(llm=llm)._stage1('r',['model-a'],report,emit)
        assert c.verified_stage_prefix()==1
    await ctx.aclose()
asyncio.run(main())
"""
    env = dict(os.environ)
    for key in (
        "OPENAI_API_KEY",
        "OPENROUTER_API_KEY",
        "ANTHROPIC_API_KEY",
        "GEMINI_API_KEY",
        "GOOGLE_API_KEY",
        "CLIPROXY_API_KEY",
    ):
        env[key] = ""
    env.update(
        DATA_DIR=str(tmp_path / "scratch" / "data"),
        QEEG_ANALYSIS_ROOT=str(tmp_path / "scratch"),
    )
    died = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path), boundary],
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert died.returncode == 71, died.stderr
    reclaimed = ExecutionStore(storage.engine).claim_run_owner("r")
    assert reclaimed is not None
    assert reclaimed.generation > owner.generation
    reclaimed.release(state="pending")
    resumed = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path), "resume"],
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert resumed.returncode == 0, resumed.stderr
    assert (tmp_path / "sends.txt").read_text().splitlines() == ["send"]


@pytest.mark.asyncio
async def test_owned_pipeline_uses_contiguous_receipts_and_projects_recovered_counts(
    owner, tmp_path, monkeypatch
):
    from backend.council.workflow import core, stages
    from backend.tests.test_council_execution import fixture

    models = ["mock-a", "mock-b"]
    seed_stages(owner, tmp_path, monkeypatch, models=tuple(models))
    monkeypatch.setattr(core, "ARTIFACTS_DIR", tmp_path / "artifacts")
    sent = []
    llm = client(
        lambda req: sent.append(req.content)
        or answer("Complete\n<!-- END CONSOLIDATED REPORT -->")
    )

    async def review(**kwargs):
        return json.dumps(fixture("stage5_approve_valid.json"))

    monkeypatch.setattr(stages, "run_stage2_peer_review_json", review)
    monkeypatch.setattr(stages, "run_stage5_final_review_json", review)
    ctx = e.prepare_execution(owner, llm_client=llm)
    events = []

    async def emit(event):
        events.append(event)

    with e.execution_context(ctx):
        workflow = QEEGCouncilWorkflow(llm=llm)
        await workflow.run_pipeline("r", emit)
        assert completion().verified_stage_prefix() == 6
        count = len(sent)
        assert count >= 6  # Stages 1,3,4,6 really dispatched despite pre-existing rows.
        events.clear()
        await workflow.run_pipeline("r", emit)
        assert len(sent) == count
    assert events[-1]["status"] == "complete"
    assert events[-1]["success_count"] == events[-1]["requested_count"] == 1
    assert [event["stage_num"] for event in events if event.get("stage_num")] == list(
        range(1, 7)
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("damage", ["paid_row", "paid_file", "product_intent"])
async def test_completed_stage_checks_extraction_product_authority(
    owner, tmp_path, monkeypatch, damage
):
    c = completion()
    report = seed_stages(owner, tmp_path, monkeypatch, models=("model-a",))
    llm = client(lambda req: answer("Complete\n<!-- END STAGE1 ANALYSIS -->"))
    ctx = e.prepare_execution(owner, llm_client=llm)
    workflow = QEEGCouncilWorkflow(llm=llm)

    async def extract(**kwargs):
        e.bind_source(
            "s1/data-pack-input", {"report_id": report.id}, consumers="s1/data-pack/"
        )
        await e.execute_unit(
            "s1/data-pack/0/vision-a/chunk/1",
            llm.chat_completions(
                model_id="vision-a", messages=[{"role": "user", "content": "extract"}]
            ),
        )
        c.save_product("data_pack", "{}")
        return {}

    monkeypatch.setattr(workflow, "_ensure_data_pack", extract)
    with e.execution_context(ctx):
        await workflow._stage1("r", ["model-a"], report, silent)
        if damage == "product_intent":
            c._path("artifact/s1/data-pack").unlink()
        else:
            with owner.transaction() as session:
                paid = session.scalar(
                    select(storage.PaidRequest).where(
                        storage.PaidRequest.scope_key
                        == "s1/data-pack/0/vision-a/chunk/1"
                    )
                )
                if damage == "paid_row":
                    paid.request_hash = "0" * 64
                else:
                    Path(paid.response_path).write_text("changed")
        with pytest.raises(ExecutionConflict):
            c.verified_stage_prefix()


@pytest.mark.asyncio
async def test_failed_consolidation_is_terminal_without_a_completion_receipt(
    owner, tmp_path, monkeypatch
):
    seed_stages(owner, tmp_path, monkeypatch)
    monkeypatch.setenv("QEEG_STAGE4_REPAIR_CALLS", "0")
    sent = []
    llm = client(lambda req: sent.append(req.content) or answer("incomplete"))
    ctx = e.prepare_execution(owner, llm_client=llm)
    workflow = QEEGCouncilWorkflow(llm=llm)
    with e.execution_context(ctx):
        for _ in range(2):
            with pytest.raises(RuntimeError, match="remained incomplete"):
                await workflow._stage4("r", silent)
        assert completion()._read("member/s4/consolidation")["disposition"] == "failed"
        with owner.transaction() as session:
            assert session.get(storage.StageReceipt, ("r", 4)) is None
    assert len(sent) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "models",
    [
        ("model-a",) * 12,
        ("model/a", "model_a", "model_a.member-1"),
        ("model-a", "model-a", "model-a.member-1", "model-a.member-1-1"),
    ],
)
async def test_duplicate_and_normalized_model_paths_preserve_every_member(
    owner, tmp_path, monkeypatch, models
):
    from backend.council.db_utils import _stage_artifacts

    report = seed_stages(owner, tmp_path, monkeypatch, models=models)
    sent = []

    async def send(req):
        index = len(sent)
        sent.append(req.content)
        await asyncio.sleep(0.001 * (len(models) - index))
        return answer(f"Member {index}\n<!-- END STAGE1 ANALYSIS -->")

    llm = client(send)
    ctx = e.prepare_execution(owner, llm_client=llm)
    with e.execution_context(ctx):
        await QEEGCouncilWorkflow(llm=llm)._stage1("r", list(models), report, silent)
        with owner.transaction() as session:
            rows = [
                a
                for a in _stage_artifacts(session, "r", 1, kind="analysis")
                if a.operation_key
            ]
        assert [int(a.operation_key.split("/")[2]) for a in rows] == list(
            range(len(models))
        )
        assert len({a.content_path for a in rows}) == len(models)
        assert all(
            Path(a.content_path).read_text().startswith(f"Member {i}\n")
            for i, a in enumerate(rows)
        )
        await QEEGCouncilWorkflow(llm=llm)._stage1("r", list(models), report, silent)
    assert len(sent) == len(models)


@pytest.mark.asyncio
async def test_definite_consolidator_auth_failure_preserves_clinical_status_on_replay(
    owner, tmp_path, monkeypatch
):
    from backend.council.workflow.exceptions import _NeedsAuth

    seed_stages(owner, tmp_path, monkeypatch)
    sent = []
    llm = client(
        lambda req: sent.append(req.content)
        or httpx.Response(
            401,
            json={
                "error": {"type": "authentication_error", "message": "invalid API key"}
            },
        )
    )
    ctx = e.prepare_execution(owner, llm_client=llm)
    with e.execution_context(ctx):
        for _ in range(2):
            with pytest.raises(_NeedsAuth):
                await QEEGCouncilWorkflow(llm=llm)._stage4("r", silent)
    assert len(sent) == 1


@pytest.mark.asyncio
async def test_pipeline_resumes_the_remaining_member_after_predispatch_cancellation(
    owner, tmp_path, monkeypatch
):
    from backend.council.workflow import core

    seed_stages(owner, tmp_path, monkeypatch)
    monkeypatch.setattr(core, "ARTIFACTS_DIR", tmp_path / "artifacts")
    sent = []
    llm = client(
        lambda req: sent.append(json.loads(req.content)["model"])
        or answer("Complete\n<!-- END STAGE1 ANALYSIS -->")
    )
    ctx = e.prepare_execution(owner, llm_client=llm)
    ready = asyncio.Event()

    async def interrupted_emit(event):
        if (
            event.get("task") == "stage1_model"
            and event.get("model_id") == "model-b"
            and event.get("status") == "start"
        ):
            await (
                asyncio.Event().wait()
            )  # Still before any paid marker for this member.
        if (
            event.get("task") == "stage1_model"
            and event.get("model_id") == "model-a"
            and event.get("status") == "complete"
        ):
            ready.set()

    workflow = QEEGCouncilWorkflow(llm=llm)
    with e.execution_context(ctx):
        task = asyncio.create_task(workflow.run_pipeline("r", interrupted_emit))
        await ready.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert sent == ["model-a"]
        assert completion().verified_stage_prefix() == 0
        events = []

        async def resumed_emit(event):
            events.append(event)
            if (
                event.get("stage_num") == 1
                and event.get("status") == "complete"
                and not event.get("task")
            ):
                raise ExecutionConflict("stop after the committed resumed stage")

        with pytest.raises(ExecutionConflict, match="stop after"):
            await workflow.run_pipeline("r", resumed_emit)
        assert completion().verified_stage_prefix() == 1
    assert sent == ["model-a", "model-b"]
    assert events[-1]["success_count"] == events[-1]["requested_count"] == 2
