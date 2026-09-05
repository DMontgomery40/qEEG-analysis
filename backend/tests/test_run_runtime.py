"""Lifecycle ownership tests: real SQLite/flocks and synthetic continuations."""

import asyncio
import importlib

import pytest
from sqlalchemy.orm import Session

from backend import storage
from backend.run_execution import ExecutionStore


def runtime_type():
    return importlib.import_module("backend.run_runtime").RunRuntime


def add_run(store, run_id, *, status="created", admitted=True):
    with Session(store.engine) as session:
        session.add(
            storage.Run(
                id=run_id,
                patient_id=run_id,
                report_id="q",
                status=status,
                council_model_ids_json='["m"]',
                consolidator_model_id="m",
                analysis_input_fingerprint="source",
            )
        )
        session.commit()
    if admitted:
        store.request_run_start(run_id)


async def until(predicate):
    async def wait():
        while not predicate():
            await asyncio.sleep(0.01)

    await asyncio.wait_for(wait(), 5)


@pytest.mark.asyncio
async def test_empty_consumer_continues_after_failure_and_later_admission(
    temp_data_dir,
):
    store = ExecutionStore(storage.engine)
    calls = []

    async def continuation(owner):
        calls.append(owner.run_id)
        if owner.run_id == "bad":
            raise OSError("temporary local fault")
        return "done", None

    runtime = runtime_type()(
        store,
        continuation=continuation,
        poll_interval=0.02,
        retry_delay=0.15,
        concurrency=2,
        page_size=1,
    )
    await runtime.start()
    try:
        await asyncio.sleep(0.05)
        add_run(store, "bad")
        add_run(store, "good")
        await until(lambda: "good" in calls)
        add_run(store, "later")
        await until(lambda: "later" in calls)
        assert calls.count("bad") <= 2
    finally:
        await runtime.stop()
    assert not runtime.tasks


@pytest.mark.asyncio
async def test_contended_first_pages_do_not_starve_independent_run(temp_data_dir):
    store = ExecutionStore(storage.engine)
    for name in ("a", "b", "c", "d"):
        add_run(store, name)
    owners = [store.claim_run_owner(name) for name in ("a", "b", "c")]
    calls = []

    async def continuation(owner):
        calls.append(owner.run_id)
        return "done", None

    runtime = runtime_type()(
        store, continuation=continuation, poll_interval=0.01, concurrency=1, page_size=1
    )
    await runtime.start()
    try:
        await until(lambda: calls == ["d"])
        assert all(owner._fd is not None for owner in owners)
    finally:
        await runtime.stop()
        for owner in owners:
            owner.close()


@pytest.mark.asyncio
async def test_two_consumers_one_owner_and_shutdown_retains_tasks(temp_data_dir):
    store = ExecutionStore(storage.engine)
    add_run(store, "r")
    entered, release = asyncio.Event(), asyncio.Event()
    calls = []

    async def continuation(owner):
        calls.append(owner.run_id)
        entered.set()
        await release.wait()
        assert store.claim_run_owner("r") is None
        return "done", None

    runtimes = [
        runtime_type()(store, continuation=continuation, poll_interval=0.01)
        for _ in range(2)
    ]
    for runtime in runtimes:
        await runtime.start()
    await entered.wait()
    stopping = [asyncio.create_task(runtime.stop()) for runtime in runtimes]
    await asyncio.sleep(0.03)
    for task in stopping:
        task.cancel()
        task.cancel()
    await asyncio.sleep(0.03)
    assert calls == ["r"]
    assert store.claim_run_owner("r") is None
    assert not all(task.done() for task in stopping)
    release.set()
    await asyncio.gather(*stopping)
    assert all(not runtime.tasks for runtime in runtimes)


def test_legacy_inventory_is_read_only(temp_data_dir):
    mod = importlib.import_module("backend.run_runtime")
    store = ExecutionStore(storage.engine)
    for status in ("created", "running", "failed", "needs_auth", "complete"):
        add_run(store, status, status=status, admitted=False)
    result = mod.compatibility_inventory(store)
    assert result["counts"]["legacy_reconciliation_required"] == 3
    assert result["counts"]["legacy_complete_no_post_intent"] == 1
    with Session(store.engine) as session:
        assert all(row.start_requested_at is None for row in session.query(storage.Run))
        assert session.query(storage.PostObligation).count() == 0


@pytest.mark.asyncio
async def test_scan_database_error_does_not_kill_continuing_consumer(
    temp_data_dir, monkeypatch
):
    store = ExecutionStore(storage.engine)
    original = store.list_due_runs
    calls = []
    failures = [True]

    def scan(*args, **kwargs):
        if failures:
            failures.pop()
            raise OSError("temporary DB path failure")
        return original(*args, **kwargs)

    monkeypatch.setattr(store, "list_due_runs", scan)

    async def continuation(owner):
        calls.append(owner.run_id)
        return "done", None

    runtime = runtime_type()(store, continuation=continuation, poll_interval=0.01)
    await runtime.start()
    try:
        add_run(store, "r")
        await until(lambda: calls == ["r"])
    finally:
        await runtime.stop()


@pytest.fixture
def clinical(temp_data_dir, monkeypatch):
    from backend import config
    from backend.council.workflow import core, stages
    from backend.tests.test_council_execution import seed_stages, fixture

    store = ExecutionStore(storage.engine)
    add_run(store, "r")
    with Session(store.engine) as session:
        row = session.get(storage.Run, "r")
        row.patient_id = "p"
        session.add(storage.Patient(id="p", label="ZZ_01-01-1900", notes=""))
        session.commit()
    owner = store.claim_run_owner("r")
    seed_stages(owner, temp_data_dir, monkeypatch, models=("mock-a", "mock-b"))
    owner.release()
    config.DISCOVERED_MODEL_IDS.update(("mock-a", "mock-b"))
    monkeypatch.setattr(core, "ARTIFACTS_DIR", temp_data_dir / "artifacts")
    monkeypatch.setenv("QEEG_AUTO_PATIENT_FACING", "0")
    monkeypatch.setenv("QEEG_AUTO_CATHODE_VIDEO", "1")
    monkeypatch.setenv("QEEG_ROUTE_OPENROUTER_EXTRAS_DIRECT", "0")

    async def review(**kwargs):
        import json

        return json.dumps(fixture("stage5_approve_valid.json"))

    monkeypatch.setattr(stages, "run_stage2_peer_review_json", review)
    monkeypatch.setattr(stages, "run_stage5_final_review_json", review)
    return store


def saved_run(store, run_id="r"):
    with Session(store.engine) as session:
        row = session.get(storage.Run, run_id)
        return row.execution_state, row.status, row.blocked_reason


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "boundary", ["observer", "stage1_progress", "all_members", "council_complete"]
)
async def test_actual_six_stage_resume_preserves_policy_and_avoids_second_dispatch(
    clinical, monkeypatch, boundary
):
    from backend import main
    from backend.council import QEEGCouncilWorkflow, completion
    from backend.council.workflow import core
    from backend.tests.test_council_execution import client
    from backend.tests.test_council_completion import answer

    sent = []
    llm = client(
        lambda req: sent.append(req.content)
        or answer("Complete\n<!-- END CONSOLIDATED REPORT -->")
    )
    original_progress = core._append_progress_event
    original_publish = completion._publish
    interrupted = []

    def progress(run_id, event):
        original_progress(run_id, event)
        stop = (
            boundary == "stage1_progress"
            and event.get("stage_num") == 1
            and event.get("status") == "complete"
            and not event.get("task")
        ) or (
            boundary == "council_complete"
            and event.get("status") == "complete"
            and not event.get("stage_num")
        )
        if stop and not interrupted:
            interrupted.append(True)
            raise OSError("temporary progress filesystem fault")

    def publish(owner, path, data):
        if boundary == "all_members" and not interrupted and b'"key":"stage/1"' in data:
            interrupted.append(True)
            raise OSError("stage receipt write interrupted after member saves")
        return original_publish(owner, path, data)

    monkeypatch.setattr(core, "_append_progress_event", progress)
    monkeypatch.setattr(completion, "_publish", publish)

    async def broken_observer(*args):
        if boundary == "observer":
            raise OSError("SSE disconnected")

    async def forbidden(*args):
        raise AssertionError("legacy auto generator launched")

    monkeypatch.setattr(main, "_auto_generate_patient_facing_for_run", forbidden)
    monkeypatch.setattr(main, "_auto_generate_cathode_video_for_run", forbidden)
    runtime = runtime_type()(
        clinical,
        llm=llm,
        workflow=QEEGCouncilWorkflow(llm=llm),
        publish=broken_observer,
        poll_interval=0.01,
        retry_delay=0.02,
    )
    await runtime.start()
    try:
        await until(lambda: saved_run(clinical)[0] in ("done", "blocked"))
    finally:
        await runtime.stop()
        await llm.aclose()
    assert saved_run(clinical)[:2] == ("done", "complete"), saved_run(clinical)
    # Existing six-stage fixture sends 2 Stage1 + 2 revisions + consolidation + final.
    assert len(sent) == 6
    with Session(clinical.engine) as session:
        assert session.query(storage.StageReceipt).count() == 6
        assert session.query(storage.PaidRequest).count() == 6
        assert {p.kind: p.state for p in session.query(storage.PostObligation)} == {
            "patient_facing": "skipped",
            "cathode": "skipped",
        }


@pytest.mark.asyncio
async def test_runtime_unknown_remains_blocked_on_start_retries(clinical):
    import httpx
    from backend.council import QEEGCouncilWorkflow
    from backend.tests.test_council_execution import client

    sent = []

    def send(req):
        sent.append(req.content)
        raise httpx.ReadError("lost after dispatch")

    llm = client(send)
    runtime = runtime_type()(
        clinical,
        llm=llm,
        workflow=QEEGCouncilWorkflow(llm=llm),
        poll_interval=0.01,
        retry_delay=0.01,
    )
    await runtime.start()
    try:
        await until(lambda: saved_run(clinical)[0] == "blocked")
        for _ in range(3):
            clinical.request_run_start("r")
            runtime.wake()
            await asyncio.sleep(0.02)
        assert len(sent) == 1
        assert saved_run(clinical)[2] == "paid_outcome_unknown"
    finally:
        await runtime.stop()
        await llm.aclose()


@pytest.mark.asyncio
async def test_catalogue_guard_recovers_free_receipt_but_stops_new_pinned_send(
    clinical,
):
    from backend import config
    from backend.council.execution import owned_execution, execute_unit
    from backend.run_runtime import current_catalogue_guard, ModelUnavailable
    from backend.tests.test_council_execution import client
    from backend.tests.test_council_completion import answer

    sent = []
    llm = client(lambda req: sent.append(req.content) or answer("saved"))
    owner = clinical.claim_run_owner("r")
    try:
        async with owned_execution(owner, llm_client=llm):
            with current_catalogue_guard():
                for _ in range(2):
                    assert (
                        await execute_unit(
                            "s1/member/0/mock-a",
                            llm.chat_completions(
                                model_id="mock-a",
                                messages=[{"role": "user", "content": "exact"}],
                            ),
                        )
                        == "saved"
                    )
                    config.DISCOVERED_MODEL_IDS.clear()
                with pytest.raises(ModelUnavailable):
                    await execute_unit(
                        "s1/member/1/mock-b",
                        llm.chat_completions(
                            model_id="mock-b",
                            messages=[{"role": "user", "content": "next"}],
                        ),
                    )
        with owner.transaction() as session:
            assert sorted(p.state for p in session.query(storage.PaidRequest)) == [
                "prepared",
                "response_saved",
            ]
        assert len(sent) == 1
    finally:
        owner.release()
        await llm.aclose()


@pytest.mark.asyncio
async def test_contended_post_admission_keeps_frozen_config_and_has_retryable_budget(
    temp_data_dir, monkeypatch
):
    from backend.tests.test_patient_postprocessing import ready as ready_fixture
    from backend.run_runtime import AdmissionUnavailable
    from backend import patient_postprocessing as post

    store, run_id, cfg = ready_fixture.__wrapped__(temp_data_dir, monkeypatch)
    store.request_run_start(run_id)
    owner = store.claim_run_owner(run_id)
    runtime = runtime_type()(store)
    try:
        with pytest.raises(AdmissionUnavailable):
            await runtime.admit_post(run_id, config_snapshot=cfg, budget=0.05)
        assert post.project_patient_facing(store, run_id)["state"] == "absent"
        task = asyncio.create_task(runtime.admit_post(run_id, config_snapshot=cfg))
        await asyncio.sleep(0.06)
        owner.release()
        monkeypatch.setenv("QEEG_PATIENT_FACING_MODEL", "changed-after-request")
        result = await task
        assert result["state"] == "pending"
        manifest = post._load(result["manifest_path"])
        assert manifest["config"]["model_id"] == "writer"
    finally:
        owner.close()
        await runtime.stop()


_PROCESS_RUNTIME = r"""
import asyncio, json, os, sys, time
from pathlib import Path
from sqlalchemy.orm import Session
from backend import storage, config
from backend.council import QEEGCouncilWorkflow, completion
from backend.council.workflow import core, stages
from backend.run_execution import ExecutionStore
from backend.run_runtime import RunRuntime
from backend.tests.test_council_execution import client, fixture
from backend.tests.test_council_completion import answer
root=Path(sys.argv[1]); boundary=sys.argv[2]
storage.reset_engine(f'sqlite:///{root / "app.db"}'); storage.init_db()
config.ARTIFACTS_DIR=root/'artifacts'; core.ARTIFACTS_DIR=config.ARTIFACTS_DIR
config.DISCOVERED_MODEL_IDS.update(('mock-a','mock-b'))
stages.DISCOVERED_MODEL_IDS=config.DISCOVERED_MODEL_IDS
async def review(**kwargs): return json.dumps(fixture('stage5_approve_valid.json'))
stages.run_stage2_peer_review_json=review; stages.run_stage5_final_review_json=review
async def main():
    sent=root/'sends.txt'
    def send(request):
        with sent.open('a') as f: f.write('send\n'); f.flush(); os.fsync(f.fileno())
        if boundary=='dispatched': os._exit(71)
        return answer('Complete\n<!-- END CONSOLIDATED REPORT -->')
    llm=client(send)
    original=completion._publish
    def publish(owner,path,data):
        original(owner,path,data)
        try: key=json.loads(data).get('binding',{}).get('key','')
        except ValueError: key=''
        if boundary=='member' and key.startswith('member/s1/') or boundary=='stage' and key=='stage/1': os._exit(71)
    completion._publish=publish
    progress=core._append_progress_event
    def append(run_id,event):
        progress(run_id,event)
        if boundary=='complete' and event.get('status')=='complete' and not event.get('stage_num'): os._exit(71)
    core._append_progress_event=append
    runtime=RunRuntime(ExecutionStore(storage.engine),llm=llm,workflow=QEEGCouncilWorkflow(llm=llm),poll_interval=.01,retry_delay=.01)
    await runtime.start()
    deadline=time.monotonic()+10
    while time.monotonic()<deadline:
        with Session(storage.engine) as s: state=s.get(storage.Run,'r').execution_state
        if state in ('done','blocked'): break
        await asyncio.sleep(.01)
    await runtime.stop(); await llm.aclose()
    assert state==('blocked' if boundary=='resume_unknown' else 'done'), state
asyncio.run(main())
"""


@pytest.mark.parametrize(
    "boundary", ["intent", "member", "stage", "complete", "dispatched"]
)
def test_process_replacement_finds_original_intent_without_another_post(
    clinical, temp_data_dir, boundary
):
    import os
    import subprocess
    import sys

    env = dict(os.environ)
    env.update(
        DATA_DIR=str(temp_data_dir / "scratch" / "data"),
        QEEG_ANALYSIS_ROOT=str(temp_data_dir / "scratch"),
    )
    for key in (
        "OPENAI_API_KEY",
        "OPENROUTER_API_KEY",
        "ANTHROPIC_API_KEY",
        "GEMINI_API_KEY",
        "GOOGLE_API_KEY",
        "CLIPROXY_API_KEY",
    ):
        env[key] = ""
    if boundary != "intent":
        died = subprocess.run(
            [sys.executable, "-c", _PROCESS_RUNTIME, str(temp_data_dir), boundary],
            env=env,
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert died.returncode == 71, died.stderr
    resumed = subprocess.run(
        [
            sys.executable,
            "-c",
            _PROCESS_RUNTIME,
            str(temp_data_dir),
            "resume_unknown" if boundary == "dispatched" else "resume",
        ],
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert resumed.returncode == 0, resumed.stderr
    assert (temp_data_dir / "sends.txt").read_text().splitlines() == ["send"] * (
        1 if boundary == "dispatched" else 6
    )
    assert saved_run(clinical)[0] == ("blocked" if boundary == "dispatched" else "done")


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["render", "sync"])
async def test_runtime_post_only_recovers_free_work_without_council(
    temp_data_dir, monkeypatch, failure
):
    from backend import patient_postprocessing as post
    from backend.tests.test_patient_postprocessing import ready as ready_fixture, llm

    store, run_id, cfg = ready_fixture.__wrapped__(temp_data_dir, monkeypatch)
    post.admit_patient_facing(store, run_id, config_snapshot=cfg)
    calls = []
    failed = []

    def render(*args, **kwargs):
        if failure == "render" and not failed:
            failed.append(True)
            raise OSError("render interrupted")
        args[1].write_bytes(b"%PDF synthetic original")

    def sync(*args):
        if failure == "sync" and not failed:
            failed.append(True)
            raise OSError("sync interrupted")
        return True

    monkeypatch.setattr(post.writer, "render_patient_facing_markdown_to_pdf", render)

    class NoCouncil:
        async def run_pipeline(self, *args, **kwargs):
            raise AssertionError("historical post-only request entered council")

    client = llm(calls)
    runtime = runtime_type()(
        store,
        llm=client,
        workflow=NoCouncil(),
        sync=sync,
        poll_interval=0.01,
        retry_delay=0.02,
    )
    await runtime.start()
    try:
        await until(lambda: saved_run(store, run_id)[0] in ("done", "blocked"))
    finally:
        await runtime.stop()
        await client.aclose()
    assert saved_run(store, run_id)[:2] == ("done", "complete"), saved_run(
        store, run_id
    )
    result = post.project_patient_facing(store, run_id)
    assert result["verified"] and not result["delivery_verified"]
    assert len(calls) == 1
    with Session(store.engine) as session:
        run = session.get(storage.Run, run_id)
        assert (
            run.analysis_input_fingerprint == "" and run.execution_manifest_hash is None
        )
        assert session.query(storage.StageReceipt).count() == 0


_THREAD_CONSUMER = r"""
import asyncio,json,os,sys,time
from pathlib import Path
import httpx
from sqlalchemy.orm import Session
from backend import storage
from backend.council.execution import owned_execution,execute_unit
from backend.paid_transport import PaidClient,owned_to_thread
from backend.run_runtime import RunRuntime
from backend.run_execution import ExecutionStore
root=Path(sys.argv[1]); role=sys.argv[2]
storage.reset_engine(f'sqlite:///{root / "app.db"}'); storage.init_db()
store=ExecutionStore(storage.engine)
async def continuation(owner):
    def send(request):
        with (root/'sends.txt').open('a') as out:out.write(owner.run_id+'\n');out.flush();os.fsync(out.fileno())
        if owner.run_id=='r':
            (root/'entered').touch()
            deadline=time.monotonic()+15
            while not (root/'release').exists() and time.monotonic()<deadline:time.sleep(.01)
            assert (root/'release').exists()
        return httpx.Response(200,json={'choices':[{'message':{'content':'saved'}}]})
    def call():
        with PaidClient(transport=httpx.MockTransport(send)) as client:
            return client.post('http://synthetic/v1/chat/completions',json={'model':'m','messages':[{'role':'user','content':'original'}]}).json()['choices'][0]['message']['content']
    async with owned_execution(owner):
        await execute_unit('s1/member/0/m',owned_to_thread(call))
    return 'done',None
async def main():
    runtime=RunRuntime(store,continuation=continuation,concurrency=1,page_size=1,poll_interval=.01)
    await runtime.start()
    if role=='first':
        while not (root/'entered').exists():await asyncio.sleep(.01)
        stopping=asyncio.create_task(runtime.stop())
        await asyncio.sleep(.03)
        stopping.cancel();stopping.cancel()
        (root/'shutdown-draining').touch()
        await stopping
    else:
        deadline=time.monotonic()+10
        while time.monotonic()<deadline:
            with Session(store.engine) as s: state=s.get(storage.Run,'z').execution_state
            if state=='done':break
            await asyncio.sleep(.01)
        assert state=='done'
        (root/'independent-complete').touch()
        await runtime.stop()
asyncio.run(main())
"""


def test_real_consumers_keep_thread_owner_during_cancelled_shutdown_and_progress_independent_patient(
    temp_data_dir,
):
    import os
    import subprocess
    import sys
    import time

    store = ExecutionStore(storage.engine)
    add_run(store, "r")
    env = dict(os.environ)
    env.update(
        DATA_DIR=str(temp_data_dir / "scratch" / "data"),
        QEEG_ANALYSIS_ROOT=str(temp_data_dir / "scratch"),
    )

    def wait_file(name):
        deadline = time.monotonic() + 10
        while not (temp_data_dir / name).exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert (temp_data_dir / name).exists(), name

    first = subprocess.Popen(
        [sys.executable, "-c", _THREAD_CONSUMER, str(temp_data_dir), "first"],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    second = None
    try:
        wait_file("shutdown-draining")
        assert store.claim_run_owner("r") is None
        add_run(store, "z")
        second = subprocess.Popen(
            [sys.executable, "-c", _THREAD_CONSUMER, str(temp_data_dir), "second"],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        wait_file("independent-complete")
        assert first.poll() is None
        assert store.claim_run_owner("r") is None
        assert (temp_data_dir / "sends.txt").read_text().splitlines() == ["r", "z"]
        (temp_data_dir / "release").touch()
        _, err = first.communicate(timeout=10)
        assert first.returncode == 0, err
        _, err = second.communicate(timeout=10)
        assert second.returncode == 0, err
        assert saved_run(store, "r")[0] == saved_run(store, "z")[0] == "done"
    finally:
        (temp_data_dir / "release").touch()
        for process in (first, second):
            if process is not None and process.poll() is None:
                process.kill()
                process.communicate()


@pytest.mark.asyncio
@pytest.mark.parametrize("stage", [2, 5])
async def test_sdk_wrapped_unavailable_pinned_model_stops_without_repair_or_paid_send(
    clinical, stage
):
    from backend import config
    from backend.council import ai_review_agents as agents
    from backend.council.execution import owned_execution, execute_unit
    from backend.run_runtime import current_catalogue_guard, ModelUnavailable
    from backend.tests.test_council_execution import client

    sent = []
    llm = client(lambda req: sent.append(req.content))
    owner = clinical.claim_run_owner("r")
    config.DISCOVERED_MODEL_IDS.clear()
    try:
        async with owned_execution(owner, llm_client=llm):
            with current_catalogue_guard(), pytest.raises(ModelUnavailable):
                function = (
                    agents.run_stage2_peer_review
                    if stage == 2
                    else agents.run_stage5_final_review
                )
                kwargs = {"expected_labels": ["A", "B"]} if stage == 2 else {}
                await execute_unit(
                    f"s{stage}/reviewer/0/mock-a",
                    function(
                        llm_client=llm,
                        model_id="mock-a",
                        prompt_text="Original review.",
                        **kwargs,
                    ),
                )
        assert sent == []
        with owner.transaction() as session:
            assert [p.state for p in session.query(storage.PaidRequest)] == ["prepared"]
    finally:
        owner.release()
        await llm.aclose()


@pytest.mark.asyncio
async def test_startup_accepts_recovery_while_initial_catalogue_is_unavailable(
    temp_data_dir, monkeypatch
):
    from backend import main
    from backend.tests.test_council_execution import client

    started, release = asyncio.Event(), asyncio.Event()
    llm = client(lambda _: (_ for _ in ()).throw(AssertionError("unexpected HTTP")))

    async def catalogue():
        started.set()
        await release.wait()
        return []

    monkeypatch.setattr(llm, "list_models", catalogue)
    monkeypatch.setattr(main, "_get_mock_llm_client", lambda: None)
    monkeypatch.setattr(main, "AsyncOpenAICompatClient", lambda **kwargs: llm)
    monkeypatch.setattr(main, "_ensure_project_clipr_config", lambda: None)
    monkeypatch.setattr(main, "_sync_home_auth_to_project", lambda: None)
    monkeypatch.setenv("QEEG_PORTAL_RAW_SYNC_WATCHER", "0")
    monkeypatch.setattr(main.app.state, "run_runtime", None, raising=False)
    try:
        await asyncio.wait_for(main._startup(), 0.5)
        await started.wait()
        assert main.app.state.run_runtime._scan_task is not None
        assert not main.app.state.model_refresh_task.done()
    finally:
        release.set()
        await main._shutdown()


@pytest.mark.asyncio
@pytest.mark.parametrize("state", ["pending", "owned", "blocked", "complete_post"])
async def test_legacy_pipeline_cannot_bypass_owned_run_even_when_batch_selects_it(
    clinical, state
):
    from backend.council import QEEGCouncilWorkflow
    from backend.run_execution import ExecutionConflict
    from backend.tests.test_council_execution import client
    from backend.tests.test_council_completion import answer

    sent = []
    llm = client(lambda req: sent.append(req.content) or answer("Complete"))
    owner = clinical.claim_run_owner("r") if state in ("owned", "blocked") else None
    if state == "blocked":
        owner.release(state="blocked", blocked_reason="paid_outcome_unknown")
    if state == "complete_post":
        with Session(clinical.engine) as session:
            session.get(storage.Run, "r").status = "complete"
            session.add(
                storage.PostObligation(
                    run_id="r",
                    kind="patient_facing",
                    manifest_path="original.json",
                    manifest_hash="a" * 64,
                    owner_token="original",
                    owner_generation=1,
                )
            )
            session.commit()
    before = saved_run(clinical)
    try:
        with pytest.raises(ExecutionConflict, match="owned consumer"):
            await QEEGCouncilWorkflow(llm=llm).run_pipeline("r")
        assert sent == []
        assert saved_run(clinical) == before
        with Session(clinical.engine) as session:
            assert not storage.claim_run_start(session, "r")
        assert saved_run(clinical) == before
    finally:
        if owner is not None:
            owner.close()
        await llm.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "authority", ["intent", "post_only", "blocked_post", "done_post"]
)
async def test_legacy_post_hook_cannot_generate_outside_original_authority(
    temp_data_dir, monkeypatch, authority
):
    from backend import main
    from backend.run_execution import ExecutionConflict
    from backend.tests.test_patient_postprocessing import ready as ready_fixture

    store, run_id, cfg = ready_fixture.__wrapped__(temp_data_dir, monkeypatch)
    if authority == "intent":
        store.request_run_start(run_id)
    else:
        with Session(store.engine) as session:
            session.add(
                storage.PostObligation(
                    run_id=run_id,
                    kind="patient_facing",
                    manifest_path="original.json",
                    manifest_hash="a" * 64,
                    owner_token="original",
                    owner_generation=1,
                    state={
                        "post_only": "pending",
                        "blocked_post": "blocked",
                        "done_post": "done",
                    }[authority],
                )
            )
            session.commit()
    monkeypatch.setenv("QEEG_AUTO_PATIENT_FACING", "1")
    sends = []

    async def forbidden(*args, **kwargs):
        sends.append(args)
        raise AssertionError("old post dispatcher attempted a subprocess")

    monkeypatch.setattr(main.asyncio, "create_subprocess_exec", forbidden)

    class Broker:
        async def publish(self, *args):
            pass

    with pytest.raises(ExecutionConflict, match="owned consumer"):
        await main._auto_generate_patient_facing_for_run(run_id, Broker())
    assert sends == []
    assert saved_run(store, run_id)[1] == "complete"
