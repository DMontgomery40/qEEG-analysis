"""Durable patient outputs: synthetic HTTP, local scratch files, no delivery."""

import json
from pathlib import Path

import httpx
import pytest
from sqlalchemy import select

from backend import storage
from backend.llm_client import AsyncOpenAICompatClient
from backend.run_execution import ExecutionStore, ExecutionConflict
from backend.paid_transport import paid_scope
from backend import patient_postprocessing as post
from scripts import generate_patient_facing_writeups as writer


@pytest.fixture
def ready(temp_data_dir, monkeypatch):
    monkeypatch.setenv("QEEG_ROUTE_OPENROUTER_EXTRAS_DIRECT", "0")
    monkeypatch.setenv("QEEG_PORTAL_NETLIFY_SYNC_ON_PUBLISH", "0")
    monkeypatch.setenv(
        "QEEG_PORTAL_PATIENTS_DIR", str(temp_data_dir / "portal_patients")
    )
    with storage.session_scope() as session:
        patient = storage.create_patient(session, label="ZZ_01-01-1900", notes="")
        run = storage.Run(
            id="original-run",
            patient_id=patient.id,
            report_id="report",
            status="complete",
            council_model_ids_json='["writer"]',
            consolidator_model_id="writer",
        )
        session.add(run)
        for stage, kind in (
            (2, "peer_review"),
            (3, "revision"),
            (4, "consolidation"),
            (6, "final_draft"),
        ):
            path = temp_data_dir / f"stage-{stage}.md"
            path.write_text(f"Original council source stage {stage}")
            storage.create_artifact(
                session,
                run_id=run.id,
                stage_num=stage,
                stage_name=kind,
                model_id="writer",
                kind=kind,
                content_path=path,
                content_type="text/markdown",
            )
        session.commit()
    cfg = post.snapshot_post_config(
        {
            "QEEG_PATIENT_FACING_MODEL": "writer",
            "QEEG_ROUTE_OPENROUTER_EXTRAS_DIRECT": "0",
        },
        ["writer"],
        base_url="http://mock",
        timeout_s=600.0,
    )
    return ExecutionStore(storage.engine), run.id, cfg


def text():
    return "\n\n".join(
        h + "\nClinical discussion." for h in writer._REQUIRED_PATIENT_FACING_HEADINGS
    )


def llm(sent, *, response=None, catalogue=True):
    def send(request):
        if request.method == "GET":
            if not catalogue:
                raise httpx.ConnectError("catalogue unavailable")
            return httpx.Response(200, json={"data": [{"id": "writer"}]})
        sent.append(request.content)
        if isinstance(response, Exception):
            raise response
        return httpx.Response(
            200,
            json=response
            if response is not None
            else {"choices": [{"message": {"content": text()}}]},
        )

    return AsyncOpenAICompatClient(
        base_url="http://mock",
        api_key="",
        timeout_s=600.0,
        transport=httpx.MockTransport(send),
    )


def admit(ready):
    store, run_id, cfg = ready
    result = post.admit_patient_facing(store, run_id, config_snapshot=cfg)
    assert result["state"] == "pending"
    return store.claim_run_owner(run_id)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "boundary",
    [
        "md",
        "pdf",
        "meta",
        "local",
        "sync",
        "complete",
        "publish_md",
        "publish_pdf",
        "publish_meta",
    ],
)
async def test_each_output_boundary_replays_once(ready, monkeypatch, boundary):
    owner = admit(ready)
    sent = []
    original = post._publish
    failed = False

    def publish(owner, path, data):
        nonlocal failed
        original(owner, path, data)
        if not failed and (
            path.name == boundary + ".json"
            or path.suffix == "." + boundary
            or (boundary == "publish_md" and path.suffix == ".md")
            or (boundary == "publish_pdf" and path.suffix == ".pdf")
            or (boundary == "publish_meta" and path.name.endswith("__meta.json"))
        ):
            failed = True
            raise OSError("death after publication")

    monkeypatch.setattr(post, "_publish", publish)
    client = llm(sent)
    try:
        with pytest.raises(OSError):
            await post.continue_patient_facing(owner, llm_client=client)
    finally:
        owner.close()
    monkeypatch.setattr(post, "_publish", original)
    owner = ready[0].claim_run_owner(ready[1])
    try:
        result = await post.continue_patient_facing(
            owner, llm_client=llm(sent, catalogue=False)
        )
        assert result["verified"] is True
        assert len(sent) == 1
        assert result["delivery_verified"] is False
        assert set(result["outputs"]) == {"md", "pdf", "meta"}
        metadata = json.loads(Path(result["outputs"]["meta"]["path"]).read_text())
        assert metadata["run_id"] == "original-run"
        assert metadata["patient_id"] is not None
        assert metadata["llm_model_id"] == "writer"
        assert all(
            Path(b["path"]).name.startswith("ZZ_01-01-1900")
            for b in result["outputs"].values()
        )
    finally:
        owner.release()


@pytest.mark.asyncio
async def test_pdf_failure_new_run_settings_catalogue_cannot_change_original(
    ready, monkeypatch
):
    owner = admit(ready)
    sent = []
    renderer = writer.render_patient_facing_markdown_to_pdf

    def fail(*a, **kw):
        raise OSError("pdf unavailable")

    monkeypatch.setattr(writer, "render_patient_facing_markdown_to_pdf", fail)
    try:
        with pytest.raises(OSError):
            await post.continue_patient_facing(owner, llm_client=llm(sent))
        with owner.transaction() as session:
            original = session.get(storage.Run, owner.run_id)
            session.add(
                storage.Run(
                    id="newer-run",
                    patient_id=original.patient_id,
                    report_id="report",
                    status="complete",
                    council_model_ids_json="[]",
                    consolidator_model_id="changed",
                )
            )
        monkeypatch.setenv("QEEG_PATIENT_FACING_MODEL", "changed")
        monkeypatch.setenv("QEEG_PATIENT_FACING_AUTO_VERSION_PREFIX", "changed")
        monkeypatch.setattr(writer, "_example_writeup_text", lambda: "changed example")
        monkeypatch.setattr(writer, "render_patient_facing_markdown_to_pdf", renderer)
        result = await post.continue_patient_facing(
            owner, llm_client=llm(sent, catalogue=False)
        )
        assert result["verified"]
        assert len(sent) == 1
        assert b"Original council source" in sent[0]
        assert all("__auto-original__" in b["path"] for b in result["outputs"].values())
    finally:
        owner.release()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "failure", ["unknown", "malformed", "invalid", "missing_receipt", "source_changed"]
)
async def test_unresolved_and_invalid_authority_never_repeat_paid(
    ready, failure, monkeypatch
):
    owner = admit(ready)
    sent = []
    response = (
        httpx.ReadError("unknown")
        if failure == "unknown"
        else {}
        if failure == "malformed"
        else {"choices": [{"message": {"content": "incomplete"}}]}
        if failure == "invalid"
        else None
    )
    if failure in ("missing_receipt", "source_changed"):
        renderer = writer.render_patient_facing_markdown_to_pdf
        monkeypatch.setattr(
            writer,
            "render_patient_facing_markdown_to_pdf",
            lambda *a, **kw: (_ for _ in ()).throw(OSError("pdf failed")),
        )
        with pytest.raises(OSError):
            await post.continue_patient_facing(owner, llm_client=llm(sent))
        monkeypatch.setattr(writer, "render_patient_facing_markdown_to_pdf", renderer)
        if failure == "missing_receipt":
            with owner.transaction() as session:
                row = session.scalar(select(storage.PaidRequest))
            Path(row.response_path).unlink()
        else:
            with owner.transaction() as session:
                row = session.get(
                    storage.PostObligation, (owner.run_id, "patient_facing")
                )
            data = post._load(row.manifest_path)
            Path(data["sources"][0]["content_path"]).write_text("changed")
    try:
        with pytest.raises(Exception):
            await post.continue_patient_facing(
                owner,
                llm_client=llm(
                    sent,
                    response=response,
                    catalogue=failure not in ("missing_receipt", "source_changed"),
                ),
            )
        result = await post.continue_patient_facing(owner, llm_client=llm(sent))
        assert not result["verified"]
        assert len(sent) == 1
    finally:
        owner.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["md", "pdf", "meta", "paid"])
async def test_done_projection_rejects_binding_corruption(ready, kind):
    owner = admit(ready)
    try:
        result = await post.continue_patient_facing(owner, llm_client=llm([]))
        if kind == "paid":
            with owner.transaction() as session:
                paid = session.scalar(select(storage.PaidRequest))
            Path(paid.response_path).unlink()
        else:
            Path(result["outputs"][kind]["path"]).write_bytes(b"corrupt")
        assert not post.project_patient_facing(owner.store, owner.run_id)["verified"]
    finally:
        owner.release()


@pytest.mark.asyncio
async def test_sync_retry_uses_same_outputs_and_receipt(ready):
    ready[2]["sync_enabled"] = True
    owner = admit(ready)
    sent = []
    try:
        with pytest.raises(OSError, match="sync"):
            await post.continue_patient_facing(
                owner, llm_client=llm(sent), sync=lambda label: False
            )
        bindings = post._load(post._root(owner) / "outputs" / "local.json")
        result = await post.continue_patient_facing(
            owner, llm_client=llm(sent, catalogue=False), sync=lambda label: True
        )
        assert result["outputs"] == bindings
        assert result["sync"]["status"] == "handed_off"
        assert len(sent) == 1
    finally:
        owner.release()


def test_explicit_repeated_requests_keep_legacy_attestation(ready):
    owner = admit(ready)
    store, run_id, cfg = ready
    try:
        first = post.project_patient_facing(store, run_id)
        changed = {**cfg, "model_id": "new-model"}
        second = post.admit_patient_facing(store, run_id, config_snapshot=changed)
        assert first == second
        with owner.transaction() as session:
            run = session.get(storage.Run, run_id)
            assert run.analysis_input_fingerprint == ""
            assert run.execution_manifest_hash is None
            assert (
                session.scalar(
                    select(storage.PostObligation).where(
                        storage.PostObligation.run_id == run_id
                    )
                )
                is not None
            )
        with pytest.raises(ExecutionConflict):
            with paid_scope(owner, "s1/member", first["manifest_hash"], "invented"):
                pass
    finally:
        owner.release()


@pytest.mark.parametrize(
    "enabled,missing", [(True, False), (False, False), (True, True), (False, True)]
)
def test_clinical_complete_and_independent_post_dispositions_are_atomic(
    ready, enabled, missing, monkeypatch
):
    from backend.council import completion
    from types import SimpleNamespace

    store, run_id, cfg = ready
    cfg["enabled"] = enabled
    cfg["retired_cathode_flag"] = "1"
    store.request_run_start(run_id)
    owner = store.claim_run_owner(run_id)
    with owner.transaction() as session:
        run = session.get(storage.Run, run_id)
        run.status = "running"
        if missing:
            session.get(storage.Patient, run.patient_id).label = "invalid identity"
    ctx = SimpleNamespace(owner=owner, manifest={"postprocessing": cfg})
    monkeypatch.setattr(completion, "current_execution", lambda: ctx)
    try:
        completion.project_run_status(None, run_id, status="complete")
        with owner.transaction() as session:
            run = session.get(storage.Run, run_id)
            rows = {r.kind: r for r in session.scalars(select(storage.PostObligation))}
        assert run.status == "complete"
        assert rows["patient_facing"].state == (
            "skipped" if not enabled else "blocked" if missing else "pending"
        )
        assert rows["cathode"].state == "skipped"
        assert rows["cathode"].blocked_reason == "manual_fallback"
        assert (
            post._load(rows["cathode"].manifest_path)["diagnostic"]
            == "automatic_cathode_routing_retired"
        )
        completion.project_run_status(None, run_id, status="complete")
        with owner.transaction() as session:
            assert len(list(session.scalars(select(storage.PostObligation)))) == 2
    finally:
        owner.release()


def test_complete_transaction_rolls_back_both_dispositions(ready, monkeypatch):
    from backend.council import completion
    from types import SimpleNamespace

    store, run_id, cfg = ready
    store.request_run_start(run_id)
    owner = store.claim_run_owner(run_id)
    with owner.transaction() as session:
        session.get(storage.Run, run_id).status = "running"
    monkeypatch.setattr(
        completion,
        "current_execution",
        lambda: SimpleNamespace(owner=owner, manifest={"postprocessing": cfg}),
    )
    register = post.register_completion_posts

    def fail(session, owner, prepared):
        register(session, owner, prepared)
        raise OSError("transaction interrupted")

    monkeypatch.setattr(post, "register_completion_posts", fail)
    try:
        with pytest.raises(OSError):
            completion.project_run_status(None, run_id, status="complete")
        with owner.transaction() as session:
            assert session.get(storage.Run, run_id).status == "running"
            assert not list(session.scalars(select(storage.PostObligation)))
        monkeypatch.setattr(post, "register_completion_posts", register)
        completion.project_run_status(None, run_id, status="complete")
        with owner.transaction() as session:
            assert session.get(storage.Run, run_id).status == "complete"
            assert len(list(session.scalars(select(storage.PostObligation)))) == 2
    finally:
        owner.release()


@pytest.mark.asyncio
async def test_unavailable_pinned_model_cannot_select_fallback(ready):
    ready[2]["model_id"] = "unavailable-pinned-model"
    owner = admit(ready)
    sent = []
    try:
        with pytest.raises(ExecutionConflict, match="pinned patient model unavailable"):
            await post.continue_patient_facing(owner, llm_client=llm(sent))
        assert sent == []
        assert (
            post.project_patient_facing(owner.store, owner.run_id)["state"] == "blocked"
        )
    finally:
        owner.close()


def test_concurrent_explicit_admission_rejoins_one_obligation(ready):
    from concurrent.futures import ThreadPoolExecutor

    store, run_id, cfg = ready
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(
            pool.map(
                lambda _: post.admit_patient_facing(store, run_id, config_snapshot=cfg),
                range(4),
            )
        )
    assert {r["state"] for r in results} <= {"pending", "admitting"}
    result = post.admit_patient_facing(
        store, run_id, config_snapshot={**cfg, "model_id": "changed"}
    )
    assert result["state"] == "pending"
    with storage.session_scope() as session:
        rows = list(session.scalars(select(storage.PostObligation)))
        assert len(rows) == 1
        assert post._load(rows[0].manifest_path)["config"]["model_id"] == "writer"


@pytest.mark.asyncio
@pytest.mark.parametrize("death", ["after_response", "during_dispatch"])
async def test_real_process_death_preserves_paid_authority(ready, tmp_path, death):
    import os
    import subprocess
    import sys

    store, run_id, cfg = ready
    post.admit_patient_facing(store, run_id, config_snapshot=cfg)
    marker = tmp_path / "paid-sends.txt"
    program = r"""
import asyncio,os,sys
from pathlib import Path
from backend import storage,patient_postprocessing as post
from backend.run_execution import ExecutionStore
from backend.tests.test_patient_postprocessing import llm
from scripts import generate_patient_facing_writeups as writer
storage.reset_engine('sqlite:///'+sys.argv[1])
store=ExecutionStore(storage.engine)
owner=store.claim_run_owner('original-run')
class Sends(list):
    def append(self,value):
        with open(sys.argv[2],'a') as stream:
            stream.write('sent\n');stream.flush();os.fsync(stream.fileno())
        if sys.argv[3]=='during_dispatch':os._exit(71)
writer.render_patient_facing_markdown_to_pdf=lambda *a,**kw:os._exit(72)
asyncio.run(post.continue_patient_facing(owner,llm_client=llm(Sends())))
"""
    result = subprocess.run(
        [sys.executable, "-c", program, str(store.db_path), str(marker), death],
        env=dict(os.environ),
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == (
        71 if death == "during_dispatch" else 72
    ), result.stderr
    assert marker.read_text() == "sent\n"
    owner = store.claim_run_owner(run_id)
    sent = []
    try:
        if death == "during_dispatch":
            from backend.paid_transport import PaidOutcomeUnknown

            with pytest.raises(PaidOutcomeUnknown):
                await post.continue_patient_facing(
                    owner, llm_client=llm(sent, catalogue=False)
                )
        else:
            assert (
                await post.continue_patient_facing(
                    owner, llm_client=llm(sent, catalogue=False)
                )
            )["verified"]
        assert sent == []
    finally:
        owner.close()


def test_missing_source_becomes_separate_blocked_post(ready):
    store, run_id, cfg = ready
    store.request_run_start(run_id)
    owner = store.claim_run_owner(run_id)
    try:
        with owner.transaction() as session:
            artifact = session.scalar(
                select(storage.Artifact).where(storage.Artifact.stage_num == 6)
            )
        Path(artifact.content_path).unlink()
        prepared = post.prepare_completion_posts(owner, cfg)
        with owner.transaction() as session:
            post.register_completion_posts(session, owner, prepared)
            session.get(storage.Run, run_id).status = "complete"
        assert post.project_patient_facing(store, run_id)["state"] == "blocked"
    finally:
        owner.release()


def test_logo_snapshot_is_task_local_and_survives_setting_changes(ready, monkeypatch):
    from backend import patient_facing_pdf as pdf

    saved = ready[2]["logo_uri"]
    monkeypatch.setenv("QEEG_PATIENT_FACING_LOGO_PATH", "/missing/changed.png")
    with pdf.patient_pdf_assets(saved):
        assert pdf._get_logo_base64() == saved
        with pdf.patient_pdf_assets(""):
            assert pdf._get_logo_base64() == ""
        assert pdf._get_logo_base64() == saved


def test_explicit_admission_can_enroll_complete_execution_without_prior_post(ready):
    store, run_id, cfg = ready
    store.request_run_start(run_id)
    owner = store.claim_run_owner(run_id)
    owner.release(state="done")
    result = post.admit_patient_facing(store, run_id, config_snapshot=cfg)
    assert result["state"] == "pending"
    with storage.session_scope() as session:
        run = session.get(storage.Run, run_id)
        assert run.status == "complete"
        assert run.analysis_input_fingerprint == ""


@pytest.mark.asyncio
async def test_completed_explicit_request_rejoins_original_output(ready):
    owner = admit(ready)
    try:
        original = await post.continue_patient_facing(owner, llm_client=llm([]))
    finally:
        owner.release(state="done")
    result = post.admit_patient_facing(
        ready[0], ready[1], config_snapshot={**ready[2], "model_id": "new"}
    )
    assert result == original


@pytest.mark.asyncio
async def test_acknowledged_endpoint_fallback_recovers_without_catalogue(
    ready, monkeypatch
):
    owner = admit(ready)
    sent = []
    catalogue = True

    def send(request):
        if request.method == "GET":
            if not catalogue:
                raise httpx.ConnectError("catalogue down")
            return httpx.Response(200, json={"data": [{"id": "writer"}]})
        sent.append(request.url.path)
        if request.url.path.endswith("/chat/completions"):
            return httpx.Response(
                400,
                json={
                    "error": {"message": "chat not supported; use responses endpoint"}
                },
            )
        return httpx.Response(200, json={"output_text": text()})

    def client():
        return AsyncOpenAICompatClient(
            base_url="http://mock",
            api_key="",
            timeout_s=600.0,
            transport=httpx.MockTransport(send),
        )

    renderer = writer.render_patient_facing_markdown_to_pdf
    monkeypatch.setattr(
        writer,
        "render_patient_facing_markdown_to_pdf",
        lambda *a, **kw: (_ for _ in ()).throw(OSError("pdf failed")),
    )
    try:
        with pytest.raises(OSError):
            await post.continue_patient_facing(owner, llm_client=client())
        assert len(sent) == 2
        catalogue = False
        monkeypatch.setattr(writer, "render_patient_facing_markdown_to_pdf", renderer)
        assert (await post.continue_patient_facing(owner, llm_client=client()))[
            "verified"
        ]
        assert len(sent) == 2
    finally:
        owner.close()


@pytest.mark.asyncio
async def test_changed_render_dependency_parks_without_new_generation(
    ready, monkeypatch
):
    owner = admit(ready)
    sent = []
    real_version = post.version
    monkeypatch.setattr(
        post,
        "version",
        lambda name: "incompatible" if name == "weasyprint" else real_version(name),
    )
    try:
        with pytest.raises(ExecutionConflict, match="recipe"):
            await post.continue_patient_facing(owner, llm_client=llm(sent))
        assert sent == []
    finally:
        owner.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["render", "sync"])
async def test_post_free_work_allows_independent_coroutine_progress(
    ready, monkeypatch, operation
):
    import asyncio
    import threading
    from backend import patient_facing_pdf

    ready[2]["sync_enabled"] = operation == "sync"
    owner = admit(ready)
    started = threading.Event()
    progressed = threading.Event()
    observations = []

    def blocking_work():
        started.set()
        progressed.wait(timeout=0.2)
        observations.append(progressed.is_set())

    def render(md, path, *, patient_label):
        assert patient_facing_pdf._get_logo_base64() == ready[2]["logo_uri"]
        if operation == "render":
            blocking_work()
        path.write_bytes(b"%PDF-synthetic-owned-output")

    def sync(label):
        blocking_work()
        return True

    async def independent_patient():
        while not started.is_set():
            await asyncio.sleep(0.001)
        progressed.set()

    monkeypatch.setattr(writer, "render_patient_facing_markdown_to_pdf", render)
    monkeypatch.setattr(writer, "sync_patient_to_thrylen", sync)
    try:
        result, _ = await asyncio.gather(
            post.continue_patient_facing(owner, llm_client=llm([])),
            independent_patient(),
        )
        assert result["verified"]
        assert observations == [True], f"{operation} starved another async patient task"
    finally:
        owner.release()


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["render", "sync"])
@pytest.mark.parametrize("worker_error", [False, True])
async def test_cancelled_post_drains_worker_before_process_can_claim_owner(
    ready, monkeypatch, operation, worker_error
):
    import asyncio
    import os
    import subprocess
    import sys
    import threading

    ready[2]["sync_enabled"] = operation == "sync"
    owner = admit(ready)
    started = threading.Event()
    finish = threading.Event()
    settled = threading.Event()

    def blocking_work():
        started.set()
        try:
            assert finish.wait(timeout=5), "test worker was not released"
            if worker_error:
                raise OSError("synthetic free worker failed after cancellation")
        finally:
            settled.set()

    def render(md, path, *, patient_label):
        if operation == "render":
            blocking_work()
        path.write_bytes(b"%PDF-synthetic-owned-output")

    def sync(label):
        blocking_work()
        return True

    probe = """
import sys
from backend import storage
from backend.run_execution import ExecutionStore
storage.reset_engine('sqlite:///'+sys.argv[1])
owner=ExecutionStore(storage.engine).claim_run_owner('original-run')
print('claimed' if owner is not None else 'contended')
if owner is not None:owner.close()
"""

    def contender():
        result = subprocess.run(
            [sys.executable, "-c", probe, str(owner.store.db_path)],
            env=dict(os.environ),
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert result.returncode == 0, result.stderr
        return result.stdout.strip()

    async def owned_call():
        try:
            return await post.continue_patient_facing(owner, llm_client=llm([]))
        finally:
            owner.release()

    monkeypatch.setattr(writer, "render_patient_facing_markdown_to_pdf", render)
    monkeypatch.setattr(writer, "sync_patient_to_thrylen", sync)
    task = asyncio.create_task(owned_call())
    try:
        while not started.is_set() and not task.done():
            await asyncio.sleep(0.001)
        assert (
            started.is_set() and not settled.is_set()
        ), "free worker blocked cancellation delivery"
        task.cancel()
        await asyncio.sleep(0.01)
        task.cancel()
        await asyncio.sleep(0.01)
        assert not task.done()
        assert await asyncio.to_thread(contender) == "contended"
        assert not settled.is_set()
        finish.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert settled.is_set()
        assert await asyncio.to_thread(contender) == "claimed"
    finally:
        finish.set()
        await asyncio.gather(task, return_exceptions=True)
        owner.close()
