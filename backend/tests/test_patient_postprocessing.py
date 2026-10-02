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
        # Retired Cathode routing leaves no obligation and no manifest behind.
        assert set(rows) == {"patient_facing"}
        assert not (post._root(owner) / "cathode.json").exists()
        completion.project_run_status(None, run_id, status="complete")
        with owner.transaction() as session:
            assert len(list(session.scalars(select(storage.PostObligation)))) == 1
    finally:
        owner.release()


def test_complete_transaction_rolls_back_status_and_post_together(ready, monkeypatch):
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
            assert len(list(session.scalars(select(storage.PostObligation)))) == 1
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
    assert result.returncode == (71 if death == "during_dispatch" else 72), (
        result.stderr
    )
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


def _deploy_edits_a_recipe_file(monkeypatch):
    """A deploy between admission and generation changes a post recipe file."""
    real = post._recipe

    def edited():
        recipe = real()
        files = {**recipe["files"], "backend/patient_facing_pdf.py": "edited-on-disk"}
        return {**recipe, "files": files}

    monkeypatch.setattr(post, "_recipe", edited)


def _die_after_markdown_is_bound(monkeypatch):
    original = post._publish

    def publish(owner, path, data):
        original(owner, path, data)
        if path.name == "md.json":
            raise OSError("death after the markdown was bound")

    monkeypatch.setattr(post, "_publish", publish)
    return original


@pytest.mark.asyncio
async def test_a_deploy_before_the_write_up_is_paid_adopts_the_new_code(
    ready, monkeypatch
):
    """EN-H8: three live write-ups sat blocked because a deploy changed a
    recipe file after admission, and only a paid regeneration recovered them.
    Nothing was paid yet, so the write-up runs on the code now on disk and makes
    its one authorized generation."""
    owner = admit(ready)
    _deploy_edits_a_recipe_file(monkeypatch)
    sent = []
    try:
        result = await post.continue_patient_facing(owner, llm_client=llm(sent))
    finally:
        owner.release()
    assert result["state"] == "done" and result["verified"] is True, result
    assert len(sent) == 1
    assert post.project_patient_facing(ready[0], ready[1])["verified"] is True
    (record,) = Path(result["manifest_path"]).parent.glob("recipe-adopted-*.json")
    assert json.loads(record.read_text())["adopted"] == post._recipe()


@pytest.mark.asyncio
async def test_a_deploy_after_the_paid_answer_finishes_from_the_saved_answer(
    ready, monkeypatch
):
    """The answer was paid for and its markdown bound before the process died;
    then a deploy changed the writer. The saved answer finishes the write-up
    with no new send, and the PDF is made from the markdown that was bound, not
    from what the new writer code would make of the answer."""
    owner = admit(ready)
    sent = []
    original = _die_after_markdown_is_bound(monkeypatch)
    try:
        with pytest.raises(OSError):
            await post.continue_patient_facing(owner, llm_client=llm(sent))
    finally:
        owner.close()
    monkeypatch.setattr(post, "_publish", original)
    _deploy_edits_a_recipe_file(monkeypatch)
    real_chat = writer._chat_with_retries

    async def new_writer_code(*args, **kwargs):
        return (await real_chat(*args, **kwargs)) + "\n\nAdded by the new writer."

    rendered = []
    real_render = writer.render_patient_facing_markdown_to_pdf

    def render(md, path, **kwargs):
        rendered.append(md)
        return real_render(md, path, **kwargs)

    monkeypatch.setattr(writer, "_chat_with_retries", new_writer_code)
    monkeypatch.setattr(writer, "render_patient_facing_markdown_to_pdf", render)
    owner = ready[0].claim_run_owner(ready[1])
    try:
        result = await post.continue_patient_facing(
            owner, llm_client=llm(sent, catalogue=False)
        )
    finally:
        owner.release()
    assert result["state"] == "done" and result["verified"] is True, result
    assert len(sent) == 1
    bound = Path(result["outputs"]["md"]["path"]).read_text()
    assert "new writer" not in bound
    assert rendered == [bound.removesuffix("\n")]
    assert post.project_patient_facing(ready[0], ready[1])["verified"] is True


@pytest.mark.asyncio
@pytest.mark.parametrize("unsettled", ["prepared", "dispatched", "unknown"])
async def test_a_deploy_never_steps_over_an_unsettled_write_up_call(
    ready, monkeypatch, unsettled
):
    owner = admit(ready)
    sent = []
    original = _die_after_markdown_is_bound(monkeypatch)
    try:
        with pytest.raises(OSError):
            await post.continue_patient_facing(owner, llm_client=llm(sent))
    finally:
        owner.close()
    monkeypatch.setattr(post, "_publish", original)
    with storage.session_scope() as session:
        session.scalar(select(storage.PaidRequest)).state = unsettled
        session.commit()
    _deploy_edits_a_recipe_file(monkeypatch)
    owner = ready[0].claim_run_owner(ready[1])
    try:
        with pytest.raises(ExecutionConflict, match="recipe"):
            await post.continue_patient_facing(owner, llm_client=llm(sent))
    finally:
        owner.close()
    assert len(sent) == 1
    assert _post_row(ready[1])[0] == "blocked"
    root = post._root(owner)
    assert not list(root.glob("recipe-adopted-*.json"))


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
        assert started.is_set() and not settled.is_set(), (
            "free worker blocked cancellation delivery"
        )
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


@pytest.mark.asyncio
@pytest.mark.parametrize("damage", [None, "manifest", "other_block", "receipt_commit"])
async def test_reconcile_saved_post_billing_rejection_uses_post_admission(
    ready, monkeypatch, damage
):
    from backend import paid_transport as paid

    owner = admit(ready)
    store, run_id, _ = ready
    sent = []
    client = llm(sent)

    def send(request):
        if request.method == "GET":
            return httpx.Response(200, json={"data": [{"id": "writer"}]})
        sent.append(request.content)
        return httpx.Response(402, json={"error": {"message": "Insufficient Balance"}})

    client._transport = httpx.MockTransport(send)
    original = paid._rejection
    with monkeypatch.context() as old:
        old.setattr(
            paid,
            "_rejection",
            lambda status, body: None if status == 402 else original(status, body),
        )
        with pytest.raises(paid.PaidOutcomeUnknown):
            await post.continue_patient_facing(owner, llm_client=client)
    with owner.transaction() as session:
        obligation = session.get(storage.PostObligation, (run_id, "patient_facing"))
        manifest_path, manifest_hash = (
            obligation.manifest_path,
            obligation.manifest_hash,
        )
        request = session.scalar(select(storage.PaidRequest))
        binding = (
            request.request_hash,
            request.response_hash,
            request.dispatch_ordinal,
        )
        if damage == "other_block":
            obligation.blocked_reason = "original source changed"
    if damage == "manifest":
        Path(manifest_path).write_text("{}")
    owner.release(state="blocked", blocked_reason="paid_outcome_unknown")
    invalid = damage in ("manifest", "other_block")
    if damage == "receipt_commit":
        reconcile = paid._Receipt.reconcile

        def interrupted(receipt):
            reconcile(receipt)
            raise OSError("interrupted after receipt commit")

        with monkeypatch.context() as failure:
            failure.setattr(paid._Receipt, "reconcile", interrupted)
            with pytest.raises(OSError):
                paid.reconcile_blocked_run(store, run_id)
    try:
        if invalid:
            with pytest.raises(ExecutionConflict):
                paid.reconcile_blocked_run(store, run_id)
        else:
            assert paid.reconcile_blocked_run(store, run_id)
        with storage.session_scope() as session:
            obligation = session.get(storage.PostObligation, (run_id, "patient_facing"))
            assert obligation.state == ("blocked" if invalid else "pending")
            assert (obligation.manifest_path, obligation.manifest_hash) == (
                manifest_path,
                manifest_hash,
            )
            request = session.scalar(select(storage.PaidRequest))
            assert (
                request.request_hash,
                request.response_hash,
                request.dispatch_ordinal,
            ) == binding
            assert request.state == ("unknown" if invalid else "rejected")
            run = session.get(storage.Run, run_id)
            assert run.execution_state == ("blocked" if invalid else "pending")
            assert run.execution_manifest_hash is None
        assert len(sent) == 1
    finally:
        owner.close()
        await client.aclose()


@pytest.mark.asyncio
async def test_relabel_after_completion_files_the_document_under_the_new_id(
    ready, temp_data_dir
):
    """EN-H7 + EN-H15: the clinic corrected the id between council completion
    and generation. The document used to block on "original patient identity
    changed". It is the same patient (same UUID), so the files land under the
    current id, the meta names the clinic id, and the paid prompt is unchanged."""
    ready[2]["sync_enabled"] = True
    owner = admit(ready)
    from sqlalchemy.orm import Session

    with Session(storage.engine) as session:
        patient = session.get(storage.Patient, session.get(storage.Run, owner.run_id).patient_id)
        uuid = patient.id
        patient.label = "ZA_01-01-1900"
        session.commit()
    sent, synced = [], []
    try:
        result = await post.continue_patient_facing(
            owner,
            llm_client=llm(sent),
            sync=lambda label: synced.append(label) or True,
        )
    finally:
        owner.release()
    assert result["state"] == "done", result
    folder = temp_data_dir / "portal_patients"
    old = folder / "ZZ_01-01-1900"
    assert not old.exists() or not list(old.glob("*__meta.json")), list(old.iterdir())
    written = sorted(p.name for p in (folder / "ZA_01-01-1900").iterdir())
    assert len(written) == 3 and all(n.startswith("ZA_01-01-1900") for n in written), written
    meta = json.loads(next((folder / "ZA_01-01-1900").glob("*__meta.json")).read_text())
    assert meta["patient_id"] == meta["patient_label"] == "ZA_01-01-1900"
    assert uuid not in json.dumps(meta)
    assert synced == ["ZA_01-01-1900"]
    # One paid send, of the prompt as admitted: the relabel never rewrites it.
    assert len(sent) == 1 and b"ZA_01-01-1900" not in sent[0]
    # The finished document verifies: the projection used to compare the
    # outputs with the pinned old-id paths, so it read "local output
    # destinations changed" and the runtime failed the done run.
    assert result["verified"] and result["local_complete"], result
    # Only the id may differ from what was pinned: an output filed under any
    # other name still fails, even when its bytes check out.
    local_path = Path(result["manifest_path"]).parent / "outputs" / "local.json"
    local = json.loads(local_path.read_text())
    moved = Path(local["md"]["path"]).with_name("ZA_01-01-1900-other.md")
    moved.write_bytes(Path(local["md"]["path"]).read_bytes())
    local["md"]["path"] = str(moved)
    local_path.write_text(json.dumps(local))
    tampered = post.project_patient_facing(owner.store, owner.run_id)
    assert tampered["integrity_error"] == "local output destinations changed", tampered


@pytest.mark.asyncio
async def test_a_rekey_after_the_write_up_keeps_it_verified_under_the_new_id(
    ready, temp_data_dir
):
    """The rekey renames a finished write-up's files under the patient's new id
    and leaves their bytes alone, but the receipts keep the old paths, so the
    projection read "required output unavailable" for finished work."""
    from sqlalchemy.orm import Session
    from backend import patient_rekey

    ready[2]["sync_enabled"] = True
    owner = admit(ready)
    try:
        result = await post.continue_patient_facing(
            owner, llm_client=llm([]), sync=lambda label: True
        )
    finally:
        owner.release()
    assert result["verified"], result
    folder = temp_data_dir / "portal_patients"
    with Session(storage.engine) as session:
        uuid = session.get(storage.Run, owner.run_id).patient_id
        plan = patient_rekey.plan_patient_rekey(
            "ZZ_01-01-1900", "ZA_01-01-1900", portal_root=folder
        )
        patient_rekey.apply_patient_rekey(plan, session=session, patient_uuid=uuid)
    assert not (folder / "ZZ_01-01-1900").exists()
    moved = post.project_patient_facing(owner.store, owner.run_id)
    assert moved["verified"] and moved["local_complete"], moved
    # Renamed is fine; changed bytes are not.
    pdf = next((folder / "ZA_01-01-1900").glob("*.pdf"))
    pdf.write_bytes(pdf.read_bytes() + b"%")
    changed = post.project_patient_facing(owner.store, owner.run_id)
    assert changed["integrity_error"] == "required output binding changed", changed


# EN-H3: a blocked write-up used to be final; the only way to get it was a whole
# new paid council run. An explicit request now starts its next attempt.
INVALID = {"choices": [{"message": {"content": "incomplete"}}]}


async def _block_attempt(owner, sent, response=INVALID):
    """Drive one attempt to a blocked write-up and park the run as the runtime does."""
    with pytest.raises(Exception):
        await post.continue_patient_facing(
            owner, llm_client=llm(sent, response=response)
        )
    with owner.transaction() as session:
        row = session.get(storage.PostObligation, (owner.run_id, "patient_facing"))
        assert row.state == "blocked"
        reason = row.blocked_reason
    owner.release(
        state="blocked", blocked_reason="patient-facing document blocked: " + reason
    )


def _post_row(run_id):
    with storage.session_scope() as session:
        row = session.get(storage.PostObligation, (run_id, "patient_facing"))
        return row.state, row.manifest_path, row.manifest_hash, row.blocked_reason


def _paid_rows(run_id):
    with storage.session_scope() as session:
        return {
            (r.scope_key, r.dispatch_ordinal): (r.state, r.request_hash, r.response_hash)
            for r in session.scalars(
                select(storage.PaidRequest).where(storage.PaidRequest.run_id == run_id)
            )
        }


@pytest.mark.asyncio
async def test_regenerate_blocked_write_up_asks_again_without_a_new_council(ready):
    store, run_id, cfg = ready
    sent = []
    await _block_attempt(admit(ready), sent)
    _, first_path, first_hash, _ = _post_row(run_id)
    first = post._load(first_path, first_hash)
    first_bytes = Path(first_path).read_bytes()

    result = post.admit_patient_facing(
        store, run_id, config_snapshot=cfg, regenerate=True
    )

    assert result["state"] == "pending" and result["blocked_reason"] is None
    second = post._load(result["manifest_path"], result["manifest_hash"])
    assert second["attempt"] == 2 and second["prompt"] == first["prompt"]
    assert Path(result["manifest_path"]).parent.name == "attempt-2"
    assert Path(first_path).read_bytes() == first_bytes
    for kind, ext in (("md", ".md"), ("pdf", ".pdf"), ("meta", "__meta.json")):
        old_name = first["destinations"][kind]
        assert second["destinations"][kind] == old_name.removesuffix(ext) + "-2" + ext
        assert not Path(second["destinations"][kind]).exists()
    with storage.session_scope() as session:
        run = session.get(storage.Run, run_id)
        assert (run.execution_state, run.blocked_reason) == ("pending", None)

    # The ordinary consumer picks it up: one more paid send, byte-identical.
    owner = store.claim_run_owner(run_id)
    assert owner is not None
    try:
        done = await post.continue_patient_facing(owner, llm_client=llm(sent))
    finally:
        owner.release(state="done")
    assert done["state"] == "done" and done["verified"] is True, done
    assert len(sent) == 2 and sent[1] == sent[0]
    paid = _paid_rows(run_id)
    assert set(paid) == {
        ("post/patient_facing/generation", 0),
        ("post/patient_facing/2", 0),
    }
    assert (
        paid[("post/patient_facing/2", 0)][1]
        == paid[("post/patient_facing/generation", 0)][1]
    )
    assert {k: b["path"] for k, b in done["outputs"].items()} == second["destinations"]
    # A finished write-up rejoins; asking again never spends again.
    again = post.admit_patient_facing(
        store, run_id, config_snapshot=cfg, regenerate=True
    )
    assert again["state"] == "done" and len(sent) == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("unsettled", ["unknown_send", "prepared", "dispatched"])
async def test_regenerate_never_steps_over_an_unsettled_paid_call(ready, unsettled):
    from types import SimpleNamespace

    store, run_id, cfg = ready
    sent = []
    await _block_attempt(
        admit(ready),
        sent,
        response=httpx.ReadError("lost") if unsettled == "unknown_send" else INVALID,
    )
    if unsettled != "unknown_send":
        with storage.session_scope() as session:
            session.scalar(select(storage.PaidRequest)).state = unsettled
            session.commit()
    assert _paid_rows(run_id)[("post/patient_facing/generation", 0)][0] == (
        "unknown" if unsettled == "unknown_send" else unsettled
    )
    before = _post_row(run_id)
    assert before[0] == "blocked"

    with pytest.raises(ExecutionConflict, match="reconcile it first"):
        post.admit_patient_facing(store, run_id, config_snapshot=cfg, regenerate=True)

    assert _post_row(run_id) == before
    root = post._root(SimpleNamespace(store=store, run_id=run_id))
    assert not (root / "attempt-2").exists()
    assert len(sent) == 1


@pytest.mark.asyncio
async def test_second_regeneration_takes_attempt_three(ready):
    store, run_id, cfg = ready
    sent = []
    await _block_attempt(admit(ready), sent)
    post.admit_patient_facing(store, run_id, config_snapshot=cfg, regenerate=True)
    await _block_attempt(store.claim_run_owner(run_id), sent)

    result = post.admit_patient_facing(
        store, run_id, config_snapshot=cfg, regenerate=True
    )

    third = post._load(result["manifest_path"], result["manifest_hash"])
    assert result["state"] == "pending" and third["attempt"] == 3
    assert Path(result["manifest_path"]).parent.name == "attempt-3"
    for kind, ending in (("md", "-3.md"), ("pdf", "-3.pdf"), ("meta", "-3__meta.json")):
        assert third["destinations"][kind].endswith(ending), third["destinations"]
    owner = store.claim_run_owner(run_id)
    try:
        done = await post.continue_patient_facing(owner, llm_client=llm(sent))
    finally:
        owner.release(state="done")
    assert done["verified"] is True
    assert {key[0] for key in _paid_rows(run_id)} == {
        "post/patient_facing/generation",
        "post/patient_facing/2",
        "post/patient_facing/3",
    }
    assert len(sent) == 3 and sent[0] == sent[1] == sent[2]


@pytest.mark.asyncio
async def test_reconcile_script_skips_settled_rows_of_earlier_attempts(
    ready, monkeypatch
):
    """The paid-run reconcile script re-checked every post row against the
    current manifest, so a regenerated write-up whose new attempt ended unknown
    could never be reconciled: the first attempt's row failed that check."""
    import subprocess
    import sys

    from backend import paid_transport as paid

    store, run_id, cfg = ready
    sent = []
    await _block_attempt(admit(ready), sent)
    post.admit_patient_facing(store, run_id, config_snapshot=cfg, regenerate=True)
    first_row = _paid_rows(run_id)[("post/patient_facing/generation", 0)]

    def billing(request):
        if request.method == "GET":
            return httpx.Response(200, json={"data": [{"id": "writer"}]})
        sent.append(request.content)
        return httpx.Response(402, json={"error": {"message": "Insufficient Balance"}})

    client = llm(sent)
    client._transport = httpx.MockTransport(billing)
    original = paid._rejection
    owner = store.claim_run_owner(run_id)
    with monkeypatch.context() as old:
        old.setattr(
            paid,
            "_rejection",
            lambda status, body: None if status == 402 else original(status, body),
        )
        with pytest.raises(paid.PaidOutcomeUnknown):
            await post.continue_patient_facing(owner, llm_client=client)
    owner.release(state="blocked", blocked_reason="paid_outcome_unknown")
    await client.aclose()
    assert _paid_rows(run_id)[("post/patient_facing/2", 0)][0] == "unknown"

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "backend.scripts.reconcile_paid_run",
            "--data-dir",
            str(Path(store.engine.url.database).parent),
            run_id,
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert result.returncode == 0, result.stderr
    assert "Saved receipts reconciled" in result.stdout
    rows = _paid_rows(run_id)
    assert rows[("post/patient_facing/generation", 0)] == first_row
    assert rows[("post/patient_facing/2", 0)][0] == "rejected"
    assert _post_row(run_id)[0] == "pending"
    assert len(sent) == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("case", ["asked", "unknown", "not_asked"])
async def test_regenerate_action_reopens_a_blocked_write_up(
    ready, temp_data_dir, monkeypatch, case
):
    """The action used to answer a blocked write-up with scheduled: false and
    nothing else; it only admitted a post not yet made. Another attempt spends,
    so it happens only when the request carries the clinic's explicit yes."""
    unknown = case == "unknown"
    from unittest.mock import AsyncMock

    from fastapi.testclient import TestClient

    from backend import main
    from backend.run_runtime import RunRuntime

    store, run_id, _ = ready
    sent = []
    await _block_attempt(
        admit(ready),
        sent,
        response=httpx.ReadError("lost") if unknown else INVALID,
    )
    with storage.session_scope() as session:
        patient_id = session.get(storage.Run, run_id).patient_id
    monkeypatch.setenv("QEEG_MOCK_LLM", "1")
    monkeypatch.setattr(
        main, "_ensure_project_clipr_config", lambda: temp_data_dir / "clipr.conf"
    )
    monkeypatch.setattr(main, "_sync_home_auth_to_project", lambda: 0)
    monkeypatch.setattr(RunRuntime, "start", AsyncMock())
    with TestClient(main.app, raise_server_exceptions=False) as client:
        response = client.post(
            f"/api/patients/{patient_id}/actions/regenerate_patient_facing",
            json={"run_id": run_id, "regenerate_blocked": case != "not_asked"},
        )
    if case == "not_asked":
        assert response.status_code == 200, response.text
        assert response.json()["scheduled"] is False
        assert response.json()["postprocessing"]["state"] == "blocked"
        assert _post_row(run_id)[0] == "blocked"
    elif unknown:
        assert response.status_code == 409, response.text
        assert "reconcile it first" in response.text
        assert _post_row(run_id)[0] == "blocked"
    else:
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["scheduled"] is True
        assert body["postprocessing"]["state"] == "pending"
        assert Path(body["postprocessing"]["manifest_path"]).parent.name == "attempt-2"
    assert len(sent) == 1
