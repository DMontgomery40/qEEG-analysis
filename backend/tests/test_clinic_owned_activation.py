"""Original E5/E6 handoff never launches unowned or duplicate paid work."""

import asyncio
import pytest
from sqlalchemy import select
from backend import storage, patient_postprocessing as post
from backend.tests.test_patient_postprocessing import ready  # noqa: F401
from backend.tests.clinic_test_helpers import forbid_clinic_paid  # noqa: F401


@pytest.mark.asyncio
async def test_already_entered_legacy_helper_rejoins_competing_original_post(
    ready, monkeypatch
):
    from backend import main

    store, run_id, cfg = ready
    entered = asyncio.Event()
    release = asyncio.Event()

    class Broker:
        async def publish(self, run, payload):
            if payload.get("status") == "start":
                entered.set()
                await release.wait()

    async def forbidden(*args, **kwargs):
        pytest.fail("Unowned subprocess must never launch")

    monkeypatch.setattr(main.asyncio, "create_subprocess_exec", forbidden)
    task = asyncio.create_task(
        main._auto_generate_patient_facing_for_run(run_id, Broker())
    )
    await asyncio.wait_for(entered.wait(), 2)
    original = post.admit_patient_facing(store, run_id, config_snapshot=cfg)
    assert original["state"] == "pending"
    release.set()
    assert await task is False  # queued is not completed
    with storage.session_scope() as s:
        obligations = list(s.scalars(select(storage.PostObligation)))
        assert len(obligations) == 1
        assert obligations[0].run_id == run_id and obligations[0].state == "pending"
    assert post.project_patient_facing(store, run_id) == original


def test_confirmed_original_policy_survives_settings_drift_and_replay(
    temp_data_dir, monkeypatch
):
    from backend import clinic_analysis_intents as intents, config
    from backend.tests.test_clinic_intake import submit
    from backend.council import execution
    from backend.run_execution import ExecutionStore
    import json

    monkeypatch.setenv("QEEG_STAGE6_FINAL_DRAFT_MODEL", "original-writer")
    result = submit(
        file_meta=[{"documentKind": "report"}, {}],
        analysis_intent={
            "operationId": "confirmed-op",
            "confirmed": True,
            "reportItemIndexes": [0],
            "specialInstructions": "Original instructions",
        },
    )["upload"]
    policy = result["analysis"]["policy"]
    monkeypatch.setattr(
        config,
        "DISCOVERED_MODEL_IDS",
        set(
            policy["councilModelIds"]
            + [policy["consolidatorModelId"], "original-writer"]
        ),
    )
    monkeypatch.setenv("QEEG_STAGE6_FINAL_DRAFT_MODEL", "changed-writer")
    run = intents.admit_confirmed_upload(result["uploadId"])
    again = intents.admit_confirmed_upload(result["uploadId"])
    assert run.id == again.id and run.operation_id == "confirmed-op"
    store = ExecutionStore(storage.engine)
    store.request_run_start(run.id)
    owner = store.claim_run_owner(run.id)
    try:
        context = execution.prepare_execution(owner)
        assert (
            context.manifest["settings"]["QEEG_STAGE6_FINAL_DRAFT_MODEL"]
            == "original-writer"
        )
        assert (
            context.manifest["prompts"]
            == intents.confirmed_analysis_binding(result["uploadId"])["policySnapshot"][
                "prompts"
            ]
        )
        assert json.loads(run.source_report_ids_json) == [
            result["items"][0]["sourceId"]
        ]
    finally:
        owner.release()
        owner.close()


def test_shared_execution_retires_old_dispatch_before_remote_reads(
    temp_data_dir, monkeypatch
):
    from backend.portal_sync import spawn_portal_pipeline
    from scripts.portal_pipeline_worker import process_patient

    monkeypatch.setenv("QEEG_CLINIC_SHARED_EXECUTION", "1")

    class NoRemote:
        def __getattr__(self, name):
            pytest.fail(
                "Old metadata/index/dispatch path must retire before remote reads"
            )

    assert spawn_portal_pipeline("ZZ_01-01-1900") is False
    result = process_patient(
        client=NoRemote(),
        portal_dir=temp_data_dir,
        status_dir=temp_data_dir / "status",
        patient_id="ZZ_01-01-1900",
        job_reports=[],
        dry_run=False,
        allow_paid_runs=True,
    )
    assert result.status == "retired" and not result.ran_batch


@pytest.mark.asyncio
@pytest.mark.parametrize("started", [False, True])
async def test_consumer_admitted_upload_avoids_source_readmission(
    temp_data_dir, monkeypatch, started
):
    from types import SimpleNamespace
    from backend import clinic_analysis_intents as intents, config
    from backend.tests.test_clinic_intake import submit
    from backend.run_execution import ExecutionStore

    result = submit(
        file_meta=[{"documentKind": "report"}, {}],
        analysis_intent={
            "operationId": "poll-op",
            "confirmed": True,
            "reportItemIndexes": [0],
            "specialInstructions": "",
        },
    )["upload"]
    policy = result["analysis"]["policy"]
    monkeypatch.setattr(
        config,
        "DISCOVERED_MODEL_IDS",
        set(policy["councilModelIds"] + [policy["consolidatorModelId"]]),
    )
    from backend import main

    monkeypatch.setattr(main, "DISCOVERED_MODEL_IDS", config.DISCOVERED_MODEL_IDS)
    run = intents.admit_confirmed_upload(result["uploadId"])
    store = ExecutionStore(storage.engine)
    if started:
        store.request_run_start(run.id)

    def forbid(*args):
        pytest.fail("Already admitted upload must not rehash/re-admit")

    monkeypatch.setattr(intents, "admit_confirmed_upload", forbid)

    async def admission(fn, *args):
        return fn(*args)

    runtime = SimpleNamespace(store=store, admission=admission)
    await intents.activate_confirmed_uploads(runtime)
    await intents.activate_confirmed_uploads(runtime)
    with storage.session_scope() as s:
        assert s.get(storage.Run, run.id).start_requested_at is not None


@pytest.mark.asyncio
async def test_consumer_retries_admission_and_missing_start_separately(
    temp_data_dir, monkeypatch
):
    from types import SimpleNamespace
    from backend import (
        clinic_analysis_intents as intents,
        config,
        main,
        clinic_catalogue,
    )
    from backend.tests.test_clinic_intake import submit
    from backend.run_execution import ExecutionStore

    result = submit(
        file_meta=[{"documentKind": "report"}, {}],
        analysis_intent={
            "operationId": "retry-poll",
            "confirmed": True,
            "reportItemIndexes": [0],
            "specialInstructions": "",
        },
    )["upload"]
    policy = result["analysis"]["policy"]
    monkeypatch.setattr(
        config,
        "DISCOVERED_MODEL_IDS",
        set(policy["councilModelIds"] + [policy["consolidatorModelId"]]),
    )
    monkeypatch.setattr(main, "DISCOVERED_MODEL_IDS", config.DISCOVERED_MODEL_IDS)
    counts = {"admit": 0, "start": 0, "source": 0}
    original = clinic_catalogue._read_local

    def read(*args, **kwargs):
        counts["source"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(clinic_catalogue, "_read_local", read)

    async def admission(fn, *args):
        which = (
            "admit"
            if fn is intents.admit_confirmed_upload
            else "start"
            if fn is main._new_start_intent
            else None
        )
        if which:
            counts[which] += 1
            if counts[which] == 1:
                raise RuntimeError("transient interruption")
        return fn(*args)

    runtime = SimpleNamespace(store=ExecutionStore(storage.engine), admission=admission)
    for _ in range(4):
        await intents.activate_confirmed_uploads(runtime)
    assert counts["admit"] == 2 and counts["start"] == 2 and counts["source"] == 1
    with storage.session_scope() as s:
        run = s.scalar(
            select(storage.Run).where(storage.Run.operation_id == "retry-poll")
        )
        assert run.start_requested_at is not None
        # Historical completed records without a start timestamp are never re-admitted.
        run.status = "complete"
        run.start_requested_at = None
        s.commit()

    async def forbidden(*args):
        pytest.fail("Completed historical run must be skipped")

    runtime.admission = forbidden
    await intents.activate_confirmed_uploads(runtime)
