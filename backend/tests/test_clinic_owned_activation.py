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
