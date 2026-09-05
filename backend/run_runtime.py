"""Continuing host-local consumer of the engine's existing run/post intentions.

The SQLite intent and actual flock are authority. Task references are retained
through cleanup; stop drains work naturally, including SDK worker threads.
"""

from __future__ import annotations

import asyncio
from collections import Counter
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from functools import partial
import json

from sqlalchemy import select
from sqlalchemy.orm import Session

from . import storage
from .council.execution import drain_task, owned_execution
from .logging_utils import get_logger, log_context
from .paid_transport import PaidOutcomeUnknown, dispatch_validation
from .run_execution import ExecutionConflict

LOGGER = get_logger(__name__)


class ModelUnavailable(ExecutionConflict):
    """No dispatch occurred; retry the original model when discovery recovers."""


class AdmissionUnavailable(RuntimeError):
    """Shutdown or bounded contention prevented a durable admission."""


def compatibility_reason(run, *, has_post=False):
    if run.start_requested_at is not None:
        if run.execution_manifest_hash or has_post:
            return "receipt_covered_intent"
        if run.status == "created" and run.analysis_input_fingerprint:
            return "explicit_start_pending"
        if run.status == "complete":
            return "explicit_post_admission_incomplete"
        return "legacy_reconciliation_required"
    if run.status == "complete":
        return "legacy_complete_no_post_intent"
    if run.status == "created":
        return (
            "created_awaiting_explicit_start"
            if run.analysis_input_fingerprint
            else "legacy_created_missing_source_admission"
        )
    return "legacy_reconciliation_required"


def compatibility_inventory(store):
    """Read only; neither artifacts, age nor raw status create permission to spend."""
    with Session(store.engine) as session:
        post_ids = set(session.scalars(select(storage.PostObligation.run_id)))
        records = [
            dict(
                run_id=run.id,
                clinical_status=run.status,
                reason=compatibility_reason(run, has_post=run.id in post_ids),
            )
            for run in session.scalars(select(storage.Run).order_by(storage.Run.id))
        ]
    return {
        "counts": dict(Counter(row["reason"] for row in records)),
        "records": records,
    }


def _validate_dispatch(request):
    from .config import DISCOVERED_MODEL_IDS
    from .paid_transport import current_paid_scope

    scope = current_paid_scope()
    if scope is not None and scope.semantic_key.startswith("post/"):
        # E5 independently discovers the pinned post model before fresh generation.
        return
    try:
        model = json.loads(request.content)["model"]
    except (ValueError, KeyError, TypeError) as error:
        raise ExecutionConflict("paid request has no pinned model") from error
    if model not in DISCOVERED_MODEL_IDS:
        raise ModelUnavailable(f"Pinned model is currently unavailable: {model}")


@contextmanager
def current_catalogue_guard():
    token = dispatch_validation.set(_validate_dispatch)
    try:
        yield
    finally:
        dispatch_validation.reset(token)


def _read_run(store, run_id):
    with Session(store.engine, expire_on_commit=False) as session:
        return session.get(storage.Run, run_id)


def _post_states(owner):
    with owner.transaction() as session:
        return [
            (p.kind, p.state, p.blocked_reason)
            for p in session.scalars(
                select(storage.PostObligation).where(
                    storage.PostObligation.run_id == owner.run_id
                )
            )
        ]


async def continue_owned_run(owner, *, llm, workflow, publish=None, sync=None):
    """Drive original six stages, then the independent admitted patient outputs."""
    from .council.completion import verified_stage_prefix
    from .patient_postprocessing import continue_patient_facing, project_patient_facing

    async def observe(payload):
        if publish is not None:
            try:
                await publish(owner.run_id, payload)
            except Exception:
                LOGGER.exception("run_observer_failed", run_id=owner.run_id)

    run = await asyncio.to_thread(_read_run, owner.store, owner.run_id)
    posts = await asyncio.to_thread(_post_states, owner)
    with log_context(run_id=run.id, patient_id=run.patient_id, report_id=run.report_id):
        with current_catalogue_guard():
            if run.status != "complete" or run.execution_manifest_hash:
                if not run.execution_manifest_hash and (
                    run.status != "created" or not run.analysis_input_fingerprint
                ):
                    raise ExecutionConflict("legacy_reconciliation_required")
                if not run.execution_manifest_hash:
                    from .config import DISCOVERED_MODEL_IDS

                    required = json.loads(run.council_model_ids_json) + [
                        run.consolidator_model_id
                    ]
                    if any(model not in DISCOVERED_MODEL_IDS for model in required):
                        raise ModelUnavailable(
                            "Pinned council catalogue is unavailable before first execution"
                        )
                async with owned_execution(owner, llm_client=llm) as execution:
                    if execution.manifest["postprocessing"]["retired_cathode_flag"]:
                        LOGGER.warning(
                            "automatic_cathode_routing_retired", run_id=run.id
                        )
                        await observe(
                            {
                                "run_id": run.id,
                                "diagnostic": "automatic_cathode_routing_retired",
                            }
                        )
                    if run.status != "complete":
                        await workflow.run_pipeline(
                            run.id, on_event=observe, propagate_owned_errors=True
                        )
                    # A returned coroutine or a projected complete row is not proof.
                    if verified_stage_prefix() != 6:
                        raise RuntimeError(
                            "Council continuation lacks six verified stage receipts"
                        )
                run = await asyncio.to_thread(_read_run, owner.store, run.id)
                if run.status != "complete":
                    raise RuntimeError("Council receipts await complete projection")
                posts = await asyncio.to_thread(_post_states, owner)
            if not posts:
                # E5 can commit start before acquiring its post-only admission lock.
                # Retain retryability without inventing a config/obligation.
                return "pending", "explicit_post_admission_incomplete"
            for kind, state, reason in posts:
                if kind == "patient_facing" and state in ("pending", "owned"):
                    await continue_patient_facing(owner, llm_client=llm, sync=sync)
            posts = await asyncio.to_thread(_post_states, owner)
            projection = await asyncio.to_thread(
                project_patient_facing, owner.store, run.id
            )
            await observe(
                {
                    "run_id": run.id,
                    "status": run.status,
                    "execution_state": "owned",
                    "patient_facing": projection,
                }
            )
            blocked = [
                reason or kind for kind, state, reason in posts if state == "blocked"
            ]
            if blocked:
                return "blocked", "; ".join(blocked)
            if any(state not in ("done", "skipped") for _, state, _ in posts):
                return "pending", "postprocessing_pending"
            if projection["state"] == "done" and not projection["verified"]:
                raise ExecutionConflict("completed patient outputs failed verification")
            return "done", None


def _failure_disposition(error):
    from .council.completion import is_clinical_failure

    seen = set()
    chain = []
    while error is not None and id(error) not in seen:
        seen.add(id(error))
        chain.append(error)
        error = error.__cause__ or error.__context__
    if any(isinstance(e, PaidOutcomeUnknown) for e in chain):
        return "blocked", "paid_outcome_unknown"
    if any(isinstance(e, ModelUnavailable) for e in chain):
        return "pending", str(chain[0])
    if any(isinstance(e, ExecutionConflict) for e in chain):
        if any(str(e).startswith("completion persistence failed;") for e in chain):
            return "pending", str(chain[0])
        return "blocked", str(chain[0])
    if is_clinical_failure(chain[0]):
        return "blocked", "clinical_policy_exhausted: " + str(chain[0])
    return "pending", str(chain[0])


class RunRuntime:
    """Finite fair scans, bounded active owners and bounded retry intervals.

    continuation injection is for local synthetic tests; production uses the
    original owned pipeline. There is no second ledger, lease or patient queue.
    """

    def __init__(
        self,
        store,
        *,
        llm=None,
        workflow=None,
        publish=None,
        sync=None,
        continuation=None,
        concurrency=2,
        poll_interval=1.0,
        retry_delay=5.0,
        page_size=100,
    ):
        if not 1 <= concurrency <= 16 or not 1 <= page_size <= 1000:
            raise ValueError("invalid consumer bounds")
        if not 0 < poll_interval <= 60 or not 0 < retry_delay <= 300:
            raise ValueError("invalid consumer intervals")
        self.store = store
        self.publish = publish
        self.continuation = continuation or partial(
            continue_owned_run, llm=llm, workflow=workflow, publish=publish, sync=sync
        )
        self.concurrency, self.poll_interval = concurrency, poll_interval
        self.retry_delay, self.page_size = retry_delay, page_size
        self.tasks = {}
        self.admissions = set()
        self._wake = asyncio.Event()
        self._stopping = False
        self._scan_task = None
        self.inventory = None
        self._cursor = None
        self._retry_after = {}

    async def start(self):
        if self._scan_task is not None:
            return
        try:
            self.inventory = await asyncio.to_thread(
                compatibility_inventory, self.store
            )
            LOGGER.info("run_compatibility_inventory", counts=self.inventory["counts"])
        except Exception:
            LOGGER.exception("run_compatibility_inventory_failed")
            self.inventory = {"error": "inventory_unavailable"}
        self._scan_task = asyncio.create_task(self._scan(), name="qeeg-run-consumer")
        self.wake()

    def wake(self):
        self._wake.set()

    async def _scan(self):
        while not self._stopping:
            self._wake.clear()
            from .clinic_execution_cutover import shared_execution_enabled

            if shared_execution_enabled():
                from .clinic_analysis_intents import activate_confirmed_uploads

                await activate_confirmed_uploads(self)
            now = asyncio.get_running_loop().time()
            self._retry_after = {
                key: due for key, due in self._retry_after.items() if due > now
            }
            for run_id, task in list(self.tasks.items()):
                if task.done():
                    try:
                        task.result()
                    except BaseException:
                        LOGGER.exception("run_consumer_task_failed", run_id=run_id)
                    del self.tasks[run_id]
            if len(self.tasks) < self.concurrency:
                try:
                    rows = await asyncio.to_thread(
                        self.store.list_due_runs,
                        datetime.now(timezone.utc),
                        self.page_size,
                        self._cursor,
                    )
                    if not rows:
                        self._cursor = None
                    for row in rows:
                        if self._stopping or len(self.tasks) >= self.concurrency:
                            break
                        self._cursor = row.id
                        if (
                            row.id in self.tasks
                            or self._retry_after.get(row.id, 0)
                            > asyncio.get_running_loop().time()
                        ):
                            continue
                        self.tasks[row.id] = asyncio.create_task(
                            self._work(row.id), name=f"qeeg-run-{row.id}"
                        )
                except Exception:
                    LOGGER.exception("run_consumer_scan_failed")
            try:
                await asyncio.wait_for(self._wake.wait(), timeout=self.poll_interval)
            except asyncio.TimeoutError:
                pass

    async def _work(self, run_id):
        # Even direct cancellation of a lifecycle worker retains its actual task.
        task = asyncio.create_task(self._run_owned(run_id))
        await drain_task(task)

    async def _run_owned(self, run_id):
        owner = None
        try:
            owner = await asyncio.to_thread(self.store.claim_run_owner, run_id)
            if owner is None:
                return
            try:
                state, reason = await self.continuation(owner)
            except Exception as error:
                LOGGER.exception("run_continuation_interrupted", run_id=run_id)
                state, reason = _failure_disposition(error)
            await asyncio.to_thread(
                owner.release,
                state=state,
                next_check_at=datetime.now(timezone.utc)
                + timedelta(seconds=self.retry_delay)
                if state == "pending"
                else None,
                blocked_reason=reason,
            )
            if self.publish is not None:
                from .patient_postprocessing import project_patient_facing

                run = await asyncio.to_thread(_read_run, self.store, run_id)
                post = await asyncio.to_thread(
                    project_patient_facing, self.store, run_id
                )
                try:
                    await self.publish(
                        run_id,
                        {
                            "run_id": run_id,
                            "status": run.status,
                            "execution_state": run.execution_state,
                            "blocked_reason": run.blocked_reason,
                            "next_check_at": run.next_check_at.isoformat()
                            if run.next_check_at
                            else None,
                            "patient_facing": post,
                        },
                    )
                except Exception:
                    LOGGER.exception("run_observer_failed", run_id=run_id)
        except Exception:
            LOGGER.exception("run_consumer_record_failed", run_id=run_id)
        finally:
            if owner is not None:
                # Continuation/context has drained every send and publication worker.
                owner.close()
            self._retry_after[run_id] = (
                asyncio.get_running_loop().time() + self.retry_delay
            )
            self.wake()

    async def admission(self, function, *args, **kwargs):
        if self._stopping:
            raise AdmissionUnavailable(
                "Engine consumer is shutting down; retry admission"
            )
        task = asyncio.create_task(asyncio.to_thread(function, *args, **kwargs))
        self.admissions.add(task)
        try:
            # Retain the result/worker on HTTP disconnection and during shutdown.
            result = await drain_task(task)
            self.wake()
            return result
        finally:
            self.admissions.discard(task)

    async def admit_post(self, run_id, *, config_snapshot, budget=1.0):
        from .patient_postprocessing import admit_patient_facing

        deadline = asyncio.get_running_loop().time() + budget
        while True:
            result = await self.admission(
                admit_patient_facing,
                self.store,
                run_id,
                config_snapshot=config_snapshot,
            )
            if result["state"] != "admitting":
                return result
            if asyncio.get_running_loop().time() >= deadline:
                raise AdmissionUnavailable(
                    "Post admission is contended; retry the same action"
                )
            await asyncio.sleep(min(0.05, budget))

    async def stop(self):
        self._stopping = True
        self.wake()
        if self._scan_task is not None:
            await drain_task(self._scan_task)
        # No worker cancellation: a synchronous paid send may still be running.
        for task in list(self.admissions) + list(self.tasks.values()):
            try:
                await drain_task(task)
            except BaseException:
                LOGGER.exception("run_consumer_drain_failed")
        self.tasks.clear()
        self.admissions.clear()
