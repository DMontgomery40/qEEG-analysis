"""Durable intent and host ownership invariants; all DBs and workers are scratch."""

import importlib
import multiprocessing
import sqlite3
from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

from backend import storage


def execution():
    # Keep collection working in RED so absent implementation is an assertion failure.
    assert importlib.util.find_spec(
        "backend.run_execution"
    ), "ownership primitives absent"
    return importlib.import_module("backend.run_execution")


@pytest.fixture
def db(tmp_path):
    url = f"sqlite:///{tmp_path / 'app.db'}"
    storage.reset_engine(url)
    storage.init_db()
    return url


def add_run(run_id, status="created"):
    with storage.session_scope() as session:
        session.add(
            storage.Run(
                id=run_id,
                patient_id="patient",
                report_id="report",
                status=status,
                council_model_ids_json='["original"]',
                source_report_ids_json='["source"]',
                analysis_input_fingerprint="original-input",
            )
        )
        session.commit()


def read_run(run_id):
    with storage.session_scope() as session:
        return session.get(storage.Run, run_id)


@pytest.mark.parametrize(
    "status", ["created", "running", "complete", "failed", "needs_auth"]
)
def test_explicit_intent_preserves_clinical_snapshot_and_is_idempotent(db, status):
    ex = execution()
    add_run("run", status)
    store = ex.ExecutionStore(storage.engine)
    assert read_run("run").start_requested_at is None
    assert store.list_due_runs(datetime.now(timezone.utc)) == []
    first = store.request_run_start("run")
    second = store.request_run_start("run")
    assert first.start_requested_at == second.start_requested_at
    assert second.execution_state == "pending"
    assert second.owner_generation == 0
    assert second.status == status
    assert second.council_model_ids_json == '["original"]'
    assert second.source_report_ids_json == '["source"]'
    assert second.analysis_input_fingerprint == "original-input"


@pytest.mark.parametrize("state", ["blocked", "done"])
def test_repeated_start_does_not_reopen_terminal_or_blocked(db, state):
    ex = execution()
    add_run("run")
    store = ex.ExecutionStore(storage.engine)
    store.request_run_start("run")
    owner = store.claim_run_owner("run")
    owner.bind_manifest("manifest.json", "a" * 64)
    owner.release(state=state, blocked_reason="unknown" if state == "blocked" else None)
    before = read_run("run")
    store.request_run_start("run")
    after = read_run("run")
    assert (
        after.execution_state,
        after.owner_generation,
        after.blocked_reason,
        after.execution_manifest_hash,
        after.start_requested_at,
    ) == (
        before.execution_state,
        before.owner_generation,
        before.blocked_reason,
        before.execution_manifest_hash,
        before.start_requested_at,
    )
    assert store.claim_run_owner("run") is None


def test_missing_and_unrequested_run(db):
    ex = execution()
    store = ex.ExecutionStore(storage.engine)
    with pytest.raises(KeyError):
        store.request_run_start("missing")
    with pytest.raises(KeyError):
        store.claim_run_owner("missing")
    add_run("run")
    assert store.claim_run_owner("run") is None


def test_fair_keyset_scan_traverses_contended_rows_and_complete_postprocessing(db):
    ex = execution()
    store = ex.ExecutionStore(storage.engine)
    now = datetime.now(timezone.utc)
    for name in ["a", "b", "c", "d", "e", "f"]:
        add_run(name, "complete" if name == "e" else "created")
        store.request_run_start(name)
    held = store.claim_run_owner("a")
    with store.claim_run_owner("b") as owner:
        owner.release(state="blocked", blocked_reason="unknown")
    with store.claim_run_owner("c") as owner:
        owner.release(state="done")
    with store.claim_run_owner("d") as owner:
        owner.release(next_check_at=now + timedelta(days=1))
    with store.claim_run_owner("e") as owner:
        owner.ensure_post_obligation("patient_facing", "post.json", "b" * 64)
    page1 = store.list_due_runs(now, limit=1)
    assert [r.id for r in page1] == ["a"]
    page2 = store.list_due_runs(now, limit=1, after_id=page1[-1].id)
    assert [r.id for r in page2] == ["e"]
    assert [r.id for r in store.list_due_runs(now, limit=1, after_id="e")] == ["f"]
    held.release()


@pytest.mark.parametrize("invalidate", ["token", "generation", "closed", "released"])
def test_stale_owner_cannot_write(db, invalidate):
    ex = execution()
    add_run("run")
    store = ex.ExecutionStore(storage.engine)
    store.request_run_start("run")
    owner = store.claim_run_owner("run")
    if invalidate in ["token", "generation"]:
        with storage.engine.begin() as conn:
            if invalidate == "token":
                conn.exec_driver_sql("UPDATE runs SET owner_token='replacement'")
            else:
                conn.exec_driver_sql(
                    "UPDATE runs SET owner_generation=owner_generation+1"
                )
    elif invalidate == "closed":
        owner.close()
    else:
        owner.release()
    with pytest.raises(ex.StaleOwner):
        owner.checkpoint()
    with pytest.raises(ex.StaleOwner):
        owner.bind_manifest("manifest", "a" * 64)
    with pytest.raises(ex.StaleOwner):
        owner.release(state="done")
    owner.close()


def test_manifest_immutable_and_post_identity_fenced(db):
    ex = execution()
    add_run("run")
    store = ex.ExecutionStore(storage.engine)
    store.request_run_start("run")
    with store.claim_run_owner("run") as owner:
        owner.bind_manifest("manifest", "a" * 64)
        owner.bind_manifest("manifest", "a" * 64)
        for path, digest in [("changed", "a" * 64), ("manifest", "b" * 64)]:
            with pytest.raises(ex.ExecutionConflict):
                owner.bind_manifest(path, digest)
        row = owner.ensure_post_obligation("patient_facing", "post", "b" * 64)
        same = owner.ensure_post_obligation("patient_facing", "post", "b" * 64)
        assert row.kind == same.kind
        with pytest.raises(ex.ExecutionConflict):
            owner.ensure_post_obligation("patient_facing", "post", "c" * 64)
        with pytest.raises(ex.ExecutionConflict):
            owner.release(state="done")
        owner.transition_post_obligation(
            "patient_facing", expected_state="pending", state="skipped"
        )
        owner.release(state="done")
    assert read_run("run").execution_state == "done"


def test_context_error_rolls_back_owned_transaction_and_keeps_recoverable_intent(db):
    ex = execution()
    add_run("run")
    store = ex.ExecutionStore(storage.engine)
    store.request_run_start("run")
    with pytest.raises(RuntimeError):
        with store.claim_run_owner("run") as owner:
            with owner.transaction() as session:
                session.add(
                    storage.PostObligation(
                        run_id="run",
                        kind="cathode",
                        manifest_path="post",
                        manifest_hash="a" * 64,
                        owner_token=owner.token,
                        owner_generation=owner.generation,
                    )
                )
                raise RuntimeError("crash")
    with storage.session_scope() as session:
        assert session.scalar(select(storage.PostObligation)) is None
    assert read_run("run").execution_state == "pending"
    with store.claim_run_owner("run") as next_owner:
        assert next_owner.generation == 2


def _worker(url, run_id, pipe):
    from backend.run_execution import ExecutionStore

    storage.reset_engine(url)
    store = ExecutionStore(storage.engine)
    try:
        owner = store.claim_run_owner(run_id)
        pipe.send(None if owner is None else owner.generation)
        if owner is not None:
            command = pipe.recv()
            if command == "release":
                owner.release()
    except BaseException as exc:
        pipe.send(("error", repr(exc)))
        raise
    finally:
        pipe.close()


def spawn_owner(url, run_id):
    ctx = multiprocessing.get_context("spawn")
    parent, child = ctx.Pipe()
    proc = ctx.Process(target=_worker, args=(url, run_id, child))
    proc.start()
    child.close()
    assert parent.poll(15), "worker did not respond"
    return proc, parent, parent.recv()


def test_real_process_lock_death_stalled_owner_and_independent_runs(db):
    ex = execution()
    store = ex.ExecutionStore(storage.engine)
    for name in ["same", "other"]:
        add_run(name)
        store.request_run_start(name)
    processes = []
    pipes = []
    try:
        first, first_pipe, generation = spawn_owner(db, "same")
        processes.append(first)
        pipes.append(first_pipe)
        assert generation == 1
        # A wildly stale diagnostic timestamp never authorizes stealing a live flock.
        with storage.engine.begin() as conn:
            conn.exec_driver_sql(
                "UPDATE runs SET owner_started_at='1900-01-01' WHERE id='same'"
            )
        second, second_pipe, result = spawn_owner(db, "same")
        processes.append(second)
        pipes.append(second_pipe)
        assert result is None
        second.join(10)
        assert second.exitcode == 0
        other, other_pipe, result = spawn_owner(db, "other")
        processes.append(other)
        pipes.append(other_pipe)
        assert result == 1
        other_pipe.send("release")
        other.join(10)
        assert other.exitcode == 0
        first.kill()
        first.join(10)
        replacement, replacement_pipe, result = spawn_owner(db, "same")
        processes.append(replacement)
        pipes.append(replacement_pipe)
        assert result == 2
        replacement_pipe.send("release")
        replacement.join(10)
        assert replacement.exitcode == 0
        assert read_run("same").owner_generation == 2
    finally:
        for proc in processes:
            if proc.is_alive():
                proc.kill()
            proc.join(10)
        for pipe in pipes:
            pipe.close()


def test_claim_db_exception_drops_lock_and_descriptor(db):
    ex = execution()
    add_run("run")
    store = ex.ExecutionStore(storage.engine)
    store.request_run_start("run")
    with storage.engine.begin() as conn:
        conn.exec_driver_sql(
            "CREATE TRIGGER fail_claim BEFORE UPDATE OF owner_token ON runs "
            "BEGIN SELECT RAISE(FAIL, 'claim failure'); END"
        )
    with pytest.raises(IntegrityError):
        store.claim_run_owner("run")
    with storage.engine.begin() as conn:
        conn.exec_driver_sql("DROP TRIGGER fail_claim")
    proc, pipe, result = spawn_owner(db, "run")
    try:
        assert result == 1
        pipe.send("release")
        proc.join(10)
        assert proc.exitcode == 0
    finally:
        if proc.is_alive():
            proc.kill()
        proc.join(10)
        pipe.close()


def test_release_commit_failure_keeps_lock_until_close(db):
    ex = execution()
    add_run("run")
    store = ex.ExecutionStore(storage.engine)
    store.request_run_start("run")
    owner = store.claim_run_owner("run")
    with storage.engine.begin() as conn:
        conn.exec_driver_sql(
            "CREATE TRIGGER fail_release BEFORE UPDATE OF execution_state ON runs "
            "WHEN NEW.execution_state='done' BEGIN SELECT RAISE(FAIL, 'commit failure'); END"
        )
    with pytest.raises(IntegrityError):
        owner.release(state="done")
    assert store.claim_run_owner("run") is None
    assert read_run("run").execution_state == "owned"
    with storage.engine.begin() as conn:
        conn.exec_driver_sql("DROP TRIGGER fail_release")
    owner.release(state="done")
    assert read_run("run").execution_state == "done"


def test_engine_pairing_and_pragmas_survive_reset(db, tmp_path):
    ex = execution()
    first = ex.ExecutionStore(storage.engine)
    add_run("run")
    first.request_run_start("run")
    storage.reset_engine(f"sqlite:///{tmp_path / 'second.db'}")
    storage.init_db()
    second = ex.ExecutionStore(storage.engine)
    assert first.lock_root != second.lock_root
    assert first.lock_root.is_relative_to(tmp_path)
    with first.claim_run_owner("run") as owner:
        owner.checkpoint()
    assert second.list_due_runs(datetime.now(timezone.utc)) == []
    for engine in [first.engine, second.engine]:
        with engine.connect() as conn:
            assert conn.exec_driver_sql("PRAGMA synchronous").scalar() == 2
            assert conn.exec_driver_sql("PRAGMA busy_timeout").scalar() == 5000
    storage.reset_engine("sqlite:///:memory:")
    with pytest.raises(ValueError):
        ex.ExecutionStore(storage.engine)


def test_run_output_additive_execution_metadata_does_not_relabel_status(db):
    from backend import main

    ex = execution()
    add_run("run", "complete")
    store = ex.ExecutionStore(storage.engine)
    initial = main._run_out(read_run("run"))
    assert initial["execution_state"] is None
    assert initial["postprocessing"] == []
    store.request_run_start("run")
    with store.claim_run_owner("run") as owner:
        owner.ensure_post_obligation("patient_facing", "post", "a" * 64)
        owner.release(state="blocked", blocked_reason="paid_outcome_unknown")
    out = main._run_out(read_run("run"))
    assert out["status"] == out["raw_status"] == "complete"
    assert out["execution_state"] == "blocked"
    assert out["blocked_reason"] == "paid_outcome_unknown"
    assert out["owner_generation"] == 1
    assert out["postprocessing"][0]["kind"] == "patient_facing"
    assert out["postprocessing"][0]["state"] == "pending"


def test_due_scan_respects_complete_run_post_obligation_backoff_and_blocks(db):
    ex = execution()
    store = ex.ExecutionStore(storage.engine)
    now = datetime.now(timezone.utc)
    for name in ["a-blocked", "b-future", "c-ready"]:
        add_run(name, "complete")
        store.request_run_start(name)
        with store.claim_run_owner(name) as owner:
            owner.ensure_post_obligation("patient_facing", "post", "a" * 64)
            with owner.transaction() as session:
                row = session.get(storage.PostObligation, (name, "patient_facing"))
                if name == "a-blocked":
                    row.state = "blocked"
                if name == "b-future":
                    row.next_check_at = now + timedelta(days=1)
    assert [r.id for r in store.list_due_runs(now, limit=1)] == ["c-ready"]


def _hold_sqlite_write(url, pipe):
    conn = sqlite3.connect(url.removeprefix("sqlite:///"))
    conn.execute("BEGIN IMMEDIATE")
    conn.execute("UPDATE runs SET error_message='external writer' WHERE id='run'")
    pipe.send("locked")
    pipe.recv()
    conn.commit()
    conn.close()
    pipe.close()


def test_real_sqlite_contention_waits_then_claims_without_losing_intent(db):
    ex = execution()
    import concurrent.futures

    store = ex.ExecutionStore(storage.engine)
    add_run("run")
    store.request_run_start("run")
    ctx = multiprocessing.get_context("spawn")
    pipe, child = ctx.Pipe()
    proc = ctx.Process(target=_hold_sqlite_write, args=(db, child))
    proc.start()
    child.close()
    try:
        assert pipe.poll(15) and pipe.recv() == "locked"
        with concurrent.futures.ThreadPoolExecutor() as pool:
            pending = pool.submit(store.claim_run_owner, "run")
            with pytest.raises(concurrent.futures.TimeoutError):
                pending.result(timeout=0.1)
            pipe.send("commit")
            owner = pending.result(timeout=10)
            assert owner.generation == 1
            owner.release()
        proc.join(10)
        assert proc.exitcode == 0
        assert read_run("run").error_message == "external writer"
    finally:
        if proc.is_alive():
            proc.kill()
        proc.join(10)
        pipe.close()


def test_relative_engine_pairing_stays_bound_after_working_directory_change(
    tmp_path, monkeypatch
):
    ex = execution()
    monkeypatch.chdir(tmp_path)
    storage.reset_engine("sqlite:///relative.db")
    storage.init_db()
    add_run("run")
    original = ex.ExecutionStore(storage.engine)
    other = tmp_path / "other"
    other.mkdir()
    monkeypatch.chdir(other)
    later = ex.ExecutionStore(storage.engine)
    assert later.db_path == original.db_path
    assert later.lock_root == original.lock_root


def test_released_handle_cannot_close_replacement_lock(db):
    ex = execution()
    add_run("run")
    store = ex.ExecutionStore(storage.engine)
    store.request_run_start("run")
    old = store.claim_run_owner("run")
    inode = next(store.lock_root.iterdir()).stat().st_ino
    old.release()
    with store.claim_run_owner("run") as replacement:
        old.close()
        old.close()
        assert replacement.generation == 2
        assert store.claim_run_owner("run") is None
        assert next(store.lock_root.iterdir()).stat().st_ino == inode


def test_terminal_commit_failure_rolls_back_state_and_retains_flock(db):
    from sqlalchemy import event

    ex = execution()
    add_run("run")
    store = ex.ExecutionStore(storage.engine)
    store.request_run_start("run")
    owner = store.claim_run_owner("run")

    # SQLite's deferred FK check fails at COMMIT, after every statement succeeded.
    def foreign_keys(connection, _record):
        connection.execute("PRAGMA foreign_keys=ON")

    event.listen(storage.engine, "connect", foreign_keys)
    storage.engine.dispose()
    with storage.engine.begin() as conn:
        conn.exec_driver_sql(
            "CREATE TABLE commit_guard (run_id VARCHAR REFERENCES runs(id) DEFERRABLE INITIALLY DEFERRED)"
        )
        conn.exec_driver_sql(
            "CREATE TRIGGER fail_commit AFTER UPDATE OF execution_state ON runs "
            "WHEN NEW.execution_state='done' BEGIN INSERT INTO commit_guard VALUES ('missing-run'); END"
        )
    with pytest.raises(IntegrityError):
        owner.release(state="done")
    assert read_run("run").execution_state == "owned"
    assert store.claim_run_owner("run") is None
    with storage.engine.begin() as conn:
        conn.exec_driver_sql("DROP TRIGGER fail_commit")
    owner.release(state="done")
    assert read_run("run").execution_state == "done"


def test_closing_handle_inside_transaction_prevents_commit(db):
    ex = execution()
    add_run("run")
    store = ex.ExecutionStore(storage.engine)
    store.request_run_start("run")
    owner = store.claim_run_owner("run")
    with pytest.raises(ex.StaleOwner):
        with owner.transaction() as session:
            session.add(
                storage.PostObligation(
                    run_id="run",
                    kind="cathode",
                    manifest_path="post",
                    manifest_hash="a" * 64,
                    owner_token=owner.token,
                    owner_generation=owner.generation,
                )
            )
            owner.close()
    with storage.session_scope() as session:
        assert session.get(storage.PostObligation, ("run", "cathode")) is None
    with store.claim_run_owner("run") as replacement:
        assert replacement.generation == 2


@pytest.mark.parametrize("paid_state", ["prepared", "dispatched", "unknown"])
def test_unsettled_paid_record_prevents_execution_done_and_start_cannot_clear_it(
    db, paid_state
):
    ex = execution()
    add_run("run")
    store = ex.ExecutionStore(storage.engine)
    store.request_run_start("run")
    with store.claim_run_owner("run") as owner:
        with owner.transaction() as session:
            session.add(
                storage.PaidRequest(
                    run_id="run",
                    scope_key="s1/member/0",
                    dispatch_ordinal=0,
                    request_path="req",
                    request_hash="a" * 64,
                    route_json="{}",
                    execution_manifest_hash="b" * 64,
                    input_fingerprint="input",
                    state=paid_state,
                    owner_token=owner.token,
                    owner_generation=owner.generation,
                )
            )
        with pytest.raises(ex.ExecutionConflict):
            owner.release(state="done")
        owner.release(state="blocked", blocked_reason=paid_state)
    store.request_run_start("run")
    with storage.session_scope() as session:
        assert (
            session.get(storage.PaidRequest, ("run", "s1/member/0", 0)).state
            == paid_state
        )
    assert read_run("run").execution_state == "blocked"


@pytest.mark.parametrize("terminal", ["done", "skipped", "blocked"])
def test_post_terminal_state_and_identity_are_never_reopened_by_rejoin(db, terminal):
    ex = execution()
    add_run("run")
    store = ex.ExecutionStore(storage.engine)
    store.request_run_start("run")
    with store.claim_run_owner("run") as owner:
        owner.ensure_post_obligation("cathode", "manifest", "a" * 64)
        owner.transition_post_obligation(
            "cathode",
            expected_state="pending",
            state=terminal,
            receipt_path="receipt" if terminal == "done" else None,
            receipt_hash="b" * 64 if terminal == "done" else None,
            blocked_reason="unknown" if terminal == "blocked" else None,
        )
        assert (
            owner.ensure_post_obligation("cathode", "manifest", "a" * 64).state
            == terminal
        )
        with pytest.raises(ex.ExecutionConflict):
            owner.transition_post_obligation(
                "cathode", expected_state=terminal, state="pending"
            )
