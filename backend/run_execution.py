"""Inactive, host-local execution ownership foundation for E2–E6.

No scheduler or provider runs here. Construct ExecutionStore(storage.engine) to
pin one SQLite database and its canonical sibling lock directory. A worker must
hold RunOwner until every thread/child that can write or send has been drained.
The consumer (E6) carries list_due_runs' last id into the next finite scan and
resets after an empty page; lock contention is never an ownership verdict.

E2/E4 integration: PaidRequest and StageReceipt are schema-only in this task.
Use owner.transaction() for their short metadata mutations, verify their full
immutable identity on rejoin, and validate receipt files before accepting them.
Do no file/provider work or await inside that transaction. E2 owns paid state
classification/reconciliation; E4 owns clinical stage/member success policies.
"""

from __future__ import annotations

import fcntl
import hashlib
import os
import threading
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy import or_, select, update
from sqlalchemy.orm import Session

from . import storage


class StaleOwner(RuntimeError):
    """The local handle or its durable fence no longer authorizes a write."""


class ExecutionConflict(RuntimeError):
    """A stored execution identity or terminal obligation conflicts with this write."""


def require_unowned_run(session, run):
    """Legacy entrypoints cannot spend or mutate a receipt-covered operation."""
    post = session.scalar(
        select(storage.PostObligation.run_id)
        .where(storage.PostObligation.run_id == run.id)
        .limit(1)
    )
    if (
        run.start_requested_at is not None
        or run.execution_state is not None
        or run.execution_manifest_hash is not None
        or post is not None
    ):
        raise ExecutionConflict(
            "Receipt-covered run requires the engine owned consumer; "
            "rejoin its existing start/post action or reconcile its blocked reason"
        )


def _now():
    return datetime.now(timezone.utc)


def _due_filters(now):
    posts = select(storage.PostObligation.run_id).where(
        storage.PostObligation.run_id == storage.Run.id,
        storage.PostObligation.state.not_in(["done", "skipped"]),
    )
    due_posts = posts.where(
        storage.PostObligation.state.in_(["pending", "owned"]),
        or_(
            storage.PostObligation.next_check_at.is_(None),
            storage.PostObligation.next_check_at <= now,
        ),
    )
    return (
        storage.Run.start_requested_at.is_not(None),
        storage.Run.execution_state.in_(["pending", "owned"]),
        or_(storage.Run.next_check_at.is_(None), storage.Run.next_check_at <= now),
        # A complete council with only blocked/delayed obligations has no due work.
        # All terminal obligations still allow a final execution-done checkpoint.
        or_(storage.Run.status != "complete", ~posts.exists(), due_posts.exists()),
    )


class ExecutionStore:
    """An engine and lock-root pair, unaffected by later storage.reset_engine calls.

    SQLite URI and in-memory databases have no unambiguous host file identity and
    are rejected. Locks are never unlinked. Canonicalizing resolves symlink aliases;
    supported deployments use one canonical DB file, not hardlink aliases.
    """

    def __init__(self, engine):
        database = engine.url.database
        if (
            engine.dialect.name != "sqlite"
            or not database
            or database == ":memory:"
            or engine.url.query
        ):
            raise ValueError("ownership requires a plain file-backed SQLite database")
        self.engine = engine
        self.db_path = Path(database).resolve()
        self.lock_root = self.db_path.parent / (self.db_path.name + ".run-locks")

    def request_run_start(self, run_id: str, *, expected: dict | None = None):
        """Commit intent once; never reset state, ownership, manifest, or clinical data.

        E6 must validate legacy adoption and current pinned inputs before calling.
        Optional expected fields fence an API-validated new admission against
        a concurrent change. Existing start intent always rejoins unchanged.
        """
        with Session(self.engine, expire_on_commit=False) as session:
            result = session.execute(
                update(storage.Run)
                .where(
                    storage.Run.id == run_id,
                    storage.Run.start_requested_at.is_(None),
                    storage.Run.execution_state.is_(None),
                    *(
                        getattr(storage.Run, name) == value
                        for name, value in (expected or {}).items()
                    ),
                )
                .values(start_requested_at=_now(), execution_state="pending")
            )
            run = session.get(storage.Run, run_id)
            if run is None:
                raise KeyError(run_id)
            if (
                expected is not None
                and result.rowcount != 1
                and run.start_requested_at is None
            ):
                raise ExecutionConflict("run changed during start admission")
            session.commit()
            return run

    def list_due_runs(
        self, now: datetime, limit: int = 100, after_id: str | None = None
    ):
        """Keyset page including stale owned rows and council-complete post work.

        Carry the last id even if flock is contended; after an empty page reset
        after_id to None. Stable IDs prevent an early live owner starving later
        rows. Clinical Run.status has no bearing on ownership eligibility.
        """
        if not 1 <= limit <= 1000:
            raise ValueError("limit must be between 1 and 1000")
        query = select(storage.Run).where(
            *_due_filters(now),
        )
        if after_id is not None:
            query = query.where(storage.Run.id > after_id)
        with Session(self.engine, expire_on_commit=False) as session:
            return list(session.scalars(query.order_by(storage.Run.id).limit(limit)))

    def claim_run_owner(self, run_id: str):
        """Nonblocking flock first, short DB claim second; None means unavailable.

        Tokens are random per claim and generations increase even after process
        death. PID and owner_started_at are diagnostics, never takeover authority.
        """
        self.lock_root.mkdir(mode=0o700, parents=True, exist_ok=True)
        # IDs are opaque, so a digest also prevents path traversal or slash aliases.
        lock_path = self.lock_root / (
            hashlib.sha256(run_id.encode()).hexdigest() + ".lock"
        )
        fd = os.open(lock_path, os.O_CREAT | os.O_RDWR | os.O_CLOEXEC, 0o600)
        claimed = False
        try:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                return None
            token = uuid.uuid4().hex
            with Session(self.engine, expire_on_commit=False) as session:
                result = session.execute(
                    update(storage.Run)
                    .where(
                        storage.Run.id == run_id,
                        *_due_filters(_now()),
                    )
                    .values(
                        execution_state="owned",
                        owner_token=token,
                        owner_generation=storage.Run.owner_generation + 1,
                        owner_pid=os.getpid(),
                        owner_started_at=_now(),
                        next_check_at=None,
                    )
                )
                run = session.get(storage.Run, run_id)
                if run is None:
                    raise KeyError(run_id)
                if result.rowcount != 1:
                    return None
                session.commit()
                owner = RunOwner(self, run_id, token, run.owner_generation, fd)
                claimed = True
                return owner
        finally:
            if not claimed:
                # Closing this unique open description cannot unlock another owner.
                os.close(fd)


class RunOwner:
    """Opaque active flock handle; close abandons metadata for lock reconciliation.

    release commits a fenced state before closing. A failed commit retains the
    lock so the caller can retry or explicitly close. Context exit releases to
    pending and always closes on error. Never copy a handle into another process.
    """

    def __init__(self, store, run_id, token, generation, fd):
        self.store = store
        self.run_id = run_id
        self.token = token
        self.generation = generation
        self._fd = fd
        self._pid = os.getpid()
        self._mutex = threading.RLock()

    def _assert_active(self):
        if self._fd is None or os.getpid() != self._pid:
            raise StaleOwner(self.run_id)

    @contextmanager
    def transaction(self):
        """Fenced short child-record transaction, committed atomically on exit.

        The first UPDATE acquires SQLite's write reservation and checks the run
        token/generation/state. Caller writes only this run's child metadata and
        must not commit/rollback itself or change the run fence. Exceptions roll
        back all writes. Work that can await/send/write files stays outside.
        """
        with self._mutex:
            self._assert_active()
            with self.store.engine.begin() as connection:
                result = connection.execute(
                    update(storage.Run)
                    .where(
                        storage.Run.id == self.run_id,
                        storage.Run.execution_state == "owned",
                        storage.Run.owner_token == self.token,
                        storage.Run.owner_generation == self.generation,
                    )
                    .values(owner_generation=self.generation)
                )
                if result.rowcount != 1:
                    raise StaleOwner(self.run_id)
                with Session(connection, expire_on_commit=False) as session:
                    yield session
                    # A same-thread close inside the body must roll back, too.
                    self._assert_active()
                    session.flush()

    @contextmanager
    def file_guard(self):
        """Keep close/release serialized with a short immutable file publication.

        Fenced checkpoints surround file work under the real handle mutex. No
        SQLite transaction spans the file operation; no provider call or await
        belongs here. The flock remains held until this guard has exited.
        """
        with self._mutex:
            self.checkpoint()
            yield
            self.checkpoint()

    def checkpoint(self):
        """Verify durable ownership without using elapsed time as lock authority."""
        with self.transaction():
            pass

    def bind_manifest(self, path: str, sha256: str):
        """Bind once; file creation/hash verification belongs to the execution caller."""
        if (
            not path
            or len(sha256) != 64
            or any(c not in "0123456789abcdef" for c in sha256)
        ):
            raise ValueError("manifest needs a path and lowercase SHA-256")
        with self.transaction() as session:
            run = session.get(storage.Run, self.run_id)
            identity = (run.execution_manifest_path, run.execution_manifest_hash)
            if identity not in [(None, None), (path, sha256)]:
                raise ExecutionConflict("execution manifest changed")
            run.execution_manifest_path, run.execution_manifest_hash = path, sha256

    def ensure_post_obligation(self, kind: str, manifest_path: str, manifest_hash: str):
        """Create/rejoin one pinned post obligation; never reopen its saved state."""
        if kind not in ["patient_facing", "cathode"]:
            raise ValueError("unsupported postprocessing kind")
        if (
            not manifest_path
            or len(manifest_hash) != 64
            or any(c not in "0123456789abcdef" for c in manifest_hash)
        ):
            raise ValueError("post manifest needs a path and lowercase SHA-256")
        with self.transaction() as session:
            row = session.get(storage.PostObligation, (self.run_id, kind))
            if row is not None:
                if (row.manifest_path, row.manifest_hash) != (
                    manifest_path,
                    manifest_hash,
                ):
                    raise ExecutionConflict("postprocessing manifest changed")
            else:
                row = storage.PostObligation(
                    run_id=self.run_id,
                    kind=kind,
                    manifest_path=manifest_path,
                    manifest_hash=manifest_hash,
                    owner_token=self.token,
                    owner_generation=self.generation,
                )
                session.add(row)
            session.flush()
            return row

    def transition_post_obligation(
        self,
        kind: str,
        *,
        expected_state: str,
        state: str,
        receipt_path: str | None = None,
        receipt_hash: str | None = None,
        blocked_reason: str | None = None,
    ):
        """Fenced CAS; terminal/blocked post work needs explicit future reconciliation.

        E5 verifies receipt files before done. This layer requires their binding;
        it never performs generation or interprets provider outcome.
        """
        if expected_state not in ["pending", "owned"] or state not in [
            "pending",
            "owned",
            "blocked",
            "done",
            "skipped",
        ]:
            raise ExecutionConflict("invalid postprocessing transition")
        if state == "done" and (not receipt_path or not receipt_hash):
            raise ExecutionConflict("completed postprocessing requires a receipt")
        if state == "blocked" and not blocked_reason:
            raise ExecutionConflict("blocked postprocessing requires a reason")
        with self.transaction() as session:
            row = session.get(storage.PostObligation, (self.run_id, kind))
            if row is None:
                raise KeyError((self.run_id, kind))
            if row.state != expected_state:
                raise ExecutionConflict("postprocessing state changed")
            row.state = state
            row.receipt_path, row.receipt_hash = receipt_path, receipt_hash
            row.blocked_reason = blocked_reason
            row.owner_token, row.owner_generation = self.token, self.generation
            row.updated_at = _now()
            session.flush()
            return row

    def release(
        self,
        *,
        state: str = "pending",
        next_check_at: datetime | None = None,
        blocked_reason: str | None = None,
    ):
        """Fenced release/terminal commit, then unlock; failure keeps this lock held."""
        if state not in ["pending", "blocked", "done"]:
            raise ValueError("invalid release state")
        if state == "blocked" and not blocked_reason:
            raise ValueError("blocked release requires a reason")
        with self._mutex:
            with self.transaction() as session:
                if state == "done":
                    unfinished = session.scalar(
                        select(storage.PostObligation.run_id)
                        .where(
                            storage.PostObligation.run_id == self.run_id,
                            storage.PostObligation.state.not_in(["done", "skipped"]),
                        )
                        .limit(1)
                    )
                    uncertain = session.scalar(
                        select(storage.PaidRequest.run_id)
                        .where(
                            storage.PaidRequest.run_id == self.run_id,
                            storage.PaidRequest.state.in_(
                                ["prepared", "dispatched", "unknown"]
                            ),
                        )
                        .limit(1)
                    )
                    if unfinished or uncertain:
                        raise ExecutionConflict("outstanding execution obligations")
                run = session.get(storage.Run, self.run_id)
                run.execution_state = state
                run.next_check_at = next_check_at
                run.blocked_reason = blocked_reason
                run.owner_token = None
                # generation and PID/start diagnostics retain the last claim evidence.
            self.close()

    def close(self):
        """Drop only this descriptor; leave owned metadata recoverable after abandonment."""
        with self._mutex:
            if self._fd is not None:
                os.close(self._fd)
                self._fd = None

    def __enter__(self):
        self._assert_active()
        return self

    def __exit__(self, exc_type, exc, traceback):
        try:
            if self._fd is not None:
                self.release()
        finally:
            self.close()
