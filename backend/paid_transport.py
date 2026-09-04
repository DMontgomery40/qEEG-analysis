"""Inactive, scoped send-once HTTP journal shared by compat and future SDK callers.

E3 must establish one paid_scope per independent semantic unit, pin its manifest
and admission input, use AsyncOpenAI(max_retries=0), and consult
raise_if_paid_blocked after SDK wrapping. Unscoped legacy traffic is unchanged.
The receipt is authority; text extraction and usage callbacks are projections.
"""

from __future__ import annotations

import asyncio
import base64
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import json
import os
import tempfile
import threading

import httpx
from sqlalchemy import select

from . import storage
from .run_execution import ExecutionConflict, RunOwner


class PaidOutcomeUnknown(RuntimeError):
    """Dispatch may have incurred spend. Another paid attempt is not authorized."""

    def __init__(self, key, reason="paid_outcome_unknown"):
        self.key = tuple(key)
        self.reason = reason
        super().__init__(f"{reason}: {self.key}")


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _hash(data):
    return hashlib.sha256(data).hexdigest()


def _task():
    try:
        return asyncio.current_task()
    except RuntimeError:
        return None


_scope: ContextVar[PaidScope | None] = ContextVar("qeeg_paid_scope", default=None)
_worker_scope: ContextVar[PaidScope | None] = ContextVar(
    "qeeg_paid_worker", default=None
)


@dataclass
class PaidScope:
    owner: RunOwner
    semantic_key: str
    execution_manifest_hash: str
    input_fingerprint: str
    ordinal: int = 0
    blocked: PaidOutcomeUnknown | ExecutionConflict | None = None
    _task: object = field(default_factory=_task)
    _thread: int = field(default_factory=threading.get_ident)

    def raise_if_blocked(self):
        if self.blocked is not None:
            raise self.blocked

    def next_key(self):
        self.raise_if_blocked()
        if _worker_scope.get() is not self and (
            _task() is not self._task or threading.get_ident() != self._thread
        ):
            raise ExecutionConflict("parallel units require their own paid_scope")
        key = (self.owner.run_id, self.semantic_key, self.ordinal)
        self.ordinal += 1
        return key


@contextmanager
def paid_scope(owner, semantic_key, execution_manifest_hash, input_fingerprint):
    """Reconstruction starts ordinal zero; create inside each gather task.

    Keep this context around the whole unit, including existing validator/repair
    loops. Never restart it just to retry a local parser. New keys are clinical
    caller policy (E3), not transport retry authorization.
    """
    if not semantic_key or not input_fingerprint:
        raise ValueError("scope requires semantic identity and admission fingerprint")
    with owner.transaction() as session:
        run = session.get(storage.Run, owner.run_id)
        if (
            run.execution_manifest_hash != execution_manifest_hash
            or run.analysis_input_fingerprint != input_fingerprint
            or not execution_manifest_hash
        ):
            raise ExecutionConflict("scope differs from pinned execution/admission")
    cursor = PaidScope(owner, semantic_key, execution_manifest_hash, input_fingerprint)
    token = _scope.set(cursor)
    try:
        yield cursor
    finally:
        _scope.reset(token)


def raise_if_paid_blocked(owner):
    """Recover typed unknown authority even when an SDK wrapped its exception.

    E6 must additionally reconcile previously dispatched receipts before resuming
    an abandoned run; an actively dispatched sibling is not a failed sibling.
    """
    with owner.transaction() as session:
        row = session.scalar(
            select(storage.PaidRequest)
            .where(
                storage.PaidRequest.run_id == owner.run_id,
                storage.PaidRequest.state == "unknown",
            )
            .limit(1)
        )
        if row is not None:
            raise PaidOutcomeUnknown(
                (row.run_id, row.scope_key, row.dispatch_ordinal),
                row.error_classification,
            )


def _route(request):
    url = request.url
    # Current endpoints have no query or userinfo. Reject instead of persisting
    # embedded credentials or quietly changing the route identity.
    if url.scheme not in ("http", "https") or url.userinfo or url.query or url.fragment:
        raise ExecutionConflict("paid URL must have no credentials, query, or fragment")
    if request.method != "POST" or not url.path.endswith(
        ("/chat/completions", "/responses")
    ):
        raise ExecutionConflict("unsupported scoped paid endpoint")
    return _json(
        {
            "method": request.method,
            "url": str(url),
            "headers": {
                k: request.headers[k]
                for k in (
                    "content-type",
                    "content-encoding",
                    "accept",
                    "openai-beta",
                    "openai-organization",
                    "openai-project",
                    "anthropic-version",
                    "anthropic-beta",
                )
                if k in request.headers
            },
            "timeout": request.extensions.get("timeout"),
        }
    )


def _atomic_file(owner, path, data):
    """Immutable publication; failed promotion leaves no false acknowledgement."""
    with owner.file_guard():
        # Directory creation and its parent entries are durable too.
        missing = []
        parent = path.parent
        while not parent.exists():
            missing.append(parent)
            parent = parent.parent
        for directory in reversed(missing):
            directory.mkdir(mode=0o700)
            _fsync_dir(directory.parent)
        if path.exists():
            if path.read_bytes() != data:
                raise ExecutionConflict("immutable receipt file changed")
            return
        fd, temporary = tempfile.mkstemp(prefix="." + path.name + ".", dir=path.parent)
        try:
            with os.fdopen(fd, "wb") as stream:
                stream.write(data)
                stream.flush()
                os.fsync(stream.fileno())
            # link is an atomic, no-overwrite promotion on the supported host FS.
            os.link(temporary, path)
            _fsync_dir(path.parent)
        finally:
            os.unlink(temporary)
            _fsync_dir(path.parent)


def _fsync_dir(path):
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _rejection(status, body):
    """Conservative explicit protocol rejections; generic status alone is unknown.

    The compat contract supports structured authentication errors and explicit
    chat-unsupported errors directing clients to Responses. 429/5xx and generic
    invalid requests do not prove the provider's generation acceptance boundary.
    """
    try:
        error = json.loads(body).get("error", {})
        code = error.get("code") or error.get("type")
        message = error.get("message", "").lower()
    except (ValueError, AttributeError, TypeError):
        return None
    if status == 401 and code in (
        "invalid_api_key",
        "authentication_error",
        "invalid_authentication",
    ):
        return "authentication_rejected"
    if status in (400, 404, 405) and (
        ("not supported" in message or "not support chat" in message)
        and ("chat" in message)
        and ("responses" in message or "response endpoint" in message)
    ):
        return "chat_endpoint_rejected"
    return None


class _Receipt:
    def __init__(self, scope, request):
        self.scope, self.owner, self.request = scope, scope.owner, request
        self.acknowledged = False
        self.key = scope.next_key()
        self.body = request.content
        self.route = _route(request)
        self.identity = {
            "version": 1,
            "key": list(self.key),
            "request_hash": _hash(self.body),
            "route_json": self.route,
            "execution_manifest_hash": scope.execution_manifest_hash,
            "input_fingerprint": scope.input_fingerprint,
        }
        root = self.owner.store.db_path.parent / (
            self.owner.store.db_path.name + ".paid"
        )
        stem = _hash(_json(list(self.key)).encode())
        self.request_path = root / stem / "request.body"
        self.request_identity_path = root / stem / "request.json"
        self.response_path = root / stem / "response.json"

    def _verify_row(self, row):
        actual = (
            row.request_path,
            row.request_hash,
            row.route_json,
            row.execution_manifest_hash,
            row.input_fingerprint,
        )
        expected = (
            str(self.request_path),
            self.identity["request_hash"],
            self.route,
            self.scope.execution_manifest_hash,
            self.scope.input_fingerprint,
        )
        if actual != expected:
            raise ExecutionConflict("paid request identity changed")

    def _verify_files(self):
        with self.owner.file_guard():
            try:
                body = self.request_path.read_bytes()
                identity = self.request_identity_path.read_bytes()
            except OSError as exc:
                raise ExecutionConflict("paid request file missing") from exc
            if body != self.body or identity != _json(self.identity).encode():
                raise ExecutionConflict("paid request bytes or identity changed")

    def prepare(self):
        # A row is created only after the exact request file is safely published.
        # No network occurs before the following dispatched transaction commits.
        with self.owner.transaction() as session:
            row = session.get(storage.PaidRequest, self.key)
            if row is not None:
                self._verify_row(row)
        if row is None:
            _atomic_file(
                self.owner, self.request_identity_path, _json(self.identity).encode()
            )
            _atomic_file(self.owner, self.request_path, self.body)
            with self.owner.transaction() as session:
                row = session.get(storage.PaidRequest, self.key)
                if row is None:
                    row = storage.PaidRequest(
                        run_id=self.key[0],
                        scope_key=self.key[1],
                        dispatch_ordinal=self.key[2],
                        request_path=str(self.request_path),
                        request_hash=self.identity["request_hash"],
                        route_json=self.route,
                        execution_manifest_hash=self.scope.execution_manifest_hash,
                        input_fingerprint=self.scope.input_fingerprint,
                        owner_token=self.owner.token,
                        owner_generation=self.owner.generation,
                    )
                    session.add(row)
                else:
                    self._verify_row(row)
        self._verify_files()
        if self.response_path.exists():
            return self.reconcile()
        with self.owner.transaction() as session:
            row = session.get(storage.PaidRequest, self.key)
            self._verify_row(row)
            state = row.state
            if state == "prepared":
                # A prior explicit unknown elsewhere cannot become a new request.
                blocked = session.scalar(
                    select(storage.PaidRequest)
                    .where(
                        storage.PaidRequest.run_id == self.owner.run_id,
                        storage.PaidRequest.state == "unknown",
                    )
                    .limit(1)
                )
                if blocked is not None:
                    self.scope.blocked = PaidOutcomeUnknown(
                        (blocked.run_id, blocked.scope_key, blocked.dispatch_ordinal)
                    )
                    raise self.scope.blocked
                row.state = "dispatched"
                row.dispatched_at = datetime.now(timezone.utc)
                row.owner_token, row.owner_generation = (
                    self.owner.token,
                    self.owner.generation,
                )
        if state != "prepared":
            if state in ("response_saved", "rejected"):
                raise ExecutionConflict("acknowledged response file missing")
            self.unknown("paid_outcome_unknown")
        return None

    def unknown(self, reason):
        unknown = PaidOutcomeUnknown(self.key, reason)
        self.scope.blocked = unknown
        try:
            with self.owner.transaction() as session:
                row = session.get(storage.PaidRequest, self.key)
                self._verify_row(row)
                if row.state not in ("response_saved", "rejected"):
                    row.state = "unknown"
                    row.error_classification = reason
        except Exception as exc:
            # DB/file failure or a replaced owner cannot erase the typed stop
            # signal. The previously committed dispatched marker stays authority.
            raise unknown from exc
        raise unknown

    def save(self, status, headers, body, *, wire_encoded):
        saved_headers = {
            key: value
            for key, value in headers.items()
            if key.lower()
            in (
                "content-type",
                "x-request-id",
                "request-id",
                "x-amzn-requestid",
                "anthropic-request-id",
            )
        }
        if wire_encoded and "content-encoding" in headers:
            saved_headers["content-encoding"] = headers["content-encoding"]
        envelope = {
            "identity": self.identity,
            "http_status": status,
            "headers": saved_headers,
            "body_sha256": _hash(body),
            "body_base64": base64.b64encode(body).decode(),
        }
        _atomic_file(self.owner, self.response_path, _json(envelope).encode())
        return self.reconcile()

    def reconcile(self):
        """Validate the bound complete envelope before repairing file-before-DB death."""
        self._verify_files()
        with self.owner.file_guard():
            raw = self.response_path.read_bytes()
        try:
            envelope = json.loads(raw)
            body = base64.b64decode(envelope["body_base64"], validate=True)
            status = envelope["http_status"]
            headers = envelope["headers"]
            if (
                envelope["identity"] != self.identity
                or _hash(body) != envelope["body_sha256"]
                or not isinstance(status, int)
                or not 100 <= status <= 599
                or not isinstance(headers, dict)
                or any(
                    key
                    not in (
                        "content-type",
                        "content-encoding",
                        "x-request-id",
                        "request-id",
                        "x-amzn-requestid",
                        "anthropic-request-id",
                    )
                    for key in headers
                )
            ):
                raise ValueError("receipt binding differs")
        except (ValueError, KeyError, TypeError) as exc:
            raise ExecutionConflict("invalid paid response receipt") from exc
        # Decode only after the complete envelope is durable. Even malformed
        # compression cannot permit re-dispatch; the raw file is still authority.
        rejection = None
        if status >= 400:
            try:
                decoded = httpx.Response(status, headers=headers, content=body).content
                rejection = _rejection(status, decoded)
            except Exception:
                pass
        state = (
            "response_saved"
            if 200 <= status < 300
            else ("rejected" if rejection else "unknown")
        )
        with self.owner.transaction() as session:
            row = session.get(storage.PaidRequest, self.key)
            self._verify_row(row)
            if row.state == "prepared":
                raise ExecutionConflict("response exists without dispatch authority")
            if row.response_hash is not None and (
                row.response_hash != _hash(raw)
                or row.response_path != str(self.response_path)
                or row.http_status != status
                or row.response_metadata_json != _json(headers)
            ):
                raise ExecutionConflict("saved paid response changed")
            row.state = state
            row.response_path, row.response_hash = str(self.response_path), _hash(raw)
            row.http_status, row.response_metadata_json = status, _json(headers)
            row.error_classification = (
                rejection
                if state == "rejected"
                else ("unclassified_http_response" if state == "unknown" else None)
            )
            row.response_saved_at = row.response_saved_at or datetime.now(timezone.utc)
            row.owner_token, row.owner_generation = (
                self.owner.token,
                self.owner.generation,
            )
        if state == "unknown":
            self.unknown("unclassified_http_response")
        self.acknowledged = True
        return httpx.Response(
            status, headers=headers, content=body, request=self.request
        )


class PaidAsyncTransport(httpx.AsyncBaseTransport):
    """Reusable SDK httpx hook. Inner transport must send once (retries=0).

    Build SDK with max_retries=0 and httpx.AsyncClient(transport=this_adapter).
    Streaming HTTP is consumed fully into the journal before SDK parsing begins;
    E3's structured Agent remains in charge of schemas and validator retries.
    """

    def __init__(self, inner=None):
        self.inner = inner if inner is not None else httpx.AsyncHTTPTransport(retries=0)

    async def handle_async_request(self, request):
        scope = _scope.get()
        if scope is None or request.method == "GET":
            return await self.inner.handle_async_request(request)
        await request.aread()
        try:
            receipt = _Receipt(scope, request)
            replay = receipt.prepare()
        except ExecutionConflict as exc:
            scope.blocked = exc
            raise
        if replay is not None:
            return replay
        response = None
        try:
            response = await self.inner.handle_async_request(request)
            wire_encoded = not response.is_stream_consumed
            if wire_encoded:
                body = b"".join([part async for part in response.stream])
            else:
                body = response.content
            # No await, parser, or callback between complete bytes and durable save.
            result = receipt.save(
                response.status_code, response.headers, body, wire_encoded=wire_encoded
            )
        except PaidOutcomeUnknown:
            raise
        except BaseException as exc:
            if receipt.acknowledged:
                raise
            try:
                receipt.unknown(
                    "cancelled_after_dispatch"
                    if isinstance(exc, asyncio.CancelledError)
                    else "transport_or_receipt_failure"
                )
            except PaidOutcomeUnknown as unknown:
                if isinstance(exc, asyncio.CancelledError):
                    raise exc
                raise unknown from exc
        finally:
            if response is not None:
                await response.aclose()
        return result

    async def aclose(self):
        await self.inner.aclose()


class PaidSyncTransport(httpx.BaseTransport):
    """The same journal under synchronous Claude's existing httpx client."""

    def __init__(self, inner=None):
        self.inner = inner if inner is not None else httpx.HTTPTransport(retries=0)

    def handle_request(self, request):
        scope = _scope.get()
        if scope is None or request.method == "GET":
            return self.inner.handle_request(request)
        request.read()
        try:
            receipt = _Receipt(scope, request)
            replay = receipt.prepare()
        except ExecutionConflict as exc:
            scope.blocked = exc
            raise
        if replay is not None:
            return replay
        response = None
        try:
            response = self.inner.handle_request(request)
            wire_encoded = not response.is_stream_consumed
            body = b"".join(response.stream) if wire_encoded else response.content
            result = receipt.save(
                response.status_code, response.headers, body, wire_encoded=wire_encoded
            )
            return result
        except PaidOutcomeUnknown:
            raise
        except BaseException as exc:
            if receipt.acknowledged:
                raise
            try:
                receipt.unknown("transport_or_receipt_failure")
            except PaidOutcomeUnknown as unknown:
                raise unknown from exc
        finally:
            if response is not None:
                response.close()

    def close(self):
        self.inner.close()


class PaidAsyncClient(httpx.AsyncClient):
    """Original httpx routing/auth/pooling, with the scoped physical-send hook.

    _transport_for_url is verified against installed httpx 0.28.1 source. This
    narrow override preserves environment proxies and user mounts, which passing
    an explicit default transport to AsyncClient would disable.
    """

    def _transport_for_url(self, url):
        selected = super()._transport_for_url(url)
        return (
            selected
            if isinstance(selected, PaidAsyncTransport)
            else PaidAsyncTransport(selected)
        )


class PaidClient(httpx.Client):
    """Synchronous counterpart retaining the same proxy/mount selection."""

    def _transport_for_url(self, url):
        selected = super()._transport_for_url(url)
        return (
            selected
            if isinstance(selected, PaidSyncTransport)
            else PaidSyncTransport(selected)
        )


async def owned_to_thread(function, *args):
    """Shield and drain the actual worker before propagating cancellation.

    The enclosing RunOwner context must surround this await. Repeated cancellation
    cannot detach the worker or release that context's flock while it can send or
    save. The scoped worker inherits the unit cursor deliberately, unlike gather.
    """
    scope = _scope.get()
    if scope is None:
        return await asyncio.to_thread(function, *args)
    # Reject a gather-inherited cursor before authorizing its worker thread.
    if _task() is not scope._task:
        raise ExecutionConflict("parallel thread units require their own paid_scope")

    def worker():
        token = _worker_scope.set(scope)
        try:
            return function(*args)
        finally:
            _worker_scope.reset(token)

    task = asyncio.create_task(asyncio.to_thread(worker))
    cancelled = None
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError as exc:
            cancelled = exc
        except BaseException:
            break
    if cancelled is not None:
        # Retrieve the worker exception so it cannot be lost as an unobserved
        # task error; its receipt already holds the durable outcome.
        if not task.cancelled():
            task.exception()
        raise cancelled
    return task.result()
