"""Synthetic paid-boundary durability; no live model endpoint is contacted."""

import asyncio
import importlib
import json
from pathlib import Path

import httpx
import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session

from backend import storage
from backend.run_execution import ExecutionStore, ExecutionConflict


def paid():
    assert importlib.util.find_spec("backend.paid_transport"), "paid boundary absent"
    return importlib.import_module("backend.paid_transport")


@pytest.fixture
def owner(tmp_path):
    storage.reset_engine(f'sqlite:///{tmp_path / "app.db"}')
    storage.init_db()
    with storage.session_scope() as session:
        session.add(
            storage.Run(
                id="r",
                patient_id="p",
                report_id="q",
                status="created",
                council_model_ids_json='["original"]',
                analysis_input_fingerprint="input",
            )
        )
        session.commit()
    store = ExecutionStore(storage.engine)
    store.request_run_start("r")
    handle = store.claim_run_owner("r")
    handle.bind_manifest("execution.json", "a" * 64)
    yield handle
    handle.close()


def scope(owner, key="unit"):
    return paid().paid_scope(owner, key, "a" * 64, "input")


def rows(owner):
    with Session(owner.store.engine) as session:
        return list(
            session.scalars(
                select(storage.PaidRequest).order_by(
                    storage.PaidRequest.scope_key, storage.PaidRequest.dispatch_ordinal
                )
            )
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "body",
    [
        b'  {"id":"provider-1","output_text":"ok"} \n',
        b"not json",
        b"",
        b'{"output_text":"ok"}',
    ],
)
async def test_exact_complete_response_replays_without_dispatch(owner, body):
    p = paid()
    calls = []

    def send(request):
        calls.append(request.content)
        return httpx.Response(
            200, content=body, headers={"x-request-id": "req-1", "set-cookie": "secret"}
        )

    async with httpx.AsyncClient(
        transport=p.PaidAsyncTransport(httpx.MockTransport(send))
    ) as client:
        for _ in range(2):
            with scope(owner):
                response = await client.post(
                    "http://test/v1/responses",
                    content=b'{"model":"original"}',
                    headers={"Authorization": "Bearer secret"},
                )
                assert response.content == body
                assert response.headers["x-request-id"] == "req-1"
    assert calls == [b'{"model":"original"}']
    row = rows(owner)[0]
    assert row.state == "response_saved"
    assert Path(row.request_path).read_bytes() == calls[0]
    assert b"secret" not in Path(row.response_path).read_bytes()
    assert "secret" not in row.route_json


@pytest.mark.asyncio
@pytest.mark.parametrize("change", ["body", "route", "manifest", "input", "headers"])
async def test_saved_request_identity_fails_closed(owner, change):
    p = paid()
    calls = []

    def send(request):
        calls.append(1)
        return httpx.Response(200, content=b"ok")

    async with httpx.AsyncClient(
        transport=p.PaidAsyncTransport(httpx.MockTransport(send))
    ) as client:
        with scope(owner):
            await client.post("http://test/v1/responses", content=b"original")
        with pytest.raises(ExecutionConflict):
            with p.paid_scope(
                owner,
                "unit",
                "b" * 64 if change == "manifest" else "a" * 64,
                "changed" if change == "input" else "input",
            ):
                await client.post(
                    "http://other/v1/responses"
                    if change == "route"
                    else "http://test/v1/responses",
                    content=b"changed" if change == "body" else b"original",
                    headers={"OpenAI-Beta": "different"} if change == "headers" else {},
                )
    assert len(calls) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", ["timeout", "reset", "502", "429", "400"])
async def test_ambiguous_outcome_is_sticky_even_after_wrapping(owner, outcome):
    p = paid()
    calls = []

    def send(request):
        calls.append(1)
        if outcome == "timeout":
            raise httpx.ReadTimeout("late")
        if outcome == "reset":
            raise httpx.ReadError("reset")
        return httpx.Response(int(outcome), json={"error": {"message": "ambiguous"}})

    async with httpx.AsyncClient(
        transport=p.PaidAsyncTransport(httpx.MockTransport(send))
    ) as client:
        for _ in range(2):
            with scope(owner) as cursor:
                with pytest.raises(p.PaidOutcomeUnknown):
                    await client.post("http://test/v1/responses", content=b"original")
                with pytest.raises(p.PaidOutcomeUnknown):
                    cursor.raise_if_blocked()
        with pytest.raises(p.PaidOutcomeUnknown):
            p.raise_if_paid_blocked(owner)
    assert len(calls) == 1
    assert rows(owner)[0].state == "unknown"


@pytest.mark.asyncio
async def test_parallel_units_require_explicit_independent_cursors(owner):
    p = paid()
    calls = []

    def send(request):
        calls.append(request.content)
        return httpx.Response(200, content=b"ok")

    async with httpx.AsyncClient(
        transport=p.PaidAsyncTransport(httpx.MockTransport(send))
    ) as client:

        async def unit(key):
            with scope(owner, key):
                for value in [b"one", b"two"]:
                    await client.post("http://test/v1/responses", content=value)

        await asyncio.gather(unit("a"), unit("b"))
        with scope(owner, "parent"):
            with pytest.raises(ExecutionConflict):
                await asyncio.create_task(
                    client.post("http://test/v1/responses", content=b"bad")
                )
    assert [(r.scope_key, r.dispatch_ordinal) for r in rows(owner)] == [
        ("a", 0),
        ("a", 1),
        ("b", 0),
        ("b", 1),
    ]
    assert len(calls) == 4


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "damage", ["request", "response", "binding", "response_hash", "missing_response"]
)
async def test_receipt_corruption_never_dispatches_again(owner, damage):
    p = paid()
    calls = []

    def send(request):
        calls.append(1)
        return httpx.Response(200, content=b"ok")

    async with httpx.AsyncClient(
        transport=p.PaidAsyncTransport(httpx.MockTransport(send))
    ) as client:
        with scope(owner):
            await client.post("http://test/v1/responses", content=b"original")
        row = rows(owner)[0]
        if damage == "request":
            Path(row.request_path).write_bytes(b"changed")
        elif damage == "response":
            Path(row.response_path).write_bytes(b"broken")
        elif damage == "binding":
            payload = json.loads(Path(row.response_path).read_bytes())
            payload["identity"]["key"][1] = "another-unit"
            Path(row.response_path).write_text(json.dumps(payload))
        elif damage == "response_hash":
            with owner.transaction() as session:
                session.get(storage.PaidRequest, ("r", "unit", 0)).response_hash = (
                    "b" * 64
                )
        else:
            Path(row.response_path).unlink()
        with scope(owner), pytest.raises(ExecutionConflict):
            await client.post("http://test/v1/responses", content=b"original")
    assert calls == [1]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "url",
    [
        "http://secret:password@test/v1/responses",
        "http://test/v1/responses?api_key=secret",
        "http://test/v1/responses#secret",
    ],
)
async def test_unsafe_route_rejected_before_body_or_dispatch(owner, url):
    p = paid()

    def forbidden(request):
        pytest.fail("unsafe route dispatched")

    async with httpx.AsyncClient(
        transport=p.PaidAsyncTransport(httpx.MockTransport(forbidden))
    ) as client:
        with scope(owner), pytest.raises(ExecutionConflict):
            await client.post(url, content=b"body")
    assert rows(owner) == []


@pytest.mark.asyncio
async def test_explicit_auth_response_replays_and_retains_error(owner):
    p = paid()
    calls = []
    body = b'{"error":{"code":"invalid_api_key","message":"Authentication rejected"}}'

    def send(request):
        calls.append(1)
        return httpx.Response(401, content=body)

    async with httpx.AsyncClient(
        transport=p.PaidAsyncTransport(httpx.MockTransport(send))
    ) as client:
        for _ in range(2):
            with scope(owner):
                response = await client.post(
                    "http://test/v1/responses", content=b"body"
                )
                assert response.status_code == 401
                assert response.content == body
    assert rows(owner)[0].state == "rejected"
    assert calls == [1]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "branch", ["chat", "responses", "gpt", "openrouter", "glm", "fallback"]
)
async def test_compat_paid_branches_preserve_recipe_and_replay(
    owner, monkeypatch, branch
):
    from backend.llm_client import AsyncOpenAICompatClient

    calls = []
    model = (
        "gpt-5"
        if branch == "gpt"
        else ("z-ai/glm-5.3-flash" if branch == "glm" else "model")
    )
    if branch == "openrouter":
        model = "vendor/extra"
        monkeypatch.setenv("QEEG_OPENROUTER_EXTRA_MODELS", model)
        monkeypatch.setenv("OPENROUTER_API_KEY", "synthetic-key")
        monkeypatch.setenv("QEEG_ROUTE_OPENROUTER_EXTRAS_DIRECT", "1")

    def send(request):
        payload = json.loads(request.content)
        calls.append((str(request.url), payload))
        if branch == "fallback" and request.url.path.endswith("/chat/completions"):
            return httpx.Response(
                400,
                json={
                    "error": {
                        "message": "chat completions not supported; use /v1/responses"
                    }
                },
            )
        if request.url.path.endswith("/responses"):
            return httpx.Response(200, json={"output_text": "ok", "id": "original"})
        if branch == "glm" and len(calls) == 1:
            return httpx.Response(200, json={"choices": [{"message": {"content": ""}}]})
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})

    client = AsyncOpenAICompatClient(
        base_url="http://test", api_key="", transport=httpx.MockTransport(send)
    )
    try:
        for _ in range(2):
            with scope(owner):
                if branch == "responses":
                    output = await client.responses(
                        model_id=model,
                        input_data="input",
                        reasoning_effort="high",
                        max_output_tokens=777,
                    )
                else:
                    output = await client.chat_completions(
                        model_id=model,
                        messages=[{"role": "user", "content": "hi"}],
                        temperature=0.3,
                        max_tokens=777,
                    )
                assert output == "ok"
    finally:
        await client.aclose()
    assert len(calls) == (2 if branch in ("glm", "fallback") else 1)
    assert all(payload["model"] == model for _, payload in calls)
    if branch in ("gpt", "responses"):
        assert calls[0][0].endswith("/v1/responses")
        assert calls[0][1]["max_output_tokens"] == 777
    elif branch == "openrouter":
        assert calls[0][0].startswith("https://openrouter.ai/api/")
    elif branch == "glm":
        assert calls[1][1]["reasoning"] == {"effort": "none", "exclude": True}
        assert len(calls[1][1]["messages"]) == 2
    elif branch == "fallback":
        assert [r.state for r in rows(owner)] == ["rejected", "response_saved"]
    else:
        assert calls[0][1]["temperature"] == 0.3
        assert calls[0][1]["max_tokens"] == 777


@pytest.mark.asyncio
async def test_sdk_unknown_wrapping_cannot_authorize_retry(owner):
    from openai import AsyncOpenAI, APIConnectionError

    p = paid()
    calls = []

    def send(request):
        calls.append(1)
        raise httpx.ReadTimeout("after dispatch")

    async with httpx.AsyncClient(
        transport=p.PaidAsyncTransport(httpx.MockTransport(send))
    ) as http:
        async with AsyncOpenAI(
            base_url="http://test/v1",
            api_key="synthetic",
            max_retries=0,
            http_client=http,
        ) as sdk:
            with scope(owner) as cursor:
                with pytest.raises(APIConnectionError) as caught:
                    await sdk.responses.create(model="original", input="hello")
                assert isinstance(caught.value.__cause__, p.PaidOutcomeUnknown)
                with pytest.raises(p.PaidOutcomeUnknown):
                    cursor.raise_if_blocked()
                with pytest.raises(p.PaidOutcomeUnknown):
                    p.raise_if_paid_blocked(owner)
    assert len(calls) == 1


def _process_attempt(db_url, counter, boundary):
    """Real process death at the individual durable paid boundaries."""
    import os
    from contextlib import contextmanager
    from backend import paid_transport as p

    storage.reset_engine(db_url)
    store = ExecutionStore(storage.engine)
    owner = store.claim_run_owner("r")
    assert owner is not None
    original_transaction = owner.transaction
    if boundary == "prepared":

        @contextmanager
        def stop_prepared():
            with original_transaction() as session:
                yield session
            with Session(store.engine) as session:
                row = session.get(storage.PaidRequest, ("r", "unit", 0))
                if row is not None and row.state == "prepared":
                    os._exit(73)

        owner.transaction = stop_prepared
    original_atomic = p._atomic_file
    if boundary in ("request_file", "response_file"):

        def stop_file(handle, path, data):
            original_atomic(handle, path, data)
            if path.name == (
                "request.body" if boundary == "request_file" else "response.json"
            ):
                os._exit(73)

        p._atomic_file = stop_file
    original_prepare = p._Receipt.prepare
    if boundary == "dispatched":

        def stop_dispatched(receipt):
            original_prepare(receipt)
            os._exit(73)

        p._Receipt.prepare = stop_dispatched
    original_reconcile = p._Receipt.reconcile
    if boundary == "acknowledged":

        def stop_ack(receipt):
            original_reconcile(receipt)
            os._exit(73)

        p._Receipt.reconcile = stop_ack

    def send(request):
        with open(counter, "ab") as file:
            file.write(request.content + b"\n")
            file.flush()
            os.fsync(file.fileno())
        return httpx.Response(200, content=b' {"output_text":"original"}\n')

    async def run():
        async with httpx.AsyncClient(
            transport=p.PaidAsyncTransport(httpx.MockTransport(send))
        ) as client:
            with owner, scope(owner):
                try:
                    response = await client.post(
                        "http://test/v1/responses", content=b'{"model":"original"}'
                    )
                except p.PaidOutcomeUnknown:
                    return 42
                if boundary == "parse":
                    response.json()
                    os._exit(73)
                assert response.content == b' {"output_text":"original"}\n'
                return 0

    raise SystemExit(asyncio.run(run()))


@pytest.mark.parametrize(
    "boundary,dispatches,result",
    [
        ("request_file", 1, 0),
        ("prepared", 1, 0),
        ("dispatched", 0, 42),
        ("response_file", 1, 0),
        ("acknowledged", 1, 0),
        ("parse", 1, 0),
    ],
)
def test_process_replacement_never_repeats_acknowledged_dispatch(
    owner, tmp_path, boundary, dispatches, result
):
    import multiprocessing

    owner.release()
    counter = tmp_path / "dispatches"
    context = multiprocessing.get_context("spawn")
    args = (str(owner.store.engine.url), str(counter))
    child = context.Process(target=_process_attempt, args=(*args, boundary))
    child.start()
    child.join(20)
    assert child.exitcode == 73
    child = context.Process(target=_process_attempt, args=(*args, "recover"))
    child.start()
    child.join(20)
    assert child.exitcode == result
    actual = counter.read_bytes().splitlines() if counter.exists() else []
    assert actual == [b'{"model":"original"}'] * dispatches


@pytest.mark.asyncio
async def test_real_sync_claude_thread_cancellation_drains_before_flock_release(owner):
    import threading
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    from backend.llm_client import AsyncOpenAICompatClient

    entered, finish = threading.Event(), threading.Event()
    received = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            received.append(
                json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            )
            entered.set()
            assert finish.wait(10)
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"choices":[{"message":{"content":"ok"}}]}')

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    client = AsyncOpenAICompatClient(
        base_url=f"http://127.0.0.1:{server.server_port}", api_key=""
    )

    async def call():
        with owner, scope(owner):
            await client.chat_completions(
                model_id="claude-test", messages=[{"role": "user", "content": "hi"}]
            )

    task = asyncio.create_task(call())
    try:
        assert await asyncio.to_thread(entered.wait, 5)
        task.cancel()
        await asyncio.sleep(0.02)
        task.cancel()
        await asyncio.sleep(0.02)
        assert not task.done()
        assert owner.store.claim_run_owner("r") is None
        assert rows(owner)[0].state == "dispatched"
        finish.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert rows(owner)[0].state == "response_saved"
        replacement = owner.store.claim_run_owner("r")
        assert replacement is not None
        with replacement, scope(replacement):
            assert (
                await client.chat_completions(
                    model_id="claude-test", messages=[{"role": "user", "content": "hi"}]
                )
                == "ok"
            )
        assert len(received) == 1
        assert "temperature" not in received[0]
    finally:
        finish.set()
        await client.aclose()
        await asyncio.to_thread(server.shutdown)
        server.server_close()
        thread.join(5)


@pytest.mark.asyncio
async def test_async_cancellation_leaves_unknown_and_never_dispatches_again(owner):
    p = paid()
    entered = asyncio.Event()
    calls = []

    async def send(request):
        calls.append(1)
        entered.set()
        await asyncio.Event().wait()

    async with httpx.AsyncClient(
        transport=p.PaidAsyncTransport(httpx.MockTransport(send))
    ) as client:

        async def call():
            with scope(owner):
                await client.post("http://test/v1/responses", content=b"body")

        task = asyncio.create_task(call())
        await entered.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert rows(owner)[0].state == "unknown"
        with scope(owner), pytest.raises(p.PaidOutcomeUnknown):
            await client.post("http://test/v1/responses", content=b"body")
    assert calls == [1]


@pytest.mark.asyncio
async def test_unknown_survives_metadata_write_failure(owner, monkeypatch):
    from contextlib import contextmanager

    p = paid()
    original = owner.transaction

    @contextmanager
    def broken():
        raise OSError("database unavailable")
        yield

    async def send(request):
        monkeypatch.setattr(owner, "transaction", broken)
        raise httpx.ReadError("already dispatched")

    try:
        async with httpx.AsyncClient(
            transport=p.PaidAsyncTransport(httpx.MockTransport(send))
        ) as client:
            with scope(owner) as cursor:
                with pytest.raises(p.PaidOutcomeUnknown):
                    await client.post("http://test/v1/responses", content=b"body")
                with pytest.raises(p.PaidOutcomeUnknown):
                    cursor.raise_if_blocked()
    finally:
        monkeypatch.setattr(owner, "transaction", original)
    assert rows(owner)[0].state == "dispatched"


@pytest.mark.asyncio
async def test_malformed_compressed_complete_response_is_acknowledged_before_local_decode(
    owner,
):
    p = paid()
    calls = []

    class RawStream(httpx.AsyncByteStream):
        async def __aiter__(self):
            yield b"invalid-gzip"

        async def aclose(self):
            # Stream cleanup also cannot precede durable complete body storage.
            assert rows(owner)[0].state == "response_saved"

    async def send(request):
        calls.append(1)
        return httpx.Response(
            200, headers={"Content-Encoding": "gzip"}, stream=RawStream()
        )

    async with httpx.AsyncClient(
        transport=p.PaidAsyncTransport(httpx.MockTransport(send))
    ) as client:
        for _ in range(2):
            with scope(owner), pytest.raises(httpx.DecodingError):
                await client.post("http://test/v1/responses", content=b"body")
    assert rows(owner)[0].state == "response_saved"
    assert calls == [1]


@pytest.mark.asyncio
async def test_compat_clients_preserve_httpx_environment_proxy_selection(monkeypatch):
    from backend.llm_client import AsyncOpenAICompatClient

    p = paid()
    monkeypatch.setenv("HTTPS_PROXY", "http://127.0.0.1:9")
    monkeypatch.setenv("NO_PROXY", "")
    compat = AsyncOpenAICompatClient(base_url="https://test", api_key="")
    client = compat._get_client()
    try:
        assert client._mounts, "original httpx proxy mounting was disabled"
        actual = client._transport_for_url(httpx.URL("https://test/v1/responses"))
        assert isinstance(actual, p.PaidAsyncTransport)
        assert actual.inner is not client._transport
    finally:
        await compat.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["before_response_file", "after_response_file"])
async def test_receipt_write_failure_recovery_never_sends_again(
    owner, monkeypatch, failure
):
    p = paid()
    calls = []
    original = p._atomic_file

    def broken(handle, path, data):
        if path.name == "response.json":
            if failure == "after_response_file":
                original(handle, path, data)
            raise OSError("disk write interrupted")
        return original(handle, path, data)

    def send(request):
        calls.append(1)
        return httpx.Response(200, content=b"ok")

    async with httpx.AsyncClient(
        transport=p.PaidAsyncTransport(httpx.MockTransport(send))
    ) as client:
        monkeypatch.setattr(p, "_atomic_file", broken)
        with scope(owner), pytest.raises(p.PaidOutcomeUnknown):
            await client.post("http://test/v1/responses", content=b"body")
        monkeypatch.setattr(p, "_atomic_file", original)
        with scope(owner):
            if failure == "after_response_file":
                assert (
                    await client.post("http://test/v1/responses", content=b"body")
                ).content == b"ok"
            else:
                with pytest.raises(p.PaidOutcomeUnknown):
                    await client.post("http://test/v1/responses", content=b"body")
    assert calls == [1]


@pytest.mark.asyncio
@pytest.mark.parametrize("invalidation", ["closed", "released", "token", "generation"])
async def test_stale_owner_cannot_write_request_or_response(
    owner, monkeypatch, invalidation
):
    from backend.run_execution import StaleOwner

    p = paid()
    calls = []

    def send(request):
        calls.append(1)
        return httpx.Response(200, content=b"ok")

    with scope(owner):
        if invalidation == "closed":
            owner.close()
        elif invalidation == "released":
            owner.release()
        else:
            with storage.session_scope() as session:
                run = session.get(storage.Run, "r")
                if invalidation == "token":
                    run.owner_token = "replacement"
                else:
                    run.owner_generation += 1
                session.commit()
        async with httpx.AsyncClient(
            transport=p.PaidAsyncTransport(httpx.MockTransport(send))
        ) as client:
            with pytest.raises(StaleOwner):
                await client.post("http://test/v1/responses", content=b"body")
    assert rows(owner) == []
    assert calls == []
    assert not (
        owner.store.db_path.parent / (owner.store.db_path.name + ".paid")
    ).exists()


def test_owner_file_guard_holds_flock_and_close_mutex_until_publication(
    owner, tmp_path
):
    import threading

    p = paid()
    entered, finish, closed = threading.Event(), threading.Event(), threading.Event()
    target = tmp_path / "receipt"

    def writer():
        with owner.file_guard():
            entered.set()
            assert finish.wait(5)
            target.write_bytes(b"complete")

    def closer():
        owner.close()
        closed.set()

    worker = threading.Thread(target=writer)
    worker.start()
    assert entered.wait(5)
    close_worker = threading.Thread(target=closer)
    close_worker.start()
    try:
        assert not closed.wait(0.03)
        assert owner.store.claim_run_owner("r") is None
    finally:
        finish.set()
        worker.join(5)
        close_worker.join(5)
    assert target.read_bytes() == b"complete"
    assert closed.is_set()
    replacement = owner.store.claim_run_owner("r")
    assert replacement is not None
    before = target.read_bytes()
    try:
        from backend.run_execution import StaleOwner

        with pytest.raises(StaleOwner):
            p._atomic_file(owner, target, b"old-owner-write")
        assert target.read_bytes() == before
    finally:
        replacement.close()


@pytest.mark.asyncio
async def test_sequence_conflict_is_sticky_within_attempt(owner):
    p = paid()
    calls = []

    def send(request):
        calls.append(1)
        return httpx.Response(200, content=b"ok")

    async with httpx.AsyncClient(
        transport=p.PaidAsyncTransport(httpx.MockTransport(send))
    ) as client:
        with scope(owner):
            await client.post("http://test/v1/responses", content=b"original")
        with scope(owner):
            with pytest.raises(ExecutionConflict):
                await client.post("http://test/v1/responses", content=b"changed")
            with pytest.raises(ExecutionConflict):
                await client.post("http://test/v1/responses", content=b"original")
    assert calls == [1]


@pytest.mark.asyncio
async def test_request_file_orphan_keeps_original_route_identity(owner, monkeypatch):
    p = paid()
    original = p._atomic_file
    calls = []

    def interrupted(handle, path, data):
        original(handle, path, data)
        if path.name == "request.body":
            raise OSError("death before prepared row")

    def send(request):
        calls.append(1)
        return httpx.Response(200, content=b"ok")

    async with httpx.AsyncClient(
        transport=p.PaidAsyncTransport(httpx.MockTransport(send))
    ) as client:
        monkeypatch.setattr(p, "_atomic_file", interrupted)
        with scope(owner), pytest.raises(OSError):
            await client.post("http://original/v1/responses", content=b"same-body")
        monkeypatch.setattr(p, "_atomic_file", original)
        with scope(owner), pytest.raises(ExecutionConflict):
            await client.post("http://changed/v1/responses", content=b"same-body")
        with scope(owner):
            assert (
                await client.post("http://original/v1/responses", content=b"same-body")
            ).content == b"ok"
    assert calls == [1]


@pytest.mark.asyncio
async def test_stale_inflight_owner_cannot_publish_for_replacement(owner):
    p = paid()
    calls = []
    replacement = None

    def send(request):
        nonlocal replacement
        calls.append(1)
        owner.close()
        replacement = owner.store.claim_run_owner("r")
        assert replacement is not None
        return httpx.Response(200, content=b"old-worker-result")

    async with httpx.AsyncClient(
        transport=p.PaidAsyncTransport(httpx.MockTransport(send))
    ) as client:
        with scope(owner), pytest.raises(p.PaidOutcomeUnknown):
            await client.post("http://test/v1/responses", content=b"body")
        try:
            assert rows(replacement)[0].response_path is None
            with scope(replacement), pytest.raises(p.PaidOutcomeUnknown):
                await client.post("http://test/v1/responses", content=b"body")
        finally:
            replacement.close()
    assert calls == [1]


@pytest.mark.asyncio
async def test_partial_body_is_unknown_not_a_complete_response(owner):
    p = paid()
    calls = []

    class Stream(httpx.AsyncByteStream):
        async def __aiter__(self):
            yield b'{"output_text":"incomplete'
            raise httpx.ReadError("connection reset")

    def send(request):
        calls.append(1)
        return httpx.Response(200, stream=Stream())

    async with httpx.AsyncClient(
        transport=p.PaidAsyncTransport(httpx.MockTransport(send))
    ) as client:
        for _ in range(2):
            with scope(owner), pytest.raises(p.PaidOutcomeUnknown):
                await client.post("http://test/v1/responses", content=b"body")
    assert rows(owner)[0].state == "unknown"
    assert rows(owner)[0].response_path is None
    assert calls == [1]
