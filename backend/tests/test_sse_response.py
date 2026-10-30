"""Exercise the real SSE response iterator without a server or paid workflow.

TestClient buffers an endless stream. Consuming its body iterator directly lets
these tests check wire frames and disconnect cleanup deterministically instead.
"""

from __future__ import annotations

import asyncio
import json

import pytest


@pytest.fixture
def stream_context(temp_data_dir, monkeypatch):
    from backend import main, storage

    broker = main._EventBroker()
    monkeypatch.setattr(main.app.state, "broker", broker, raising=False)
    with storage.session_scope() as session:
        patient = storage.create_patient(session, label="ZZ_01-01-1900", notes="")
        report = storage.create_report(
            session,
            patient_id=patient.id,
            filename="synthetic.txt",
            mime_type="text/plain",
            stored_path=temp_data_dir / "synthetic.txt",
            extracted_text_path=temp_data_dir / "extracted.txt",
        )
        run = storage.create_run(
            session,
            patient_id=patient.id,
            report_id=report.id,
            council_model_ids=["mock-model"],
            consolidator_model_id="mock-model",
        )
        run_id = run.id
    return main, broker, run_id


def decode_frame(frame):
    assert frame.startswith("data: ")
    assert frame.endswith("\n\n")
    assert len(frame.splitlines()) == 2
    return json.loads(frame.removeprefix("data: "))


async def test_snapshot_precedes_queued_events_and_sets_stream_headers(stream_context):
    main, broker, run_id = stream_context
    response = await main.stream(run_id)
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert response.headers["cache-control"] == "no-cache"
    assert response.headers["connection"] == "keep-alive"
    payloads = [
        {"type": "progress", "run_id": run_id, "message": "α\nsecond line"},
        {"type": "status", "run_id": run_id, "status": "complete"},
    ]
    for payload in payloads:
        await broker.publish(run_id, payload)
    try:
        snapshot = decode_frame(await anext(response.body_iterator))
        assert snapshot["type"] == "snapshot"
        assert snapshot["run_id"] == run_id
        assert snapshot["run"]["id"] == run_id
        assert snapshot["run"]["status"] == "created"
        assert snapshot["run"]["council_model_ids"] == ["mock-model"]
        for payload in payloads:
            frame = await asyncio.wait_for(anext(response.body_iterator), timeout=2)
            assert decode_frame(frame) == payload
    finally:
        await response.body_iterator.aclose()
    assert run_id not in broker._subs


async def test_idle_stream_emits_heartbeat_then_resumes_events(stream_context, monkeypatch):
    main, broker, run_id = stream_context
    response = await main.stream(run_id)
    original_wait_for = asyncio.wait_for
    timeouts = []

    async def expire_once(awaitable, timeout):
        timeouts.append(timeout)
        if len(timeouts) == 1:
            awaitable.close()
            raise asyncio.TimeoutError
        return await original_wait_for(awaitable, timeout=timeout)

    monkeypatch.setattr(main.asyncio, "wait_for", expire_once)
    try:
        await anext(response.body_iterator)  # snapshot
        assert await anext(response.body_iterator) == ":\n\n"
        payload = {"type": "progress", "completed_chunks": 2}
        await broker.publish(run_id, payload)
        assert decode_frame(await anext(response.body_iterator)) == payload
        assert timeouts == [15.0, 15.0]
    finally:
        await response.body_iterator.aclose()
    assert run_id not in broker._subs


async def test_cancelling_waiting_stream_removes_subscription(stream_context, monkeypatch):
    main, broker, run_id = stream_context
    response = await main.stream(run_id)
    waiting = asyncio.Event()
    original_wait_for = asyncio.wait_for

    async def observed_wait_for(awaitable, timeout):
        waiting.set()
        return await original_wait_for(awaitable, timeout=timeout)

    monkeypatch.setattr(main.asyncio, "wait_for", observed_wait_for)
    await anext(response.body_iterator)
    pending = asyncio.create_task(anext(response.body_iterator))
    try:
        await original_wait_for(waiting.wait(), timeout=2)
        pending.cancel()
        with pytest.raises(asyncio.CancelledError):
            await pending
        assert run_id not in broker._subs
    finally:
        if not pending.done():
            pending.cancel()
            await asyncio.gather(pending, return_exceptions=True)
        await response.body_iterator.aclose()


async def test_closing_one_stream_preserves_other_subscribers(stream_context):
    main, broker, run_id = stream_context
    survivor = await broker.subscribe(run_id)
    other_run = await broker.subscribe("other-run")
    response = await main.stream(run_id)
    try:
        await anext(response.body_iterator)
    finally:
        await response.body_iterator.aclose()
    payload = {"type": "still-connected"}
    await broker.publish(run_id, payload)
    assert survivor.get_nowait() == payload
    assert other_run.empty()
    assert broker._subs[run_id] == {survivor}
    await broker.unsubscribe(run_id, survivor)
    await broker.unsubscribe("other-run", other_run)
    assert broker._subs == {}


async def test_snapshot_failure_still_removes_subscription(stream_context, monkeypatch):
    main, broker, run_id = stream_context

    def broken_snapshot(run):
        raise RuntimeError("synthetic snapshot failure")

    monkeypatch.setattr(main, "_run_out", broken_snapshot)
    response = await main.stream(run_id)
    try:
        with pytest.raises(RuntimeError, match="synthetic snapshot failure"):
            await anext(response.body_iterator)
    finally:
        await response.body_iterator.aclose()
    assert run_id not in broker._subs
