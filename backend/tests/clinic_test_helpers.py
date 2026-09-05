"""Guards for free clinic integration tests; actual loopback HTTP remains available."""

import pytest


@pytest.fixture(autouse=True)
def forbid_clinic_paid(monkeypatch):
    from backend.llm_client import AsyncOpenAICompatClient
    from backend.paid_transport import PaidAsyncTransport, PaidSyncTransport

    calls = []

    def forbidden(*args, **kwargs):
        calls.append("paid")
        pytest.fail("Free clinic filing must not call paid/provider transports")

    for name in ("chat_completions", "responses", "list_models"):
        monkeypatch.setattr(AsyncOpenAICompatClient, name, forbidden)
    monkeypatch.setattr(PaidAsyncTransport, "handle_async_request", forbidden)
    monkeypatch.setattr(PaidSyncTransport, "handle_request", forbidden)
    yield
    assert calls == []
