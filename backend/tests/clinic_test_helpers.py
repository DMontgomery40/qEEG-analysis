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


@pytest.fixture(autouse=True)
def configured_models_discovered():
    """A running engine has discovered its configured council and consolidator.

    The shared conftest empties the discovered set for isolation; since HUB-H6
    an empty set means analysis is unavailable, so a test that confirms an
    analysis starts from an engine that has found its models. A test that sets
    the set itself still wins."""
    from backend import config

    config.DISCOVERED_MODEL_IDS.update(
        [m.id for m in config.COUNCIL_MODELS] + [config.DEFAULT_CONSOLIDATOR]
    )
    yield
