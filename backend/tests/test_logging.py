from __future__ import annotations

import importlib
import io
import logging
import re
from pathlib import Path

from fastapi.testclient import TestClient
import pytest


def _test_app(temp_data_dir, monkeypatch):
    monkeypatch.setenv("QEEG_MOCK_LLM", "1")
    from backend import main

    monkeypatch.setattr(
        main,
        "_ensure_project_clipr_config",
        lambda: Path(temp_data_dir) / "cliproxyapi.conf",
    )
    monkeypatch.setattr(main, "_sync_home_auth_to_project", lambda: 0)
    return main.app


def test_request_id_header_echoes_client_value(temp_data_dir, monkeypatch):
    app = _test_app(temp_data_dir, monkeypatch)

    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.get("/api/health", headers={"X-Request-ID": "req-test-123"})

    assert response.status_code == 200
    assert response.headers["x-request-id"] == "req-test-123"


def test_request_id_header_is_generated_when_missing(temp_data_dir, monkeypatch):
    app = _test_app(temp_data_dir, monkeypatch)

    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.get("/api/health")

    generated = response.headers.get("x-request-id")
    assert response.status_code == 200
    assert generated is not None
    assert re.fullmatch(r"[0-9a-f]{32}", generated)


def test_configure_logging_does_not_reset_root_handlers():
    import backend.logging_utils as logging_utils

    logging_utils = importlib.reload(logging_utils)
    root = logging.getLogger()
    backend_logger = logging.getLogger("backend")

    original_root_handlers = list(root.handlers)
    original_backend_handlers = list(backend_logger.handlers)
    original_backend_level = backend_logger.level
    original_backend_propagate = backend_logger.propagate

    sentinel = logging.StreamHandler(io.StringIO())
    backend_sentinel = logging.StreamHandler(io.StringIO())
    root.handlers = [sentinel]
    backend_logger.handlers = [backend_sentinel]

    try:
        logging_utils.configure_logging()
        assert root.handlers == [sentinel]
        assert backend_logger.handlers == [backend_sentinel]
    finally:
        root.handlers = original_root_handlers
        backend_logger.handlers = original_backend_handlers
        backend_logger.setLevel(original_backend_level)
        backend_logger.propagate = original_backend_propagate
        setattr(logging_utils.configure_logging, "_configured", False)


def test_request_failure_logs_operator_hint(temp_data_dir, monkeypatch):
    app = _test_app(temp_data_dir, monkeypatch)
    from backend import main

    captured: dict[str, object] = {}

    def fake_exception(event: str, **kwargs):
        captured["event"] = event
        captured["kwargs"] = kwargs

    if not any(getattr(route, "path", None) == "/_test_logging_failure" for route in app.routes):
        @app.get("/_test_logging_failure")
        async def _test_logging_failure():
            raise RuntimeError("boom")

    monkeypatch.setattr(main.LOGGER, "exception", fake_exception)

    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.get("/_test_logging_failure")

    assert response.status_code == 500
    assert captured["event"] == "request_failed"
    assert "FastAPI route boundary" in captured["kwargs"]["operatorHint"]


@pytest.mark.asyncio
async def test_pipeline_failure_emits_operator_hint(temp_data_dir, mock_llm_client, monkeypatch):
    from backend import storage
    from backend.council import QEEGCouncilWorkflow

    with storage.session_scope() as session:
        patient = storage.create_patient(session, label="09-05-1954-0", notes="")
        report_dir = Path(temp_data_dir) / "reports" / patient.id / "report-1"
        report_dir.mkdir(parents=True, exist_ok=True)
        stored_path = report_dir / "original.txt"
        extracted_path = report_dir / "extracted.txt"
        stored_path.write_text("dummy", encoding="utf-8")
        extracted_path.write_text("dummy", encoding="utf-8")
        report = storage.create_report(
            session,
            report_id="report-1",
            patient_id=patient.id,
            filename="original.txt",
            mime_type="text/plain",
            stored_path=stored_path,
            extracted_text_path=extracted_path,
        )
        run = storage.create_run(
            session,
            patient_id=patient.id,
            report_id=report.id,
            council_model_ids=["mock-council-a"],
            consolidator_model_id="mock-consolidator",
        )

    workflow = QEEGCouncilWorkflow(llm=mock_llm_client)

    async def fail_stage1(*_args, **_kwargs):
        raise RuntimeError("stage 1 exploded")

    monkeypatch.setattr(workflow, "_stage1", fail_stage1)
    events: list[dict[str, object]] = []

    async def on_event(payload: dict[str, object]) -> None:
        events.append(payload)

    await workflow.run_pipeline(run.id, on_event=on_event)

    failed = [event for event in events if event.get("status") == "failed"]
    assert failed
    assert "operatorHint" in failed[0]
    assert "Stages 1-6 sequentially" in str(failed[0]["operatorHint"])
