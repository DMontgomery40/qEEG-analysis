from __future__ import annotations

import re
from pathlib import Path

from fastapi.testclient import TestClient


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
