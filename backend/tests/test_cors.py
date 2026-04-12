from __future__ import annotations

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


@pytest.mark.parametrize(
    ("origin", "expected_allowed"),
    [
        ("http://127.0.0.1:4176", True),
        ("http://localhost:5173", True),
        ("https://example.com", False),
    ],
)
def test_cors_allows_local_dev_origins_only(
    temp_data_dir, monkeypatch, origin: str, expected_allowed: bool
):
    app = _test_app(temp_data_dir, monkeypatch)

    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.options(
            "/api/health",
            headers={
                "Origin": origin,
                "Access-Control-Request-Method": "GET",
            },
        )

    if expected_allowed:
        assert response.status_code == 200
        assert response.headers["access-control-allow-origin"] == origin
    else:
        assert response.status_code == 400
        assert "access-control-allow-origin" not in response.headers
