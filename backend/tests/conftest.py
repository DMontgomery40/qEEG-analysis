"""Pytest configuration and shared fixtures for backend tests.

Key fixtures:
- temp_data_dir: Isolated temporary directory for all data paths
- mock_llm_client: AsyncOpenAICompatClient with mocked CLIProxyAPI transport
- example_pdf_path: Path to real qEEG PDF for extraction tests
"""

from __future__ import annotations

from pathlib import Path

import pytest

# Import backend modules AFTER patching in fixtures


@pytest.fixture
def temp_data_dir(tmp_path: Path, monkeypatch):
    """Redirect all data paths to a temporary directory.

    This fixture patches all modules that import config paths at module level:
    - backend.config (the source)
    - backend.storage (imports DATA_DIR)
    - backend.reports (imports REPORTS_DIR)
    - backend.council (imports ARTIFACTS_DIR)

    Must be used before any test that creates patients, reports, runs, or artifacts.
    """
    from backend import config, council, patient_files, reports, storage

    # Patch config module paths
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    monkeypatch.setattr(config, "REPORTS_DIR", tmp_path / "reports")
    monkeypatch.setattr(config, "PATIENT_FILES_DIR", tmp_path / "patient_files")
    monkeypatch.setattr(config, "ARTIFACTS_DIR", tmp_path / "artifacts")
    monkeypatch.setattr(config, "EXPORTS_DIR", tmp_path / "exports")

    # Patch storage module's cached import of DATA_DIR
    monkeypatch.setattr(storage, "DATA_DIR", tmp_path)

    # Patch reports module's cached import of REPORTS_DIR
    monkeypatch.setattr(reports, "REPORTS_DIR", tmp_path / "reports")

    # Patch patient_files module's cached import of PATIENT_FILES_DIR
    monkeypatch.setattr(patient_files, "PATIENT_FILES_DIR", tmp_path / "patient_files")

    # Patch council module's cached import of ARTIFACTS_DIR
    monkeypatch.setattr(council, "ARTIFACTS_DIR", tmp_path / "artifacts")

    # Reset engine to use temp database
    storage.reset_engine(f"sqlite:///{tmp_path / 'app.db'}")

    # Create directories
    config.ensure_data_dirs()

    # Initialize fresh database
    storage.init_db()

    return tmp_path


@pytest.fixture
def mock_llm_client(temp_data_dir):
    """Create AsyncOpenAICompatClient with mock CLIProxyAPI transport.

    Depends on temp_data_dir to ensure proper isolation.
    Returns a client that responds with stage-appropriate mock data.
    """
    from backend.llm_client import AsyncOpenAICompatClient
    from backend.tests.fixtures.mock_llm import create_mock_transport

    return AsyncOpenAICompatClient(
        base_url="http://mock-cliproxy:8317",
        api_key="",
        transport=create_mock_transport(),
    )


@pytest.fixture
def example_pdf_path() -> Path:
    """Path to a real qEEG PDF for extraction testing.

    This uses the example PDF in the examples/ directory.
    Tests should skip if the file doesn't exist.
    """
    path = Path(__file__).parents[2] / "examples" / "Longitudinal_qEEG_Report_WAVi_Cleaned_v3.pdf"
    if not path.exists():
        pytest.skip(f"Example PDF not found at {path}")
    return path


@pytest.fixture
def example_pdf_bytes(example_pdf_path: Path) -> bytes:
    """Read the example PDF as bytes."""
    return example_pdf_path.read_bytes()
