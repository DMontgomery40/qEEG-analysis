"""Tests for /api/models UI filtering policy."""

from __future__ import annotations

import asyncio


def test_model_visible_in_ui_filters_legacy_provider_versions(temp_data_dir):
    from backend.main import _model_visible_in_ui

    # OpenAI: keep only 5.2 / 5.3 / 5.4 families (+codex, allowed effort tiers).
    assert _model_visible_in_ui("gpt-5") is False
    assert _model_visible_in_ui("gpt-5.1") is False
    assert _model_visible_in_ui("gpt-5-codex") is False
    assert _model_visible_in_ui("gpt-5.1-codex-max") is False
    assert _model_visible_in_ui("gpt-5.2") is True
    assert _model_visible_in_ui("openai/gpt-5.2-codex") is True
    assert _model_visible_in_ui("gpt-5.3-codex-high") is True
    assert _model_visible_in_ui("gpt-5.3-xhigh") is True
    assert _model_visible_in_ui("gpt-5.4") is True
    assert _model_visible_in_ui("gpt-5.4-pro") is True
    assert _model_visible_in_ui("gpt-5.3-low") is False
    assert _model_visible_in_ui("gpt-5.2-codex-mini") is False

    # Anthropic: keep >= 4.6
    assert _model_visible_in_ui("claude-opus-4-20250514") is False
    assert _model_visible_in_ui("claude-opus-4-1-20250805") is False
    assert _model_visible_in_ui("claude-sonnet-4-5-20250929") is False
    assert _model_visible_in_ui("claude-sonnet-4-6-20260101") is True
    assert _model_visible_in_ui("claude-sonnet-4-6") is True
    assert _model_visible_in_ui("claude-opus-4-6") is True
    assert _model_visible_in_ui("claude-opus-4.6") is True
    assert _model_visible_in_ui("claude-3-7-sonnet-20250219") is False

    # Gemini: keep >= 2.5 (including 3.x previews).
    assert _model_visible_in_ui("gemini-2.0-flash") is False
    assert _model_visible_in_ui("gemini-2.5-pro") is True
    assert _model_visible_in_ui("google/gemini-3-flash-preview") is True

    # Unknown ids: keep visible
    assert _model_visible_in_ui("some-custom-model") is True


def test_models_endpoint_reports_configured_availability_from_real_discovery(monkeypatch):
    from backend.config import CouncilModelConfig, set_discovered_model_ids
    import backend.main as main

    monkeypatch.setattr(
        main,
        "COUNCIL_MODELS",
        [
            CouncilModelConfig(id="gpt-5.4", name="GPT-5.4", source="test"),
            CouncilModelConfig(id="gemini-3.1-pro-preview", name="Gemini", source="test"),
        ],
    )
    set_discovered_model_ids(["openai/gpt-5.4", "google/gemini-3-pro-preview"])

    payload = asyncio.run(main.models())
    configured = {item["id"]: item for item in payload["configured_models"]}

    assert payload["discovered_models"] == ["google/gemini-3-pro-preview", "openai/gpt-5.4"]
    assert configured["gpt-5.4"]["available"] is True
    assert configured["gpt-5.4"]["resolved_discovered_id"] == "openai/gpt-5.4"
    assert configured["gemini-3.1-pro-preview"]["available"] is True
    assert configured["gemini-3.1-pro-preview"]["resolved_discovered_id"] == "google/gemini-3-pro-preview"
