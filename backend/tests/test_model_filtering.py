"""Tests for /api/models UI filtering policy."""

from __future__ import annotations


def test_model_visible_in_ui_filters_legacy_provider_versions(temp_data_dir):
    from backend.main import _model_visible_in_ui

    # OpenAI: keep >= 5.1
    assert _model_visible_in_ui("gpt-5") is False
    assert _model_visible_in_ui("gpt-5.1") is True
    assert _model_visible_in_ui("gpt-5.1-codex-max") is True
    assert _model_visible_in_ui("openai/gpt-5.2-codex") is True

    # Anthropic: keep >= 4.5
    assert _model_visible_in_ui("claude-opus-4-20250514") is False
    assert _model_visible_in_ui("claude-opus-4-1-20250805") is False
    assert _model_visible_in_ui("claude-sonnet-4-5-20250929") is True
    assert _model_visible_in_ui("claude-3-7-sonnet-20250219") is False

    # Gemini: keep >= 3.x
    assert _model_visible_in_ui("gemini-2.5-pro") is False
    assert _model_visible_in_ui("google/gemini-3-flash-preview") is True

    # Unknown ids: keep visible
    assert _model_visible_in_ui("some-custom-model") is True

