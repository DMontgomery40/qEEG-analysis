from __future__ import annotations

import importlib


def _reload_config(monkeypatch):
    import backend.config as config

    for key in (
        "COUNCIL_MODELS_JSON",
        "DEFAULT_STAGE1_VISION_MODEL",
        "DEFAULT_STAGE2_REVIEW_MODEL",
        "DEFAULT_STAGE4_CONSOLIDATOR",
        "DEFAULT_STAGE5_REVIEW_MODEL",
        "DEFAULT_STAGE6_FINAL_DRAFT_MODEL",
        "DEFAULT_PATIENT_FACING_REWRITE_MODEL",
        "DEFAULT_CONSOLIDATOR",
    ):
        monkeypatch.delenv(key, raising=False)

    return importlib.reload(config)


def test_model_role_defaults_are_quality_first_by_role(monkeypatch):
    config = _reload_config(monkeypatch)

    assert config.MODEL_ROLE_DEFAULTS.stage1_vision == "gemini-3.1-pro-preview"
    assert config.MODEL_ROLE_DEFAULTS.stage2_review == "gpt-5.4"
    assert config.MODEL_ROLE_DEFAULTS.stage4_consolidator == "claude-sonnet-4-6"
    assert config.MODEL_ROLE_DEFAULTS.stage5_final_review == "gpt-5.4"
    assert config.MODEL_ROLE_DEFAULTS.stage6_final_draft == "claude-sonnet-4-6"
    assert config.MODEL_ROLE_DEFAULTS.patient_facing_rewrite == "claude-sonnet-4-6"
    assert config.DEFAULT_CONSOLIDATOR == "claude-sonnet-4-6"
    assert [model.id for model in config.COUNCIL_MODELS] == [
        "gpt-5.4",
        "claude-sonnet-4-6",
        "gemini-3.1-pro-preview",
    ]


def test_is_vision_capable_includes_current_gemini_preview_ids(monkeypatch):
    config = _reload_config(monkeypatch)

    assert config.is_vision_capable("gemini-3.1-pro-preview") is True
    assert config.is_vision_capable("google/gemini-3.1-pro-preview") is True
