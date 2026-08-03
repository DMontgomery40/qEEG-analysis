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

    assert config.MODEL_ROLE_DEFAULTS.stage1_vision == "openai/gpt-5.6-sol"
    assert config.MODEL_ROLE_DEFAULTS.stage2_review == "openai/gpt-5.6-sol"
    assert config.MODEL_ROLE_DEFAULTS.stage4_consolidator == "z-ai/glm-5.2"
    assert config.MODEL_ROLE_DEFAULTS.stage5_final_review == "openai/gpt-5.6-sol"
    assert config.MODEL_ROLE_DEFAULTS.stage6_final_draft == "z-ai/glm-5.2"
    assert config.MODEL_ROLE_DEFAULTS.patient_facing_rewrite == "z-ai/glm-5.2"
    assert config.DEFAULT_CONSOLIDATOR == "z-ai/glm-5.2"
    assert [model.id for model in config.COUNCIL_MODELS] == [
        "openai/gpt-5.6-sol",
        "openai/gpt-5.6-terra",
    ]
    assert all(
        model.endpoint_preference == "responses"
        for model in config.COUNCIL_MODELS
    )


def test_is_vision_capable_includes_current_gemini_preview_ids(monkeypatch):
    config = _reload_config(monkeypatch)

    assert config.is_vision_capable("gemini-3.1-pro-preview") is True
    assert config.is_vision_capable("google/gemini-3.1-pro-preview") is True
    assert config.is_vision_capable("gemini-3.1-flash-lite-preview") is True
    assert config.is_vision_capable("google/gemini-3.1-flash-lite-preview") is True
