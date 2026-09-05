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

    assert config.MODEL_ROLE_DEFAULTS.stage1_vision == "z-ai/glm-5.3-flash"
    assert config.MODEL_ROLE_DEFAULTS.stage2_review == "z-ai/glm-5.3-flash"
    assert config.MODEL_ROLE_DEFAULTS.stage4_consolidator == "z-ai/glm-5.3-flash"
    assert config.MODEL_ROLE_DEFAULTS.stage5_final_review == "z-ai/glm-5.3-flash"
    assert config.MODEL_ROLE_DEFAULTS.stage6_final_draft == "z-ai/glm-5.3-flash"
    assert config.MODEL_ROLE_DEFAULTS.patient_facing_rewrite == "z-ai/glm-5.3-flash"
    assert config.DEFAULT_CONSOLIDATOR == "z-ai/glm-5.3-flash"
    assert [model.id for model in config.COUNCIL_MODELS] == [
        "deepseek-v4-flash",
        "z-ai/glm-5.3-flash",
        "openai/gpt-5.6-terra",
    ]
    assert [model.endpoint_preference for model in config.COUNCIL_MODELS] == [
        "chat",
        "chat",
        "responses",
    ]


def test_role_model_falls_back_only_when_primary_is_not_discovered(monkeypatch):
    config = _reload_config(monkeypatch)

    discovered = ["z-ai/glm-5.3-flash", "z-ai/glm-5.2"]
    assert (
        config.resolve_role_model("z-ai/glm-5.3-flash", "z-ai/glm-5.2", discovered)
        == "z-ai/glm-5.3-flash"
    )
    assert (
        config.resolve_role_model(
            "z-ai/glm-5.3-flash", "z-ai/glm-5.2", ["z-ai/glm-5.2"]
        )
        == "z-ai/glm-5.2"
    )
    assert (
        config.resolve_role_model("z-ai/glm-5.3-flash", "z-ai/glm-5.2", [])
        == "z-ai/glm-5.3-flash"
    )


def test_is_vision_capable_includes_current_gemini_preview_ids(monkeypatch):
    config = _reload_config(monkeypatch)

    assert config.is_vision_capable("gemini-3.1-pro-preview") is True
    assert config.is_vision_capable("google/gemini-3.1-pro-preview") is True
    assert config.is_vision_capable("gemini-3.1-flash-lite-preview") is True
    assert config.is_vision_capable("google/gemini-3.1-flash-lite-preview") is True
    assert config.is_vision_capable("z-ai/glm-5.3-flash") is True
