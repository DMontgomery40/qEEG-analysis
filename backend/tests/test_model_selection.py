from __future__ import annotations

from backend.model_selection import resolve_model_preference


def test_resolve_model_preference_maps_gemini_31_to_legacy_preview_id():
    discovered = ["gemini-3-pro-preview"]

    assert resolve_model_preference("gemini-3.1-pro-preview", discovered) == "gemini-3-pro-preview"


def test_resolve_model_preference_maps_legacy_gemini_preview_to_current_id():
    discovered = ["google/gemini-3.1-pro-preview"]

    assert (
        resolve_model_preference("google/gemini-3-pro-preview", discovered)
        == "google/gemini-3.1-pro-preview"
    )


def test_resolve_model_preference_does_not_downgrade_gpt54_to_plain_gpt5():
    discovered = ["gpt-5"]

    assert resolve_model_preference("gpt-5.4", discovered) is None
