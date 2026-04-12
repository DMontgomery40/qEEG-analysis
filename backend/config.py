"""Backend configuration for qEEG Council."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv

load_dotenv()

CLIPROXY_BASE_URL = os.getenv("CLIPROXY_BASE_URL", "http://127.0.0.1:8317").rstrip("/")
CLIPROXY_API_KEY = os.getenv("CLIPROXY_API_KEY", "")

DATA_DIR = Path(os.getenv("DATA_DIR", "data"))
REPORTS_DIR = DATA_DIR / "reports"
PATIENT_FILES_DIR = DATA_DIR / "patient_files"
ARTIFACTS_DIR = DATA_DIR / "artifacts"
EXPORTS_DIR = DATA_DIR / "exports"


EndpointPreference = Literal["chat", "responses"]


@dataclass(frozen=True)
class CouncilModelConfig:
    id: str
    name: str
    source: str
    endpoint_preference: EndpointPreference = "chat"


@dataclass(frozen=True)
class ModelRoleDefaults:
    stage1_vision: str
    stage2_review: str
    stage4_consolidator: str
    stage5_final_review: str
    stage6_final_draft: str
    patient_facing_rewrite: str


def _load_models_from_env() -> list[CouncilModelConfig] | None:
    raw = os.getenv("COUNCIL_MODELS_JSON")
    if not raw:
        return None
    try:
        items = json.loads(raw)
    except Exception:
        return None
    if not isinstance(items, list):
        return None
    out: list[CouncilModelConfig] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        model_id = item.get("id")
        name = item.get("name")
        source = item.get("source")
        endpoint_preference = item.get("endpoint_preference", "chat")
        if not (
            isinstance(model_id, str)
            and isinstance(name, str)
            and isinstance(source, str)
            and endpoint_preference in {"chat", "responses"}
        ):
            continue
        out.append(
            CouncilModelConfig(
                id=model_id,
                name=name,
                source=source,
                endpoint_preference=endpoint_preference,
            )
        )
    return out


MODEL_ROLE_DEFAULTS = ModelRoleDefaults(
    stage1_vision=os.getenv("DEFAULT_STAGE1_VISION_MODEL", "gemini-3.1-pro-preview"),
    stage2_review=os.getenv("DEFAULT_STAGE2_REVIEW_MODEL", "gpt-5.4"),
    stage4_consolidator=os.getenv(
        "DEFAULT_STAGE4_CONSOLIDATOR",
        os.getenv("DEFAULT_CONSOLIDATOR", "claude-sonnet-4-6"),
    ),
    stage5_final_review=os.getenv("DEFAULT_STAGE5_REVIEW_MODEL", "gpt-5.4"),
    stage6_final_draft=os.getenv(
        "DEFAULT_STAGE6_FINAL_DRAFT_MODEL", "claude-sonnet-4-6"
    ),
    patient_facing_rewrite=os.getenv(
        "DEFAULT_PATIENT_FACING_REWRITE_MODEL", "claude-sonnet-4-6"
    ),
)


COUNCIL_MODELS: list[CouncilModelConfig] = _load_models_from_env() or [
    CouncilModelConfig(
        id=MODEL_ROLE_DEFAULTS.stage2_review,
        name="GPT-5.4",
        source="Subscription via CLIProxyAPI",
        endpoint_preference="chat",
    ),
    CouncilModelConfig(
        id=MODEL_ROLE_DEFAULTS.stage6_final_draft,
        name="Claude Sonnet 4.6",
        source="Subscription via CLIProxyAPI",
        endpoint_preference="chat",
    ),
    CouncilModelConfig(
        id=MODEL_ROLE_DEFAULTS.stage1_vision,
        name="Gemini 3.1 Pro Preview",
        source="Subscription via CLIProxyAPI",
        endpoint_preference="chat",
    ),
]

DEFAULT_CONSOLIDATOR = MODEL_ROLE_DEFAULTS.stage4_consolidator

# Models that support vision/multimodal input (can process images)
# These will receive page images in addition to text for Stage 1 analysis
VISION_CAPABLE_MODELS: set[str] = {
    # OpenAI vision models (GPT-4o, GPT-4-turbo, GPT-5+ all support vision)
    "gpt-4o",
    "gpt-4o-mini",
    "gpt-4-vision-preview",
    "gpt-4-turbo",
    "gpt-5.2",
    "gpt-5.3",
    "gpt-5.4",
    "gpt-5.2-codex",
    "gpt-5.3-codex",
    "gpt-5.4-codex",
    "openai/gpt-4o",
    "openai/gpt-4o-mini",
    "openai/gpt-4-vision-preview",
    "openai/gpt-4-turbo",
    "openai/gpt-5.2",
    "openai/gpt-5.3",
    "openai/gpt-5.4",
    "openai/gpt-5.2-codex",
    "openai/gpt-5.3-codex",
    "openai/gpt-5.4-codex",
    # Anthropic Claude models (all Claude 3+ support vision)
    "claude-3-opus",
    "claude-3-sonnet",
    "claude-3-haiku",
    "claude-3.5-sonnet",
    "claude-3-5-haiku",
    "claude-3-7-sonnet",
    "claude-sonnet-4",
    "claude-opus-4",
    "claude-sonnet-4-6",
    "claude-opus-4-6",
    "claude-haiku-4-6",
    "anthropic/claude-3-opus",
    "anthropic/claude-3-sonnet",
    "anthropic/claude-3-haiku",
    "anthropic/claude-3.5-sonnet",
    "anthropic/claude-sonnet-4",
    "anthropic/claude-opus-4",
    "anthropic/claude-sonnet-4-6",
    "anthropic/claude-opus-4-6",
    # Google Gemini models (2.5/3.x families and compatible historical ids)
    "gemini-1.5-pro",
    "gemini-1.5-flash",
    "gemini-2.0-flash",
    "gemini-2.5-pro",
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
    "gemini-3.1-pro-preview",
    "gemini-3-pro",
    "gemini-3-flash",
    "gemini-3-pro-preview",
    "gemini-3-flash-preview",
    "google/gemini-1.5-pro",
    "google/gemini-1.5-flash",
    "google/gemini-2.0-flash",
    "google/gemini-2.5-pro",
    "google/gemini-2.5-flash",
    "google/gemini-2.5-flash-lite",
    "google/gemini-3.1-pro-preview",
    "google/gemini-3-pro-preview",
    "google/gemini-3-flash-preview",
    # Legacy compatibility ids kept for older live inventories.
    "gemini-pro-vision",
    "google/gemini-pro-vision",
}


def is_vision_capable(model_id: str) -> bool:
    """Check if a model supports vision/multimodal input."""
    # Check exact match
    if model_id in VISION_CAPABLE_MODELS:
        return True
    # Check if any vision model ID is a substring (handles variations)
    model_lower = model_id.lower()
    for vision_model in VISION_CAPABLE_MODELS:
        if vision_model.lower() in model_lower:
            return True
    return False


DISCOVERED_MODEL_IDS: set[str] = set()


def set_discovered_model_ids(model_ids: list[str]) -> None:
    DISCOVERED_MODEL_IDS.clear()
    DISCOVERED_MODEL_IDS.update(model_ids)


def ensure_data_dirs() -> None:
    for d in (DATA_DIR, REPORTS_DIR, PATIENT_FILES_DIR, ARTIFACTS_DIR, EXPORTS_DIR):
        d.mkdir(parents=True, exist_ok=True)
