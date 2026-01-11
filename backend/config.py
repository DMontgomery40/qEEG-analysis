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
ARTIFACTS_DIR = DATA_DIR / "artifacts"
EXPORTS_DIR = DATA_DIR / "exports"


EndpointPreference = Literal["chat", "responses"]


@dataclass(frozen=True)
class CouncilModelConfig:
    id: str
    name: str
    source: str
    endpoint_preference: EndpointPreference = "chat"


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


COUNCIL_MODELS: list[CouncilModelConfig] = _load_models_from_env() or [
    # These IDs are examples only; runtime model discovery is the source of truth.
    CouncilModelConfig(
        id="gpt-4.1-mini",
        name="GPT (example)",
        source="Subscription via CLIProxyAPI",
        endpoint_preference="chat",
    ),
    CouncilModelConfig(
        id="claude-sonnet-4.5",
        name="Claude Sonnet (example)",
        source="Subscription via CLIProxyAPI",
        endpoint_preference="chat",
    ),
]

DEFAULT_CONSOLIDATOR = os.getenv("DEFAULT_CONSOLIDATOR", "")

DISCOVERED_MODEL_IDS: set[str] = set()


def set_discovered_model_ids(model_ids: list[str]) -> None:
    DISCOVERED_MODEL_IDS.clear()
    DISCOVERED_MODEL_IDS.update(model_ids)


def ensure_data_dirs() -> None:
    for d in (DATA_DIR, REPORTS_DIR, ARTIFACTS_DIR, EXPORTS_DIR):
        d.mkdir(parents=True, exist_ok=True)
