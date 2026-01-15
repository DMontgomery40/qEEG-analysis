from __future__ import annotations

from .types import StageDef

STAGES: list[StageDef] = [
    StageDef(1, "initial_analysis", "analysis", "text/markdown", ".md"),
    StageDef(2, "peer_review", "peer_review", "application/json", ".json"),
    StageDef(3, "revision", "revision", "text/markdown", ".md"),
    StageDef(4, "consolidation", "consolidation", "text/markdown", ".md"),
    StageDef(5, "final_review", "final_review", "application/json", ".json"),
    StageDef(6, "final_draft", "final_draft", "text/markdown", ".md"),
]

DATA_PACK_SCHEMA_VERSION = 1
DATA_PACK_FILENAME = "_data_pack.json"
VISION_TRANSCRIPT_FILENAME = "_vision_transcript.md"

