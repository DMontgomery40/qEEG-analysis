from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..storage import Artifact


def _stage_artifacts(session, run_id: str, stage_num: int, *, kind: str) -> list[Artifact]:
    from sqlalchemy import select

    return list(
        session.scalars(
            select(Artifact)
            .where(Artifact.run_id == run_id, Artifact.stage_num == stage_num, Artifact.kind == kind)
            .order_by(Artifact.created_at.asc())
        )
    )


def _validate_stage5(payload: Any) -> None:
    if not isinstance(payload, dict):
        raise ValueError("Stage 5 payload must be an object")
    vote = payload.get("vote")
    if vote not in {"APPROVE", "REVISE"}:
        raise ValueError("Stage 5 vote must be APPROVE or REVISE")
    for key in ("required_changes", "optional_changes"):
        val = payload.get(key)
        if not isinstance(val, list) or not all(isinstance(x, str) for x in val):
            raise ValueError(f"Stage 5 {key} must be a string[]")
    score = payload.get("quality_score_1to10")
    if not isinstance(score, int) or not (1 <= score <= 10):
        raise ValueError("Stage 5 quality_score_1to10 must be int 1..10")


def _aggregate_required_changes(stage5_artifacts: list[Artifact]) -> list[str]:
    changes: list[str] = []
    seen: set[str] = set()
    for art in stage5_artifacts:
        try:
            payload = json.loads(Path(art.content_path).read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(payload, dict):
            continue
        for c in payload.get("required_changes", []) or []:
            if isinstance(c, str):
                c2 = c.strip()
                if c2 and c2 not in seen:
                    seen.add(c2)
                    changes.append(c2)
    return changes

