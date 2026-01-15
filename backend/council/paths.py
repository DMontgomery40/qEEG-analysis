from __future__ import annotations

import random
from pathlib import Path

from .. import config
from .constants import DATA_PACK_FILENAME, VISION_TRANSCRIPT_FILENAME


def _stage_dir(run_id: str, stage_num: int) -> Path:
    return config.ARTIFACTS_DIR / run_id / f"stage-{stage_num}"


def _artifact_path(run_id: str, stage_num: int, model_id: str, ext: str) -> Path:
    # model_id is safe enough for filenames in practice; fall back to a stable replacement if needed.
    safe_model = "".join(c if c.isalnum() or c in {"-", "_", "."} else "_" for c in model_id) or "model"
    return _stage_dir(run_id, stage_num) / f"{safe_model}{ext}"


def _data_pack_path(run_id: str) -> Path:
    return _stage_dir(run_id, 1) / DATA_PACK_FILENAME


def _vision_transcript_path(run_id: str) -> Path:
    return _stage_dir(run_id, 1) / VISION_TRANSCRIPT_FILENAME


def _stable_label_map(run_id: str, model_ids: list[str]) -> dict[str, str]:
    rng = random.Random(run_id)
    ids = list(model_ids)
    rng.shuffle(ids)
    labels = [chr(ord("A") + i) for i in range(len(ids))]
    return {label: model_id for label, model_id in zip(labels, ids)}
