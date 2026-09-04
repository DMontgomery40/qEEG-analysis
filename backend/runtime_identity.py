from __future__ import annotations

import hashlib
import json
import re
import subprocess
import uuid
from datetime import datetime, timezone
from pathlib import Path


MODEL_CONTRACT_VERSION = 1
PROCESS_STARTED_AT = datetime.now(timezone.utc).isoformat()
INSTANCE_ID = str(uuid.uuid4())


def _source_commit() -> str:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[1],
            capture_output=True,
            text=True,
            timeout=2,
            check=True,
        )
        commit = completed.stdout.strip().lower()
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    return commit if re.fullmatch(r"[0-9a-f]{40}", commit) else "unknown"


SOURCE_COMMIT = _source_commit()


def model_catalogue_fingerprint(model_ids: list[str] | set[str]) -> str:
    normalized = sorted({str(model_id).strip() for model_id in model_ids if str(model_id).strip()})
    payload = json.dumps(normalized, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def current_runtime_identity(model_ids: list[str] | set[str]) -> dict[str, object]:
    return {
        "source_commit": SOURCE_COMMIT,
        "process_started_at": PROCESS_STARTED_AT,
        "instance_id": INSTANCE_ID,
        "model_contract_version": MODEL_CONTRACT_VERSION,
        "available_model_ids": sorted(set(model_ids)),
        "model_catalogue_fingerprint": model_catalogue_fingerprint(model_ids),
    }
