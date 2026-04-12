#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path


def main() -> int:
    try:
        payload = json.load(sys.stdin)
    except Exception:
        payload = {}
    cwd = Path(str(payload.get("cwd") or ".")).resolve()
    repo = Path(__file__).resolve().parents[2]
    try:
        cwd.relative_to(repo)
    except ValueError:
        return 0

    message = (
        "qEEG portal pipeline guardrail: clinic report uploads must become durable "
        "pipeline jobs. Before editing upload, portal sync, batch runner, or pipeline "
        "worker code, read plan.md and project-local memory. Use real runs, not mock "
        "mode, for incident validation."
    )
    print(
        json.dumps(
            {
                "hookSpecificOutput": {
                    "hookEventName": "SessionStart",
                    "additionalContext": message,
                }
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
