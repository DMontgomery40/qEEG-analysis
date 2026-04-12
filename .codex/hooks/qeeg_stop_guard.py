#!/usr/bin/env python3
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


WATCHED_PREFIXES = (
    "scripts/portal_pipeline_worker.py",
    "scripts/run_portal_council_batch.py",
    "backend/portal_sync.py",
    ".codex/hooks/",
    ".codex/skills/qeeg-portal-pipeline/",
    "plan.md",
)


def _changed_files(repo: Path) -> list[str]:
    proc = subprocess.run(
        ["git", "status", "--short"],
        cwd=str(repo),
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        return []
    files: list[str] = []
    for line in proc.stdout.splitlines():
        if len(line) < 4:
            continue
        path = line[3:].strip()
        if " -> " in path:
            path = path.split(" -> ", 1)[1].strip()
        files.append(path)
    return files


def _memory_updated(repo: Path) -> bool:
    memory_root = Path.home() / ".codex" / "projects" / "-Users-davidmontgomery-qEEG-analysis"
    index = memory_root / "MEMORY.md"
    if not index.exists():
        return False
    text = index.read_text(encoding="utf-8", errors="replace")
    if "Portal Upload Pipeline Reliability" in text:
        return True
    memory_dir = memory_root / "memory"
    if not memory_dir.exists():
        return False
    for path in memory_dir.glob("*.md"):
        body = path.read_text(encoding="utf-8", errors="replace")
        if "Portal Upload Pipeline Reliability" in body:
            return True
    return False


def main() -> int:
    try:
        payload = json.load(sys.stdin)
    except Exception:
        payload = {}
    if payload.get("stop_hook_active"):
        print(json.dumps({"continue": True}))
        return 0

    repo = Path(__file__).resolve().parents[2]
    changed = _changed_files(repo)
    relevant = [path for path in changed if path.startswith(WATCHED_PREFIXES)]
    if not relevant:
        print(json.dumps({"continue": True}))
        return 0

    if not (repo / "plan.md").exists():
        print(
            json.dumps(
                {
                    "decision": "block",
                    "reason": "qEEG portal pipeline files changed, but plan.md is missing. Create or restore the plan before stopping.",
                }
            )
        )
        return 0

    if not _memory_updated(repo):
        print(
            json.dumps(
                {
                    "decision": "block",
                    "reason": "qEEG portal pipeline files changed. Update project-local memory and link it from MEMORY.md before stopping.",
                }
            )
        )
        return 0

    print(json.dumps({"continue": True}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
