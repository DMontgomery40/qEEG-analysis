#!/usr/bin/env python
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from backend.config import CLIPROXY_API_KEY, CLIPROXY_BASE_URL, set_discovered_model_ids  # noqa: E402
from backend.council import QEEGCouncilWorkflow  # noqa: E402
from backend.llm_client import AsyncOpenAICompatClient  # noqa: E402
from backend.storage import get_run, session_scope  # noqa: E402


async def _main(run_id: str) -> int:
    llm = AsyncOpenAICompatClient(
        base_url=CLIPROXY_BASE_URL,
        api_key=CLIPROXY_API_KEY,
        timeout_s=600.0,
    )
    try:
        discovered = await llm.list_models()
        set_discovered_model_ids(discovered)
        workflow = QEEGCouncilWorkflow(llm=llm)

        async def on_event(payload: dict) -> None:
            print(json.dumps(payload), flush=True)

        await workflow.run_pipeline(run_id, on_event=on_event)
    finally:
        await llm.aclose()

    with session_scope() as session:
        run = get_run(session, run_id)
        if run is None:
            print(f"Run not found after execution: {run_id}", file=sys.stderr)
            return 1
        if run.status != "complete":
            print(
                f"Run {run_id} finished with status={run.status}: {run.error_message or run.status}",
                file=sys.stderr,
            )
            return 1
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Run an existing qEEG Council run by id.")
    parser.add_argument("run_id")
    args = parser.parse_args()
    return asyncio.run(_main(args.run_id))


if __name__ == "__main__":
    raise SystemExit(main())
