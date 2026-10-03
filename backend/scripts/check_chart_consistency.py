"""Print every chart whose contents disagree with each other, as JSON.

    uv run python -m backend.scripts.check_chart_consistency
    uv run python -m backend.scripts.check_chart_consistency --cache /tmp/cc.json

Read-only against the database; no model calls. Parsed report facts are
cached by the sha256 of each report's text (default: the data folder's
``chart_consistency_cache.json``; ``--no-cache`` reads every report afresh).
The workbench health check runs this every 30 minutes and emails David about
any violation key not on its acknowledged list. Exit 0 when the check ran,
whatever it found; 2 when it could not run.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    from backend import config
    from backend.chart_consistency import CACHE_NAME, check

    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--data-dir", type=Path, default=Path(config.DATA_DIR), help="engine data folder")
    parser.add_argument("--db", type=Path, help="database (default: DATA_DIR/app.db)")
    parser.add_argument("--root", type=Path, default=Path(config.REPO_ROOT), help="base for relative report paths")
    parser.add_argument("--cache", type=Path, help=f"parsed-facts cache (default: DATA_DIR/{CACHE_NAME})")
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--patient", action="append", default=[], help="only violations naming this clinic id")
    args = parser.parse_args(argv)

    started = time.monotonic()
    db_path = args.db or args.data_dir / "app.db"
    cache = None if args.no_cache else (args.cache or args.data_dir / CACHE_NAME)
    try:
        result = check(db_path, root=args.root, data_dir=args.data_dir, cache_path=cache)
    except Exception as error:
        print(json.dumps({"ok": False, "error": f"{type(error).__name__}: {error}"}))
        return 2
    violations = result["violations"]
    if args.patient:
        wanted = set(args.patient)
        violations = [v for v in violations if wanted & set(v["patients"])]
    print(json.dumps({
        "ok": True,
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "seconds": round(time.monotonic() - started, 2),
        "charts": result["charts"],
        "reports": result["reports"],
        "unreadable_reports": result["unreadable_reports"],
        "charts_not_checked": result["charts_not_checked"],
        "violations": violations,
    }, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
