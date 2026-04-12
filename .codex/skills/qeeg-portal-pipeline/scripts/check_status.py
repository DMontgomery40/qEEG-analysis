#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Show local qEEG portal pipeline worker status for a patient.")
    parser.add_argument("patient_id")
    parser.add_argument(
        "--status-dir",
        default="/Users/davidmontgomery/qEEG-analysis/data/pipeline_jobs",
    )
    args = parser.parse_args()

    path = Path(args.status_dir) / f"{args.patient_id}.json"
    if not path.exists():
        print(json.dumps({"patient_id": args.patient_id, "status": "missing"}))
        return 1
    print(path.read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
