"""Resume an original admitted run from complete, immutable paid receipts."""

import argparse
import os
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        required=True,
        type=Path,
        help="Existing engine data directory containing app.db",
    )
    parser.add_argument("run_id", help="Original run ID to reconcile")
    args = parser.parse_args()
    data_dir = args.data_dir.resolve()
    if not (data_dir / "app.db").is_file():
        parser.error("data directory must contain an existing app.db")
    # Select storage before importing the engine configuration. No schema,
    # credentials, new admission, or provider dispatch is created by this command.
    os.environ["DATA_DIR"] = str(data_dir)
    from backend import storage
    from backend.paid_transport import reconcile_blocked_run
    from backend.run_execution import ExecutionStore

    try:
        if not reconcile_blocked_run(ExecutionStore(storage.engine), args.run_id):
            print(
                "Run is not eligible for paid-outcome recovery or is currently owned.",
                file=sys.stderr,
            )
            return 1
    except Exception as error:
        print(
            f"Recovery stopped; original work remains blocked: {error}", file=sys.stderr
        )
        return 1
    print(
        "Saved receipts reconciled. The existing runtime can resume the original run."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
