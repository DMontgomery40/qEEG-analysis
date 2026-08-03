from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from backend.portal_files import normalize_portal_patient_id  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Create a local portal patient folder under data/portal_patients/"
    )
    parser.add_argument(
        "patient_id",
        help="The clinic patient id: XX_MM-DD-YYYY, or XX_MM-DD-YYYY_N after a collision.",
    )
    args = parser.parse_args()

    patient_id = normalize_portal_patient_id(args.patient_id)
    if patient_id is None:
        raise SystemExit(
            "That is not a clinic patient id. Use XX_MM-DD-YYYY, "
            "or XX_MM-DD-YYYY_N when two patients share initials and a birthday."
        )

    out_dir = _REPO_ROOT / "data" / "portal_patients" / patient_id
    out_dir.mkdir(parents=True, exist_ok=True)
    print(str(out_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
