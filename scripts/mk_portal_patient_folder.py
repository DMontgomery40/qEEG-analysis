from __future__ import annotations

import argparse
import re
from pathlib import Path


PATIENT_ID_RE = re.compile(r"^(?P<mm>\d{2})-(?P<dd>\d{2})-(?P<yyyy>\d{4})-(?P<n>\d+)$")


def normalize_patient_id(value: str) -> str | None:
    raw = (value or "").strip()
    m = PATIENT_ID_RE.match(raw)
    if not m:
        return None
    mm = int(m.group("mm"))
    dd = int(m.group("dd"))
    yyyy = int(m.group("yyyy"))
    n = int(m.group("n"))
    if mm < 1 or mm > 12:
        return None
    if dd < 1 or dd > 31:
        return None
    if yyyy < 1900 or yyyy > 2100:
        return None
    if n < 0 or n > 999:
        return None
    return f"{mm:02d}-{dd:02d}-{yyyy:04d}-{n}"


def normalize_birthdate(value: str) -> str | None:
    raw = (value or "").strip()
    m = re.match(r"^(?P<mm>\d{2})-(?P<dd>\d{2})-(?P<yyyy>\d{4})$", raw)
    if not m:
        return None
    mm = int(m.group("mm"))
    dd = int(m.group("dd"))
    yyyy = int(m.group("yyyy"))
    if mm < 1 or mm > 12:
        return None
    if dd < 1 or dd > 31:
        return None
    if yyyy < 1900 or yyyy > 2100:
        return None
    return f"{mm:02d}-{dd:02d}-{yyyy:04d}"


def main() -> int:
    parser = argparse.ArgumentParser(description="Create a local portal patient folder under data/portal_patients/")
    parser.add_argument(
        "patient_id_or_birthdate",
        help="Either MM-DD-YYYY-N (full patient id) or MM-DD-YYYY (birthdate).",
    )
    parser.add_argument(
        "index",
        nargs="?",
        help="If providing MM-DD-YYYY, provide the duplicate index N (default: 0).",
    )
    args = parser.parse_args()

    patient_id = normalize_patient_id(args.patient_id_or_birthdate)
    if patient_id is None:
        birthdate = normalize_birthdate(args.patient_id_or_birthdate)
        if birthdate is None:
            raise SystemExit("Invalid input. Use MM-DD-YYYY-N or MM-DD-YYYY.")
        idx = int(args.index) if args.index is not None else 0
        patient_id = f"{birthdate}-{idx}"
        if normalize_patient_id(patient_id) is None:
            raise SystemExit("Invalid index. N must be a non-negative integer (0..999).")

    root = Path(__file__).resolve().parents[1]
    out_dir = root / "data" / "portal_patients" / patient_id
    out_dir.mkdir(parents=True, exist_ok=True)
    print(str(out_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

