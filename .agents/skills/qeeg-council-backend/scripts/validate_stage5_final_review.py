import json
import sys
from typing import Any

def _is_str_list(x: Any) -> bool:
    return isinstance(x, list) and all(isinstance(i, str) for i in x)

def main() -> int:
    if len(sys.argv) != 2:
        print("Usage: python validate_stage5_final_review.py <file.json>", file=sys.stderr)
        return 2

    path = sys.argv[1]
    try:
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
    except Exception as e:
        print(f"ERROR: could not read JSON: {e}", file=sys.stderr)
        return 3

    if not isinstance(obj, dict):
        print("ERROR: root must be an object", file=sys.stderr)
        return 4

    vote = obj.get("vote")
    req = obj.get("required_changes")
    opt = obj.get("optional_changes")
    score = obj.get("quality_score_1to10")

    if vote not in ("APPROVE", "REVISE"):
        print("ERROR: vote must be APPROVE or REVISE", file=sys.stderr)
        return 5

    if not _is_str_list(req):
        print("ERROR: required_changes must be an array of strings", file=sys.stderr)
        return 6

    if not _is_str_list(opt):
        print("ERROR: optional_changes must be an array of strings", file=sys.stderr)
        return 7

    if not isinstance(score, int) or score < 1 or score > 10:
        print("ERROR: quality_score_1to10 must be an integer 1-10", file=sys.stderr)
        return 8

    print("OK: Stage 5 final review JSON shape looks valid")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
