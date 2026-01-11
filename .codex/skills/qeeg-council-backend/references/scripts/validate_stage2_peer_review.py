import json
import sys
from typing import Any, Dict, List

def _is_str_list(x: Any) -> bool:
    return isinstance(x, list) and all(isinstance(i, str) for i in x)

def main() -> int:
    if len(sys.argv) != 2:
        print("Usage: python validate_stage2_peer_review.py <file.json>", file=sys.stderr)
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

    reviews = obj.get("reviews")
    ranking = obj.get("overall_ranking")

    if not isinstance(reviews, dict):
        print("ERROR: reviews must be an object", file=sys.stderr)
        return 5

    if not isinstance(ranking, list) or not all(isinstance(x, str) for x in ranking) or len(ranking) < 1:
        print("ERROR: overall_ranking must be a non-empty array of strings", file=sys.stderr)
        return 6

    required_fields = ["strengths", "issues", "missing", "suggestions"]
    for label, review in reviews.items():
        if not isinstance(label, str):
            print("ERROR: review keys must be strings", file=sys.stderr)
            return 7
        if not isinstance(review, dict):
            print(f"ERROR: reviews[{label}] must be an object", file=sys.stderr)
            return 8
        for rf in required_fields:
            if rf not in review:
                print(f"ERROR: reviews[{label}] missing field {rf}", file=sys.stderr)
                return 9
            if not _is_str_list(review[rf]):
                print(f"ERROR: reviews[{label}].{rf} must be an array of strings", file=sys.stderr)
                return 10

    print("OK: Stage 2 peer review JSON shape looks valid")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
