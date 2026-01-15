from __future__ import annotations

import json
from typing import Any


def _strip_to_json(text: str) -> str:
    s = text.strip()
    if not s:
        return s
    if s[0] in "{[" and s[-1] in "}]":
        return s
    # Try to extract the outermost JSON-looking object/array.
    starts = [s.find("{"), s.find("[")]
    first = min((i for i in starts if i != -1), default=-1)
    if first == -1:
        return s
    last_curly = s.rfind("}")
    last_square = s.rfind("]")
    last = max(last_curly, last_square)
    if last <= first:
        return s
    return s[first : last + 1].strip()


def _json_loads_loose(text: str) -> Any:
    return json.loads(_strip_to_json(text))


def _loads_json_list(text: str) -> list[str]:
    try:
        data = json.loads(text)
    except Exception:
        return []
    if not isinstance(data, list):
        return []
    return [x for x in data if isinstance(x, str)]

