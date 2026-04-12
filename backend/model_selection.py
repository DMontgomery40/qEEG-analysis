from __future__ import annotations


def _alias_candidates(preferred: str) -> list[str]:
    pref = (preferred or "").strip()
    if not pref:
        return []

    seen: set[str] = set()
    out: list[str] = []

    def add(value: str) -> None:
        candidate = (value or "").strip()
        key = candidate.lower()
        if not candidate or key in seen:
            return
        seen.add(key)
        out.append(candidate)

    add(pref)

    if "." in pref:
        add(pref.replace(".", "-"))

    lower = pref.lower()
    if lower == "gpt-5.4":
        add("gpt-5")
    if lower == "openai/gpt-5.4":
        add("openai/gpt-5")
        add("gpt-5.4")
        add("gpt-5")

    return out


def resolve_model_preference(preferred: str, discovered: list[str]) -> str | None:
    pref = (preferred or "").strip()
    if not pref:
        return None

    aliases = _alias_candidates(pref)

    for candidate in aliases:
        if candidate in discovered:
            return candidate

    for candidate in aliases:
        candidate_lower = candidate.lower()
        for mid in discovered:
            if mid.lower() == candidate_lower:
                return mid

    matches: list[str] = []
    alias_lowers = [candidate.lower() for candidate in aliases]
    for mid in discovered:
        mid_lower = mid.lower()
        if any(alias_lower in mid_lower for alias_lower in alias_lowers):
            matches.append(mid)

    if not matches:
        return None

    def rank(mid: str) -> tuple[int, int, str]:
        lower = mid.lower()
        preview_penalty = 1 if "preview" in lower else 0
        date_bonus = 0
        parts = lower.split("-")
        if parts and parts[-1].isdigit() and len(parts[-1]) >= 6:
            date_bonus = -int(parts[-1][-6:])
        return (preview_penalty, date_bonus, mid)

    return sorted(matches, key=rank)[0]
