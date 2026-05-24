from __future__ import annotations


_OPENAI_REASONING_EFFORTS = {"minimal", "low", "medium", "high", "xhigh"}


def _openai_gpt_effort_tokens(model_id: str) -> set[str]:
    mid = (model_id or "").strip().lower().removeprefix("openai/")
    if not mid.startswith("gpt-5."):
        return set()
    return {token for token in mid.split("-") if token in _OPENAI_REASONING_EFFORTS}


def _default_openai_gpt_effort_tokens(model_id: str) -> set[str]:
    mid = (model_id or "").strip().lower().removeprefix("openai/")
    if mid == "gpt-5.5":
        return {"low"}
    return set()


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
    if lower == "gemini-3.1-pro-preview":
        add("gemini-3-pro-preview")
    if lower == "gemini-3.1-flash":
        add("gemini-3.1-flash-lite-preview")
        add("gemini-3-flash-preview")
    if lower == "google/gemini-3.1-flash":
        add("google/gemini-3.1-flash-lite-preview")
        add("gemini-3.1-flash-lite-preview")
        add("google/gemini-3-flash-preview")
        add("gemini-3-flash-preview")
    if lower == "gemini-3.1-flash-lite-preview":
        add("gemini-3.1-flash")
        add("gemini-3-flash-preview")
    if lower == "google/gemini-3.1-flash-lite-preview":
        add("google/gemini-3.1-flash")
        add("gemini-3.1-flash")
        add("google/gemini-3-flash-preview")
        add("gemini-3-flash-preview")
    if lower == "google/gemini-3.1-pro-preview":
        add("google/gemini-3-pro-preview")
        add("gemini-3.1-pro-preview")
        add("gemini-3-pro-preview")
    if lower == "gemini-3-pro-preview":
        add("gemini-3.1-pro-preview")
    if lower == "google/gemini-3-pro-preview":
        add("google/gemini-3.1-pro-preview")
        add("gemini-3-pro-preview")
        add("gemini-3.1-pro-preview")

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
    preferred_efforts = set().union(
        *[_openai_gpt_effort_tokens(candidate) for candidate in aliases]
    )
    default_efforts: set[str] = set()
    if not preferred_efforts:
        default_efforts = set().union(
            *[_default_openai_gpt_effort_tokens(candidate) for candidate in aliases]
        )
    allowed_efforts = preferred_efforts or default_efforts
    for mid in discovered:
        mid_lower = mid.lower()
        mid_efforts = _openai_gpt_effort_tokens(mid)
        if (
            mid_efforts
            and allowed_efforts
            and not mid_efforts.issubset(allowed_efforts)
        ):
            continue
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
