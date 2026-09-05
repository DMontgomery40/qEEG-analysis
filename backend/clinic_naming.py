"""Current clinic presentation rules; original names remain provenance."""

from __future__ import annotations

import re
import unicodedata

from .patient_identity import parse_canonical_patient_id

POLICY_REVISION = "clinic-rules-v1"
POLICY = {
    "patientIdRulesVersion": "canonical-patient-id-v1",
    "tts": {
        "provider": "OpenRouter",
        "model": "google/gemini-3.1-flash-tts-preview",
        "voice": "Charon",
        "speed": 1.0,
    },
}
_ID = re.compile(r"[A-Z]{2}_\d{2}-\d{2}-\d{4}(?:_(?:[2-9]|[1-9]\d+))?")


def canonical_filename(name: str, patient_id: str) -> str:
    """Port of the accepted hub naming matrix, using the engine ID parser."""
    if not parse_canonical_patient_id(patient_id):
        raise ValueError("A current canonical patient ID is required")
    raw = unicodedata.normalize("NFC", str(name or "patient-file"))
    raw = re.sub(r'[\x00-\x1f\x7f"`]', "", raw)
    raw = re.sub(r"[<>:|?*]", "_", raw)
    raw = re.sub(
        r"(^|[/\\])[_\s-]*\d{1,2}-\d{1,2}-\d{4}(?:-\d+)?(?=__|[_. -]|$)(?:__)?(?=[^/\\]*$)",
        r"\1",
        raw,
    )
    raw = _ID.sub("", raw)
    raw = re.sub(r"[/\\]+", "__", raw).lstrip("_ -\t\r\n")
    match = re.search(r"(\.[A-Za-z0-9]{1,8})$", raw)
    extension = match[1] if match else ""
    stem = (raw[: -len(extension)] if extension else raw).strip(
        "_ .-\t\r\n"
    ) or "patient-file"
    budget = 240 - len(f"{patient_id}__{extension}".encode("utf-8"))
    if budget <= 0:
        raise ValueError(
            "Patient identity and extension exceed the filename byte limit"
        )
    while len(stem.encode("utf-8")) > budget:
        if len(stem) == 1:
            raise ValueError("Filename description cannot fit the identity byte budget")
        index = max(0, len(stem) - 65)
        stem = stem[:index] + stem[index + 1 :]
    return f"{patient_id}__{stem}{extension}"
