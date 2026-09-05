"""Deciding which chart an incoming report belongs to.

Reports arrive from the clinic operator through the API and from the hub through
the portal pipeline worker. Both have to answer the same question — is this
somebody already on file? — and they answer it with the code here, so a report
cannot be filed one way in one path and another way in the other.

Errors raised here are domain errors. The API maps them to status codes at its
own boundary; the worker parks the job instead.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from sqlalchemy import select

from .patient_identity import (
    PatientIdentityError,
    derive_initial,
    normalize_birthdate,
    require_initial,
)


@dataclass(frozen=True)
class IdentityInput:
    """What the operator read off the report, plus how they answered a conflict."""

    first_name: str | None = None
    last_name: str | None = None
    birthdate: str | None = None
    first_initial: str | None = None
    last_initial: str | None = None
    attach_to: str | None = None
    force_new: bool = False


class IdentityNameConflict(Exception):
    """A chart with these initials and birthday carries a different name.

    In a clinic this size that is nearly always one person written down two
    ways — Dave and David, with or without a middle initial — so allocating the
    next ordinal would quietly split their history in half. It is occasionally
    two real people, which is what the ordinal is for. Neither guess is safe, so
    the operator is asked.
    """

    def __init__(self, payload: dict[str, Any]) -> None:
        super().__init__("identity_name_mismatch")
        self.payload = payload


class IdentityMatchesAmbiguous(PatientIdentityError):
    """Original matcher ambiguity with read-only candidate evidence."""

    def __init__(self, message, candidates):
        super().__init__(message)
        self.candidates = candidates


def stored_full_name(patient: Any) -> str:
    return " ".join(
        part for part in ((patient.first_name or ""), (patient.last_name or "")) if part
    ).strip()


def identity_key(identity: IdentityInput) -> tuple[str, str, str]:
    """Derive initials and date of birth without allocating anything.

    Allocation reserves an id permanently, so the search for an existing
    patient has to happen before it — otherwise every re-upload retires a
    collision ordinal nobody is wearing.
    """
    first = (
        require_initial(identity.first_initial, field="first")
        if (identity.first_initial or "").strip()
        else derive_initial(identity.first_name, field="first")
    )
    last = (
        require_initial(identity.last_initial, field="last")
        if (identity.last_initial or "").strip()
        else derive_initial(identity.last_name, field="last")
    )
    return first, last, normalize_birthdate(identity.birthdate)


def find_patient_by_identity(
    session: Any, identity: IdentityInput, *, target_patient: Any | None = None
) -> tuple[Any | None, bool]:
    """Resolve which chart this identity belongs to.

    Returns the patient to file under (or None to allocate a new one) and
    whether that patient's stored name must be left alone.

    Same initials, same date of birth, and no name on file that contradicts the
    one given is the same person — a second report must land on that chart
    rather than allocating a `_2` and splitting the patient into two families.
    A chart created from a bare canonical label carries initials and a date of
    birth but no names, so an absent stored name matches anything and gets
    filled in.

    A name that differs raises `IdentityNameConflict` instead of choosing, and
    `attach_to` / `force_new` are how the operator answers it.

    Raises `PatientIdentityError` when two charts already answer to one
    identity, because picking between them is the operator's call too.
    """
    from .storage import Patient, find_patients_by_label

    attach_to = (identity.attach_to or "").strip()
    if attach_to:
        patient = next(iter(find_patients_by_label(session, attach_to)), None)
        if patient is None:
            raise PatientIdentityError(f"{attach_to} is not a patient on file.")
        # The operator said this is the same person, not that the name on file
        # is wrong. Correcting a name is what the patient update path is for.
        return patient, True

    if identity.force_new:
        return None, False

    first_initial, last_initial, birthdate = identity_key(identity)
    first_name = (identity.first_name or "").strip().lower()
    last_name = (identity.last_name or "").strip().lower()

    def name_fits(stored: str | None, given: str) -> bool:
        on_file = (stored or "").strip().lower()
        return not on_file or on_file == given

    if target_patient is None:
        candidates = session.scalars(
            select(Patient).where(
                Patient.first_initial == first_initial,
                Patient.last_initial == last_initial,
                Patient.birthdate == birthdate,
            )
        ).all()
    else:
        # Explicit shared-chart intake uses this same name comparison. Historical
        # canonical rows may have no normalized columns; their issued ID supplies
        # only those missing identity components, never a guessed name.
        from .patient_identity import parse_canonical_patient_id

        parsed = parse_canonical_patient_id(target_patient.label)
        stored_key = (
            target_patient.first_initial or (parsed.first_initial if parsed else None),
            target_patient.last_initial or (parsed.last_initial if parsed else None),
            target_patient.birthdate or (parsed.birthdate if parsed else None),
        )
        candidates = (
            [target_patient]
            if stored_key == (first_initial, last_initial, birthdate)
            else []
        )
    matches = [
        patient
        for patient in candidates
        if name_fits(patient.first_name, first_name)
        and name_fits(patient.last_name, last_name)
    ]
    if len(matches) > 1:
        raise IdentityMatchesAmbiguous(
            "This name and date of birth already match more than one patient: "
            + ", ".join(sorted(patient.label for patient in matches))
            + ". Say which one this report belongs to.",
            matches,
        )
    if matches:
        return matches[0], False

    differing = [patient for patient in candidates if patient not in matches]
    if differing:
        raise IdentityNameConflict(
            {
                "conflict": "identity_name_mismatch",
                "incoming_name": " ".join(
                    part
                    for part in (
                        (identity.first_name or ""),
                        (identity.last_name or ""),
                    )
                    if part
                ).strip(),
                "candidates": [
                    {
                        "patient_id": patient.label,
                        "name": stored_full_name(patient),
                    }
                    for patient in differing
                ],
                "detail": (
                    "Someone with these initials and this date of birth is "
                    "already on file under a different name. Say which patient "
                    "this is with attach_to, or force_new if they are two "
                    "different people."
                ),
            }
        )
    return None, False
