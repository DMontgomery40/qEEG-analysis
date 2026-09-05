"""The canonical clinic patient ID: ``XX_MM-DD-YYYY[_N]``.

Two initials, the date of birth, and a collision ordinal — ``ZZ_01-01-1900``,
``ZZ_01-01-1900_2``, ``ZZ_01-01-1900_10``. Ordinal 1 is the unsuffixed form, so
``_1`` never exists.

This is the one identifier clinic staff, folders, filenames, and sync keys use.
The engine's SQLite UUID stays behind it as an invisible relational key.

Every ID this module hands out is written to ``patient_id_reservations`` at
allocation and is never recomputed, compacted, or reused — not after a patient
is deleted, and not after a relabel frees it.

**Transaction contract.** ``allocate_canonical_patient_id`` and
``reserve_canonical_patient_id`` COMMIT the session they are given, because a
reservation has to be durable the moment it is issued. They never roll the
caller's session back: each reservation insert runs inside its own SAVEPOINT, so
losing a race to a simultaneous writer undoes only that insert and leaves
pending caller work intact. Call these before staging other writes, or expect
those writes to be committed along with the reservation.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session


CANONICAL_PATIENT_ID_RE = re.compile(
    r"^[A-Z]{2}_\d{2}-\d{2}-\d{4}(?:_(?:[2-9]|[1-9]\d+))?$"
)

_PARTS_RE = re.compile(
    r"^(?P<first>[A-Z])(?P<last>[A-Z])_"
    r"(?P<birthdate>\d{2}-\d{2}-\d{4})"
    r"(?:_(?P<ordinal>\d+))?$"
)

_LOOSE_BIRTHDATE_RE = re.compile(r"^(\d{1,2})-(\d{1,2})-(\d{4})$")

# Allocation walks ordinals upward. Real collisions are single digits; the cap
# only stops a broken database from spinning the loop forever.
MAX_COLLISION_ORDINAL = 999

# Matches the bound the legacy portal normalizer already applies. Reservations
# are never deleted, so a mistyped year would otherwise retire an ID forever.
BIRTH_YEAR_MIN = 1900
BIRTH_YEAR_MAX = 2100


class PatientIdentityError(ValueError):
    """Identity the operator has to correct or supply. Never guess past it."""


@dataclass(frozen=True)
class CanonicalPatientId:
    value: str
    first_initial: str
    last_initial: str
    birthdate: str
    ordinal: int


def _is_plausible_birthdate(month: int, day: int, year: int) -> bool:
    """A real calendar date inside the range a living patient can be born in."""
    if not BIRTH_YEAR_MIN <= year <= BIRTH_YEAR_MAX:
        return False
    try:
        datetime(year, month, day)
    except ValueError:
        return False
    return True


def normalize_birthdate(value: Any) -> str:
    """Return a date of birth as zero-padded ``MM-DD-YYYY``."""
    raw = str(value or "").strip()
    match = _LOOSE_BIRTHDATE_RE.match(raw)
    if match is None:
        raise PatientIdentityError(
            f"{raw or 'A blank value'} is not a date of birth written as MM-DD-YYYY."
        )
    month, day, year = (int(part) for part in match.groups())
    if not _is_plausible_birthdate(month, day, year):
        raise PatientIdentityError(
            f"{raw} is not a real date of birth between "
            f"{BIRTH_YEAR_MIN} and {BIRTH_YEAR_MAX}."
        )
    return f"{month:02d}-{day:02d}-{year:04d}"


def require_initial(value: Any, *, field: str = "patient") -> str:
    """Accept an initial the caller already knows is right."""
    raw = str(value or "").strip()
    if len(raw) != 1 or not raw.isascii() or not raw.upper().isalpha():
        raise PatientIdentityError(
            f"The {field} initial has to be a single A-Z letter. Got {value!r}."
        )
    return raw.upper()


def derive_initial(name: Any, *, field: str = "patient") -> str:
    """Derive an A-Z initial from a name, or say it cannot be done.

    Reads only the first character, so accents come through (``Peña`` gives
    ``P``) while a letter with no A-Z equivalent (``Ørsted``, ``李``) raises
    instead of being turned into whatever letter happens to come next.
    """
    raw = str(name or "").strip()
    if not raw:
        raise PatientIdentityError(
            f"No {field} name is on file, so its initial has to be supplied."
        )

    decomposed = unicodedata.normalize("NFKD", raw[0])
    candidate = (
        "".join(ch for ch in decomposed if not unicodedata.combining(ch))
        .strip()[:1]
        .upper()
    )

    if not candidate or not candidate.isascii() or not candidate.isalpha():
        raise PatientIdentityError(
            f"{raw!r} does not start with a letter that maps to A-Z. "
            f"Supply the {field} initial to use."
        )
    return candidate


def parse_canonical_patient_id(value: Any) -> CanonicalPatientId | None:
    """Return the parts of a canonical ID, or None if this is not one.

    Legacy ``MM-DD-YYYY-N`` storage keys and ``_1`` suffixes are not canonical.
    """
    if not isinstance(value, str) or not CANONICAL_PATIENT_ID_RE.match(value):
        return None

    match = _PARTS_RE.match(value)
    if match is None:
        return None

    birthdate = match.group("birthdate")
    month, day, year = (int(part) for part in birthdate.split("-"))
    if not _is_plausible_birthdate(month, day, year):
        return None

    raw_ordinal = match.group("ordinal")
    return CanonicalPatientId(
        value=value,
        first_initial=match.group("first"),
        last_initial=match.group("last"),
        birthdate=birthdate,
        ordinal=int(raw_ordinal) if raw_ordinal else 1,
    )


def canonical_patient_id(
    first_initial: str,
    last_initial: str,
    birthdate: str,
    ordinal: int = 1,
) -> str:
    """Build the canonical ID. Ordinal 1 is unsuffixed; 2 and up append ``_N``."""
    first = require_initial(first_initial, field="first")
    last = require_initial(last_initial, field="last")
    dob = normalize_birthdate(birthdate)

    if isinstance(ordinal, bool) or not isinstance(ordinal, int) or ordinal < 1:
        raise PatientIdentityError(f"A collision ordinal starts at 1. Got {ordinal!r}.")

    suffix = "" if ordinal == 1 else f"_{ordinal}"
    return f"{first}{last}_{dob}{suffix}"


def _canonical_id_is_taken(
    session: Session, candidate: str, *, exclude_patient_uuid: str | None = None
) -> bool:
    """Check both the patients currently wearing IDs and every retired one."""
    from .storage import Patient, PatientIdReservation

    worn_by = session.scalars(
        select(Patient.id).where(func.lower(Patient.label) == candidate.lower())
    ).all()
    if any(uuid != exclude_patient_uuid for uuid in worn_by):
        return True

    return session.get(PatientIdReservation, candidate) is not None


def _claim_reservation(
    session: Session, parsed: CanonicalPatientId, *, commit: bool = True
) -> bool:
    """Write the reservation row, reporting whether this caller won it.

    The reservation primary key — not the scan above it — is what stops two
    simultaneous allocations from issuing the same ID. The insert takes SQLite's
    write lock, and the writer that loses gets an IntegrityError and moves on.

    COMMITS the caller's session on success, so the reservation is durable the
    moment it is issued. The insert runs inside a SAVEPOINT, so a lost race
    undoes only the insert: anything the caller had pending survives.
    """
    from .storage import PatientIdReservation

    try:
        with session.begin_nested():
            session.add(
                PatientIdReservation(
                    patient_id=parsed.value,
                    first_initial=parsed.first_initial,
                    last_initial=parsed.last_initial,
                    birthdate=parsed.birthdate,
                    ordinal=parsed.ordinal,
                )
            )
    except IntegrityError:
        return False

    if commit:
        session.commit()
    return True


def reserve_canonical_patient_id(
    session: Session, value: Any, *, commit: bool = True
) -> CanonicalPatientId | None:
    """Record an already-formed canonical ID so it is never issued again.

    Returns None when the value is not a canonical ID, which is how pre-cutover
    labels pass through untouched.

    Commits by default. Internal commit=False joins an existing explicit SQLite
    writer transaction so an intake binding can commit with this reservation.
    """
    from .storage import PatientIdReservation

    parsed = parse_canonical_patient_id(value)
    if parsed is None:
        return None
    if session.get(PatientIdReservation, parsed.value) is None:
        _claim_reservation(session, parsed, commit=commit)
    return parsed


def allocate_canonical_patient_id(
    session: Session,
    *,
    first_initial: str,
    last_initial: str,
    birthdate: str,
    exclude_patient_uuid: str | None = None,
    commit: bool = True,
) -> str:
    """Issue this patient's canonical ID and reserve it permanently.

    Pass ``exclude_patient_uuid`` when re-confirming an existing patient's
    identity: a patient that already wears the ID keeps it instead of being
    bumped to the next ordinal.

    Commits by default and never rolls the caller back. Internal commit=False
    joins the existing explicit writer transaction; the ID must stay internal
    until the caller commits its reservation and durable intake binding together.
    """
    from .storage import Patient

    first = require_initial(first_initial, field="first")
    last = require_initial(last_initial, field="last")
    dob = normalize_birthdate(birthdate)

    already_held: str | None = None
    if exclude_patient_uuid:
        patient = session.get(Patient, exclude_patient_uuid)
        if patient is not None:
            already_held = (patient.label or "").strip().lower()

    for ordinal in range(1, MAX_COLLISION_ORDINAL + 1):
        candidate = canonical_patient_id(first, last, dob, ordinal=ordinal)

        if already_held is not None and already_held == candidate.lower():
            # A label can reach the database without ever passing through
            # allocation: pre-cutover bulk upload minted one from the filename
            # stem, and those rows are still here. Reserve it here too, or the ID
            # goes unreserved and a later patient with the same initials and
            # birthdate could be issued this person's ID.
            reserve_canonical_patient_id(session, candidate, commit=commit)
            return candidate

        if _canonical_id_is_taken(
            session, candidate, exclude_patient_uuid=exclude_patient_uuid
        ):
            continue

        parsed = parse_canonical_patient_id(candidate)
        if parsed is not None and _claim_reservation(session, parsed, commit=commit):
            return candidate

    raise PatientIdentityError(
        f"Every collision ordinal up to {MAX_COLLISION_ORDINAL} is already taken "
        f"for {first}{last}_{dob}."
    )
