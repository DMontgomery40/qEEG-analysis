"""Ordered read-only correction previews through the original identity domain."""

from sqlalchemy import text
from . import storage
from .clinic_catalogue_reads import _patient, _patient_json, _envelope
from .clinic_models import CatalogueConflict, CatalogueNotFound
from .patient_identity import (
    derive_initial,
    require_initial,
    normalize_birthdate,
    parse_canonical_patient_id,
)
from .patient_intake import (
    IdentityInput,
    IdentityNameConflict,
    IdentityMatchesAmbiguous,
    find_patient_by_identity,
)

FIELDS = {
    "firstName": "first_name",
    "lastName": "last_name",
    "firstInitial": "first_initial",
    "lastInitial": "last_initial",
    "birthdate": "birthdate",
}


def normalize_preview_fields(record):
    proposed = {}
    for key, value in record.items():
        if not isinstance(value, str):
            raise ValueError("Identity fields must be text")
        if key == "patientId":
            if not value or len(value) > 256:
                raise ValueError("Invalid patientId")
            continue
        if key == "birthdate":
            value = normalize_birthdate(value)
        elif key.endswith("Initial"):
            value = require_initial(value, field=key[:-7].lower())
        else:
            value = value.strip()
            # PATCH derives missing initials from supplied names. Validate that
            # operation here without inventing an initial in proposed material.
            if key.replace("Name", "Initial") not in record:
                derive_initial(value, field=key[:-4].lower())
        proposed[key] = value
    return proposed


def _preview_row(session, record, row):
    result = dict(
        row=row,
        status="invalid",
        patientId=None,
        current=None,
        proposed={},
        candidates=[],
        reason=None,
    )
    try:
        proposed = normalize_preview_fields(record)
        result["proposed"] = proposed
        if "patientId" in record:
            patient = _patient(session, record["patientId"])
        else:
            patient, _ = find_patient_by_identity(
                session, IdentityInput(**{FIELDS[k]: v for k, v in proposed.items()})
            )
        if patient is None or not parse_canonical_patient_id(patient.label):
            result.update(
                status="not_found", reason="No current chart matches this identity"
            )
            return result
        # A matched identity must still satisfy the shared alias/current-label
        # ambiguity rules; selection is never an implicit chart merge.
        patient = _patient(session, patient.label)
        current = _patient_json(patient)
        fields = {**current["identity"], "birthdate": current["birthdate"]}
        changed = any(fields[k] != v for k, v in proposed.items())
        result.update(
            status="change" if changed else "unchanged",
            patientId=patient.label,
            current=current,
        )
    except IdentityMatchesAmbiguous as error:
        result.update(
            status="needs_operator_answer",
            reason=str(error),
            candidates=[
                _patient_json(p)
                for p in error.candidates
                if parse_canonical_patient_id(p.label)
            ],
        )
    except IdentityNameConflict as error:
        candidates = []
        for candidate in error.payload["candidates"]:
            try:
                candidates.append(
                    _patient_json(_patient(session, candidate["patient_id"]))
                )
            except (CatalogueConflict, CatalogueNotFound):
                pass
        result.update(
            status="needs_operator_answer",
            reason=error.payload["detail"],
            candidates=candidates,
        )
    except CatalogueConflict as error:
        result.update(status="needs_operator_answer", reason=str(error))
    except CatalogueNotFound as error:
        result.update(status="not_found", reason=str(error))
    except ValueError as error:
        result.update(status="invalid", reason=str(error))
    return result


def preview_identities(body):
    if not isinstance(body, dict) or set(body) != {"records"}:
        raise ValueError("Expected records")
    records = body["records"]
    if not isinstance(records, list) or not 1 <= len(records) <= 100:
        raise ValueError("Expected one to 100 records")
    if any(
        not isinstance(r, dict) or set(r) - set(FIELDS) - {"patientId"} for r in records
    ):
        raise ValueError("Unsupported identity fields")
    with storage.session_scope() as session:
        session.execute(text("BEGIN"))
        rows = [_preview_row(session, r, i) for i, r in enumerate(records, 1)]
        return _envelope(session, rows=rows)
