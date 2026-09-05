"""Explicit chart updates with one durable mutation and original identity rules."""

import json
from . import storage
from .clinic_catalogue import _write
from .clinic_catalogue_reads import _patient, _patient_json, _envelope, _json
from .clinic_models import CatalogueConflict
from .clinic_records import ClinicMutation
from .clinic_intake import require_key
from .patient_identity import (
    allocate_canonical_patient_id,
    derive_initial,
    require_initial,
    normalize_birthdate,
    parse_canonical_patient_id,
)


def patch_patient(patient_id, *, key, changes, actor=None):
    require_key(key)
    allowed = {
        "firstName",
        "lastName",
        "firstInitial",
        "lastInitial",
        "birthdate",
        "notes",
    }
    if (
        not isinstance(changes, dict)
        or not changes
        or set(changes) - allowed
        or any(not isinstance(v, str) for v in changes.values())
    ):
        raise ValueError("Invalid patient update")
    material = _json(
        dict(kind="patient", patientId=patient_id, changes=changes, actor=actor)
    )
    with _write() as s:
        prior = s.get(ClinicMutation, key)
        if prior:
            if prior.material_json != material:
                raise CatalogueConflict("Patient update key changed")
            return json.loads(prior.result_json)
        p = _patient(s, patient_id)
        parsed = parse_canonical_patient_id(p.label)
        fields = dict(
            label=p.label,
            notes=changes.get("notes"),
            first_name=changes.get("firstName"),
            last_name=changes.get("lastName"),
        )
        if set(changes) - {"notes"}:
            first = (
                require_initial(changes["firstInitial"], field="first")
                if "firstInitial" in changes
                else derive_initial(changes["firstName"], field="first")
                if "firstName" in changes
                else (p.first_initial or parsed.first_initial)
            )
            last = (
                require_initial(changes["lastInitial"], field="last")
                if "lastInitial" in changes
                else derive_initial(changes["lastName"], field="last")
                if "lastName" in changes
                else (p.last_initial or parsed.last_initial)
            )
            dob = normalize_birthdate(
                changes.get("birthdate", p.birthdate or parsed.birthdate)
            )
            fields.update(
                label=allocate_canonical_patient_id(
                    s,
                    first_initial=first,
                    last_initial=last,
                    birthdate=dob,
                    exclude_patient_uuid=p.id,
                    commit=False,
                ),
                first_initial=first,
                last_initial=last,
                birthdate=dob,
            )
        p = storage.update_patient(s, p.id, **fields, commit=False)
        result = _envelope(s, patient=_patient_json(p))
        s.add(
            ClinicMutation(key=key, material_json=material, result_json=_json(result))
        )
        return result
