"""Immutable clinic feedback events; optional notification is a separate receipt."""

import json
from sqlalchemy import select, func
from . import storage
from .clinic_catalogue import _write, _bump, _now
from .clinic_catalogue_reads import _patient, _envelope, _json
from .clinic_models import ClinicArtifact, CatalogueConflict, CatalogueNotFound
from .clinic_records import ClinicFeedback
from .clinic_intake import require_key


def _event(row):
    material = json.loads(row.material_json)
    return dict(
        eventId=row.id,
        patientId=material["patientId"],
        fileId=row.artifact_id,
        version=material["version"],
        action=row.action,
        notes=row.notes,
        submittedBy=row.author,
        submittedAt=row.created_at,
        notification=json.loads(row.notification_json)
        if row.notification_json
        else None,
    )


def current_feedback(session, file_id):
    events = list(
        session.scalars(
            select(ClinicFeedback)
            .where(ClinicFeedback.artifact_id == file_id)
            .order_by(ClinicFeedback.sequence)
        )
    )
    if not events:
        return None
    approval = next(
        (e for e in reversed(events) if e.action in ("approve", "reject")), None
    )
    last = approval or events[-1]
    return dict(**_event(last), latestEvent=_event(events[-1]))


def record_feedback(
    *,
    key,
    patient_id,
    file_id,
    version,
    action,
    notes="",
    actor=None,
    created_at=None,
    principal=None,
):
    require_key(key)
    require_key(file_id)
    if principal not in (None, "workbench", "thrylen-service"):
        raise ValueError("Invalid trusted principal")
    if (
        action not in ("approve", "reject", "notes", "archive", "unarchive")
        or not isinstance(notes, str)
        or len(notes) > 100000
        or (action == "reject" and not notes.strip())
    ):
        raise ValueError("Valid feedback and rejection notes required")
    if type(version) is not int or version < 1:
        raise ValueError("Exact file version required")
    if created_at is not None and (type(created_at) is not int or created_at < 0):
        raise ValueError("Invalid feedback timestamp")
    material = _json(
        dict(
            patientId=patient_id,
            fileId=file_id,
            version=version,
            action=action,
            notes=notes,
            actor=actor,
            createdAt=created_at,
        )
    )
    with _write() as s:
        prior = s.get(ClinicFeedback, key)
        if prior:
            if prior.material_json != material:
                raise CatalogueConflict("Feedback key binds different material")
            return _envelope(s, feedback=_event(prior))
        patient = _patient(s, patient_id)
        file = s.get(ClinicArtifact, file_id)
        if not file or file.patient_uuid != patient.id:
            raise CatalogueNotFound("Exact patient file not found")
        if file.version != version:
            raise CatalogueConflict("File version does not match")
        seq = (s.scalar(select(func.max(ClinicFeedback.sequence))) or 0) + 1
        row = ClinicFeedback(
            id=key,
            material_json=material,
            artifact_id=file.id,
            action=action,
            notes=notes,
            author=actor,
            principal=principal,
            created_at=created_at if created_at is not None else _now(),
            sequence=seq,
        )
        s.add(row)
        if action in ("archive", "unarchive"):
            file.archived = action == "archive"
        if action == "approve":
            for older in s.scalars(
                select(ClinicArtifact).where(
                    ClinicArtifact.patient_uuid == patient.id,
                    ClinicArtifact.logical_family == file.logical_family,
                    ClinicArtifact.version < version,
                )
            ):
                feedback = current_feedback(s, older.id)
                if feedback and feedback["action"] == "reject":
                    older.archived = True
        _bump(s, patient.id)
        s.flush()
        return _envelope(s, feedback=_event(row))


def feedback_history(file_id):
    with storage.session_scope() as s:
        return [
            _event(row)
            for row in s.scalars(
                select(ClinicFeedback)
                .where(ClinicFeedback.artifact_id == file_id)
                .order_by(ClinicFeedback.sequence)
            )
        ]


def record_notification(event_id, *, status, detail=""):
    if status not in ("pending", "sent", "failed", "unknown") or not isinstance(
        detail, str
    ):
        raise ValueError("Invalid notification receipt")
    with _write() as s:
        row = s.get(ClinicFeedback, event_id)
        if not row:
            raise CatalogueNotFound("Feedback event not found")
        prior = json.loads(row.notification_json) if row.notification_json else None
        if prior and prior["status"] == "sent" and status != "sent":
            raise CatalogueConflict("Sent notification is terminal")
        value = dict(status=status, detail=detail)
        if prior != value:
            row.notification_json = _json(value)
            _bump(s, s.get(ClinicArtifact, row.artifact_id).patient_uuid)
