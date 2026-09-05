"""Bounded global drawer projection from the existing catalogue and serializers."""

from collections import defaultdict
from sqlalchemy import select, text
from . import storage
from .clinic_catalogue_reads import _artifact_json, _envelope
from .clinic_models import ClinicArtifact, ClinicLocation, CatalogueConflict
from .clinic_records import ClinicFeedback
from .patient_identity import parse_canonical_patient_id


def recent_files(*, kind, content_type=None, limit=30):
    if kind not in ("video", "patient-summary"):
        raise ValueError("Expected video or patient-summary kind")
    if content_type is not None and (
        not isinstance(content_type, str)
        or not content_type
        or len(content_type) > 128
        or any(ord(c) < 32 for c in content_type)
    ):
        raise ValueError("Invalid contentType")
    if isinstance(limit, bool):
        raise ValueError("Invalid recent file limit")
    try:
        count = int(limit)
        if str(count) != str(limit) or not 1 <= count <= 120:
            raise ValueError()
    except (ValueError, TypeError, OverflowError):
        raise ValueError("Expected limit from 1 to 120") from None
    with storage.session_scope() as session:
        session.execute(text("BEGIN"))
        # Same canonical/ambiguous chart policy as the common roster. This is
        # one identity census, never a per-chart file traversal.
        patients = {
            p.id: p
            for p in session.scalars(select(storage.Patient))
            if parse_canonical_patient_id(p.label)
        }
        if len({p.label for p in patients.values()}) != len(patients):
            raise CatalogueConflict("Patient identity is ambiguous")
        query = select(ClinicArtifact).where(
            ClinicArtifact.patient_uuid.in_(patients),
            ClinicArtifact.document_kind == kind,
        )
        if content_type is not None:
            query = query.where(ClinicArtifact.content_type == content_type)
        artifacts = list(
            session.scalars(
                query.order_by(
                    ClinicArtifact.generated_at.desc(),
                    ClinicArtifact.session_date.desc(),
                    ClinicArtifact.version.desc(),
                    ClinicArtifact.id.desc(),
                ).limit(count + 1)
            )
        )
        selected = artifacts[:count]
        ids = [a.id for a in selected]
        locations, events = defaultdict(list), defaultdict(list)
        if ids:
            for row in session.scalars(
                select(ClinicLocation)
                .where(ClinicLocation.artifact_id.in_(ids))
                .order_by(ClinicLocation.kind, ClinicLocation.key)
            ):
                locations[row.artifact_id].append(row)
            for row in session.scalars(
                select(ClinicFeedback)
                .where(ClinicFeedback.artifact_id.in_(ids))
                .order_by(ClinicFeedback.sequence)
            ):
                events[row.artifact_id].append(row)
        return _envelope(
            session,
            files=[
                _artifact_json(
                    session,
                    a,
                    patients[a.patient_uuid],
                    locations=locations[a.id],
                    feedback_events=events[a.id],
                )
                for a in selected
            ],
            limit=count,
            returnedFiles=len(selected),
            truncated=len(artifacts) > count,
        )
