"""Database-owned clinic projections and exact local byte reads."""

from __future__ import annotations

import base64
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import tempfile

from sqlalchemy import Integer, case, cast, func, or_, select, text
from . import storage
from .clinic_models import (
    ClinicArtifact,
    ClinicCatalogState,
    ClinicLocation,
    ClinicPatientAlias,
    ClinicProjection,
    ClinicPatientCatalogState,
    CatalogueConflict,
    CatalogueNotFound,
    CatalogueUnavailable,
)
from .clinic_naming import POLICY_REVISION, canonical_filename
from .patient_identity import parse_canonical_patient_id


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def current_revision():
    with storage.session_scope() as session:
        state = session.get(ClinicCatalogState, 1)
        if state is None:
            raise CatalogueUnavailable("Catalogue has not been initialized")
        return state.revision


def _state(session, patient_uuid=None):
    state = session.get(ClinicCatalogState, 1)
    if state is None:
        raise CatalogueUnavailable("Catalogue has not been initialized")
    if patient_uuid is not None:
        if not state.import_complete:
            raise CatalogueUnavailable("Catalogue import is incomplete")
        if session.scalar(
            select(ClinicProjection.id)
            .where(
                ClinicProjection.patient_uuid == patient_uuid,
                ClinicProjection.artifact_id.is_(None),
            )
            .limit(1)
        ):
            raise CatalogueUnavailable("Patient source projection is incomplete")
    return state


def _envelope(session, **payload):
    return dict(
        ok=True,
        schemaVersion="clinic-v1",
        policyRevision=POLICY_REVISION,
        catalogRevision=_state(session).revision,
        **payload,
    )


def _patient(session, patient_id):
    if not isinstance(patient_id, str) or not patient_id or len(patient_id) > 256:
        raise ValueError("Invalid patientId")
    patients = list(
        session.scalars(
            select(storage.Patient).where(storage.Patient.label == patient_id)
        )
    )
    if len(patients) > 1:
        raise CatalogueConflict("Patient identity is ambiguous")
    patient = patients[0] if patients else None
    if patient is None:
        alias = session.get(ClinicPatientAlias, patient_id)
        if alias is not None and alias.ambiguous:
            raise CatalogueConflict("Historical patient alias is ambiguous")
        patient = session.get(storage.Patient, alias.patient_uuid) if alias else None
    if patient is None or not parse_canonical_patient_id(patient.label):
        raise CatalogueNotFound("Patient not found")
    if (
        session.scalar(
            select(func.count())
            .select_from(storage.Patient)
            .where(storage.Patient.label == patient.label)
        )
        != 1
    ):
        raise CatalogueConflict("Current patient identity remains ambiguous")
    return patient


def _millis(value):
    if value is None:
        return None
    return int(value.replace(tzinfo=timezone.utc).timestamp() * 1000)


def _patient_json(patient):
    parsed = parse_canonical_patient_id(patient.label)
    return dict(
        patientId=patient.label,
        birthdate=patient.birthdate or parsed.birthdate,
        index=parsed.ordinal,
        createdAt=_millis(patient.created_at),
        lastUpdatedAt=_millis(patient.updated_at),
        identity=dict(
            firstName=patient.first_name,
            lastName=patient.last_name,
            firstInitial=patient.first_initial or parsed.first_initial,
            lastInitial=patient.last_initial or parsed.last_initial,
        ),
        notes=patient.notes,
    )


def roster(patient_id=None):
    with storage.session_scope() as session:
        session.execute(text("BEGIN"))
        if patient_id is not None:
            return _envelope(
                session, patient=_patient_json(_patient(session, patient_id))
            )
        patients = [
            p
            for p in storage.list_patients(session)
            if parse_canonical_patient_id(p.label)
        ]
        if len({p.label for p in patients}) != len(patients):
            raise CatalogueConflict("Patient identity is ambiguous")
        return _envelope(session, patients=[_patient_json(p) for p in patients])


def _allowed_path(path):
    path = Path(path).resolve(strict=True)
    root = Path(storage.DATA_DIR).resolve(strict=True)
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError("Local source must be a file inside DATA_DIR")
    return path


def _fingerprint(path):
    stat = _allowed_path(path).stat()
    return _json(
        [stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns]
    )


def _location_verified(location):
    if not location.active or not location.verified:
        return False
    if location.kind == "local":
        try:
            return location.fingerprint == _fingerprint(location.key)
        except (OSError, ValueError):
            return False
    return True


def _document_kind():
    # Imported catalogue history may predate documentKind. The stored MIME type
    # still identifies video; an explicit classification always takes precedence.
    return case(
        (ClinicArtifact.document_kind.is_not(None), ClinicArtifact.document_kind),
        (func.lower(ClinicArtifact.content_type).like("video/%"), "video"),
        else_=None,
    )


def _generation_time():
    safe_provenance = case(
        (
            func.json_valid(ClinicArtifact.provenance_json),
            ClinicArtifact.provenance_json,
        ),
        else_="{}",
    )
    proof = (
        func.json_each(safe_provenance, "$.proofs")
        .table_valued("value", "type")
        .alias("proof")
    )
    proof_value = case((proof.c.type == "object", proof.c.value), else_="{}")
    receipt_stamp = func.json_extract(proof_value, "$.artifactRecord.ts")
    timezone_present = or_(
        receipt_stamp.op("GLOB")("*Z"),
        func.substr(receipt_stamp, -6).op("GLOB")("[+-][0-9][0-9]:[0-9][0-9]"),
    )
    receipt_time = (
        select(
            func.max(
                cast(
                    func.round(
                        (
                            func.julianday(
                                func.json_extract(proof_value, "$.artifactRecord.ts")
                            )
                            - 2440587.5
                        )
                        * 86400000
                    ),
                    Integer,
                )
            )
        )
        .where(
            timezone_present,
            func.json_extract(proof_value, "$.type")
            == "original-workbench-exact-artifact-receipt",
            func.json_extract(proof_value, "$.patientUuid")
            == ClinicArtifact.patient_uuid,
            func.json_extract(proof_value, "$.artifactRecord.path")
            == func.json_extract(safe_provenance, "$.originalLocalPath"),
        )
        .correlate(ClinicArtifact)
        .scalar_subquery()
    )
    return func.coalesce(ClinicArtifact.generated_at, receipt_time)


def _record_generation_time(artifact, provenance):
    if artifact.generated_at is not None:
        return artifact.generated_at
    times = []
    for proof in (
        provenance.get("proofs", [])
        if isinstance(provenance.get("proofs"), list)
        else []
    ):
        if not isinstance(proof, dict):
            continue
        record = proof.get("artifactRecord")
        if (
            proof.get("type") != "original-workbench-exact-artifact-receipt"
            or proof.get("patientUuid") != artifact.patient_uuid
            or not isinstance(record, dict)
            or not provenance.get("originalLocalPath")
            or record.get("path") != provenance["originalLocalPath"]
        ):
            continue
        try:
            if not re.search(r"(?:Z|[+-][0-9]{2}:[0-9]{2})$", record["ts"]):
                continue
            stamp = datetime.fromisoformat(record["ts"].replace("Z", "+00:00"))
            if stamp.tzinfo is not None:
                times.append(round(stamp.timestamp() * 1000))
        except (KeyError, TypeError, ValueError, AttributeError):
            continue
    return max(times) if times else None


def _working_file_ids(patient_uuids):
    # Group only when the same chart/role/session has a verified remote copy.
    # Local-only rows stay individually addressable in the list until publication;
    # no filesystem census, copied location, or source-identity mutation is needed.
    from .clinic_records import ClinicFeedback

    stored_kind = _document_kind()
    kind = case(
        (stored_kind.in_(("report", "source-report")), "source-report"),
        else_=stored_kind,
    )
    reviewed = (
        select(ClinicFeedback.id)
        .where(ClinicFeedback.artifact_id == ClinicArtifact.id)
        .exists()
    )
    separate = case(
        (kind.in_(("source-report", "report", "patient-summary", "video")), None),
        else_=ClinicArtifact.id,
    )
    remote = (
        select(ClinicLocation.id)
        .where(
            ClinicLocation.artifact_id == ClinicArtifact.id,
            ClinicLocation.kind == "netlify",
            ClinicLocation.active.is_(True),
            ClinicLocation.verified.is_(True),
        )
        .exists()
    )
    partition = (
        ClinicArtifact.patient_uuid,
        separate,
        kind,
        ClinicArtifact.sha256,
        ClinicArtifact.size,
        func.lower(ClinicArtifact.content_type),
        ClinicArtifact.session_date,
    )
    candidates = (
        select(
            ClinicArtifact.id,
            ClinicArtifact.patient_uuid,
            ClinicArtifact.sha256,
            ClinicArtifact.size,
            func.lower(ClinicArtifact.content_type).label("content_type"),
            ClinicArtifact.session_date,
            ClinicArtifact.uploaded_at,
            ClinicArtifact.version,
            reviewed.label("reviewed"),
            separate.label("separate"),
            kind.label("kind"),
            remote.label("remote"),
            _generation_time().label("generated"),
            func.max(cast(remote, Integer))
            .over(partition_by=partition)
            .label("published_group"),
        )
        .where(ClinicArtifact.patient_uuid.in_(patient_uuids))
        .subquery()
    )
    c = candidates.c
    ranked = select(
        c.id,
        c.reviewed,
        func.row_number()
        .over(
            partition_by=(
                c.patient_uuid,
                case((c.published_group == 0, c.id), else_=c.separate),
                c.kind,
                c.sha256,
                c.size,
                c.content_type,
                c.session_date,
            ),
            order_by=(
                c.remote.desc(),
                c.reviewed.desc(),
                c.generated.desc(),
                c.uploaded_at.desc(),
                c.version.desc(),
                c.id.desc(),
            ),
        )
        .label("position"),
    ).subquery()
    return select(ranked.c.id).where(or_(ranked.c.position == 1, ranked.c.reviewed))


def _artifact_json(session, artifact, patient, *, locations=None, feedback_events=None):
    from .clinic_feedback import current_feedback

    if (
        not re.fullmatch("[a-f0-9]{64}", artifact.sha256 or "")
        or type(artifact.size) is not int
        or artifact.size < 0
        or artifact.version < 1
    ):
        raise CatalogueUnavailable("Catalogue byte binding is malformed")
    try:
        provenance = json.loads(artifact.provenance_json)
        if not isinstance(provenance, dict):
            raise ValueError()
    except ValueError as error:
        raise CatalogueUnavailable("Catalogue provenance is malformed") from error
    locations = (
        locations
        if locations is not None
        else list(
            session.scalars(
                select(ClinicLocation)
                .where(ClinicLocation.artifact_id == artifact.id)
                .order_by(ClinicLocation.kind, ClinicLocation.key)
            )
        )
    )
    canonical = parse_canonical_patient_id(patient.label)
    visible = (
        canonical_filename(artifact.original_name, patient.label) if canonical else None
    )
    return dict(
        fileId=artifact.id,
        patientId=patient.label if canonical else None,
        fileKey=artifact.file_key,
        originalName=artifact.original_name,
        logicalName=canonical_filename(artifact.original_name, patient.label)
        if canonical
        else None,
        displayName=visible,
        downloadName=visible,
        version=artifact.version,
        size=artifact.size,
        sha256=artifact.sha256,
        hashVerified=any(_location_verified(location) for location in locations),
        contentType=artifact.content_type,
        uploadedAt=artifact.uploaded_at,
        uploadedBy=artifact.uploaded_by,
        generatedAt=_record_generation_time(artifact, provenance),
        documentKind=artifact.document_kind
        or ("video" if artifact.content_type.lower().startswith("video/") else None),
        sessionDate=artifact.session_date,
        feedback=current_feedback(session, artifact.id, events=feedback_events),
        archived=artifact.archived,
        provenance=provenance,
        locations=[
            dict(
                kind=location.kind,
                key=location.key if location.kind != "local" else None,
                active=location.active,
                verified=_location_verified(location),
                verifiedAt=location.verified_at,
            )
            for location in locations
        ],
    )


def _historical_binding(session, patient_id, file_key, file_id):
    """A recorded exact file location can disambiguate an old shared label.

    The original artifact's Patient row remains the binding; a label alone is
    insufficient, and a different chart's file cannot borrow this alias.
    """
    query = (
        select(ClinicArtifact, ClinicLocation)
        .join(ClinicLocation, ClinicLocation.artifact_id == ClinicArtifact.id)
        .where(ClinicLocation.patient_alias == patient_id)
    )
    if file_id:
        query = query.where(ClinicArtifact.id == file_id)
    matches = {}
    for artifact, location in session.execute(query):
        if (
            file_id
            or file_key == artifact.file_key
            or (
                location.kind == "netlify"
                and file_key in (location.key, location.key.rsplit("/", 1)[-1])
            )
        ):
            matches[artifact.id] = artifact
    if not matches:
        raise CatalogueNotFound("File not found under this historical patient binding")
    if len(matches) != 1:
        raise CatalogueConflict("Historical key has multiple byte bindings; use fileId")
    artifact = next(iter(matches.values()))
    patient = session.get(storage.Patient, artifact.patient_uuid)
    if patient is None or not parse_canonical_patient_id(patient.label):
        raise CatalogueNotFound("Current canonical patient is unavailable")
    if (
        session.scalar(
            select(func.count())
            .select_from(storage.Patient)
            .where(storage.Patient.label == patient.label)
        )
        != 1
    ):
        raise CatalogueConflict("Current patient identity remains ambiguous")
    return patient, artifact


def _binding(session, patient_id, file_key=None, file_id=None):
    if bool(file_key) == bool(file_id):
        raise ValueError("Supply exactly one fileId or fileKey")
    try:
        alias = (
            session.get(ClinicPatientAlias, patient_id)
            if isinstance(patient_id, str)
            else None
        )
        if alias is not None and alias.ambiguous:
            raise CatalogueConflict("Historical patient alias is ambiguous")
        patient = _patient(session, patient_id)
    except CatalogueConflict:
        return _historical_binding(session, patient_id, file_key, file_id)
    if file_id:
        artifacts = list(
            session.scalars(
                select(ClinicArtifact).where(
                    ClinicArtifact.id == file_id,
                    ClinicArtifact.patient_uuid == patient.id,
                )
            )
        )
    else:
        artifacts = list(
            session.scalars(
                select(ClinicArtifact).where(
                    ClinicArtifact.patient_uuid == patient.id,
                    ClinicArtifact.file_key == file_key,
                )
            )
        )
        if not artifacts and isinstance(file_key, str) and "/" in file_key:
            artifacts = list(
                session.scalars(
                    select(ClinicArtifact)
                    .join(
                        ClinicLocation, ClinicLocation.artifact_id == ClinicArtifact.id
                    )
                    .where(
                        ClinicArtifact.patient_uuid == patient.id,
                        ClinicArtifact.file_key == file_key.rsplit("/", 1)[-1],
                        ClinicLocation.kind == "netlify",
                        ClinicLocation.key == file_key,
                    )
                ).unique()
            )
        # Exact database-issued file keys remain unambiguous when several file
        # identities share one content location after storage deduplication.
        if len(artifacts) == 1:
            return patient, artifacts[0]
        locations = session.scalars(
            select(ClinicLocation)
            .join(ClinicArtifact, ClinicArtifact.id == ClinicLocation.artifact_id)
            .where(
                ClinicArtifact.patient_uuid == patient.id,
                ClinicLocation.kind == "netlify",
            )
        )
        for location in locations:
            if file_key in (location.key, location.key.rsplit("/", 1)[-1]):
                artifact = session.get(ClinicArtifact, location.artifact_id)
                if artifact not in artifacts:
                    artifacts.append(artifact)
    if not artifacts:
        raise CatalogueNotFound("File not found")
    if len(artifacts) != 1:
        raise CatalogueConflict("Historical key has multiple byte bindings; use fileId")
    return patient, artifacts[0]


def file_binding(patient_id, *, file_key=None, file_id=None):
    with storage.session_scope() as session:
        session.execute(text("BEGIN"))
        patient, artifact = _binding(session, patient_id, file_key, file_id)
        return _envelope(session, **_artifact_json(session, artifact, patient))


def open_local_file(file_id):
    """Return a verified temporary byte snapshot, safe against in-place writes.

    Equivalent same-chart locations can supply the exact requested bytes when an
    imported representative has only a remote location. Every candidate is hashed.
    Large media rolls to disk at 8 MiB; headers and every streamed range bind to
    the snapshot's hash. No mutable source is reopened after verification.
    """
    with storage.session_scope() as session:
        artifact = session.get(ClinicArtifact, file_id)
        if artifact is None:
            raise CatalogueNotFound("File not found")
        locations = list(
            session.scalars(
                select(ClinicLocation)
                .join(ClinicArtifact, ClinicArtifact.id == ClinicLocation.artifact_id)
                .where(
                    ClinicArtifact.patient_uuid == artifact.patient_uuid,
                    ClinicArtifact.sha256 == artifact.sha256,
                    ClinicArtifact.size == artifact.size,
                    ClinicLocation.kind == "local",
                    ClinicLocation.active.is_(True),
                )
                .order_by(
                    case((ClinicLocation.artifact_id == file_id, 0), else_=1),
                    ClinicLocation.verified.desc(),
                    ClinicLocation.id,
                )
            )
        )
        for location in locations:
            snapshot = tempfile.SpooledTemporaryFile(max_size=8 * 1024 * 1024)
            try:
                path = _allowed_path(location.key)
                digest, size = hashlib.sha256(), 0
                with path.open("rb") as source:
                    while chunk := source.read(1024 * 1024):
                        snapshot.write(chunk)
                        digest.update(chunk)
                        size += len(chunk)
                if (digest.hexdigest(), size) == (artifact.sha256, artifact.size):
                    snapshot.seek(0)
                    return snapshot
            except (OSError, ValueError):
                pass
            snapshot.close()
        raise CatalogueUnavailable(
            "No readable exact local bytes; remote locations may remain available"
        )


def _technical(file):
    return file["documentKind"] in (
        "council-export",
        "data-pack",
        "vision-transcript",
        "technical",
    )


def _file_records(session, artifacts, patient):
    from collections import defaultdict
    from .clinic_records import ClinicFeedback

    artifacts = list(artifacts)
    ids = [a.id for a in artifacts]
    # Page responses carry serving locations only. Retired aliases remain in
    # file_binding and full-history responses for old links and audit lookup;
    # returning them on each page would restore the duplicate payload cost.
    locations, events = defaultdict(list), defaultdict(list)
    if ids:
        for row in session.scalars(
            select(ClinicLocation)
            .where(ClinicLocation.artifact_id.in_(ids), ClinicLocation.active.is_(True))
            .order_by(ClinicLocation.kind, ClinicLocation.key)
        ):
            locations[row.artifact_id].append(row)
        for row in session.scalars(
            select(ClinicFeedback)
            .where(ClinicFeedback.artifact_id.in_(ids))
            .order_by(ClinicFeedback.sequence)
        ):
            events[row.artifact_id].append(row)
    return [
        _artifact_json(
            session, a, patient, locations=locations[a.id], feedback_events=events[a.id]
        )
        for a in artifacts
    ]


def patient_files(
    patient_id,
    *,
    mode="full",
    limit=500,
    page=None,
    cursor=None,
    relative_path=None,
    sha256=None,
    if_index_version=None,
):
    mode = mode if mode in ("initial", "archive", "full", "delivery") else "full"
    try:
        limit = min(1000, int(limit)) if float(limit) > 0 else 500
    except (ValueError, TypeError, OverflowError):
        limit = 500
    with storage.session_scope() as session:
        session.execute(text("BEGIN"))
        patient = _patient(session, patient_id)
        _state(session, patient.id)
        patient_state = session.get(ClinicPatientCatalogState, patient.id)
        patient_revision = patient_state.revision if patient_state else 0
        query = (
            select(ClinicArtifact)
            .where(ClinicArtifact.patient_uuid == patient.id)
            .order_by(
                ClinicArtifact.session_date.desc(),
                _generation_time().desc(),
                ClinicArtifact.uploaded_at.desc(),
                ClinicArtifact.version.desc(),
                ClinicArtifact.id.desc(),
            )
        )
        total = session.scalar(
            select(func.count())
            .select_from(ClinicArtifact)
            .where(ClinicArtifact.patient_uuid == patient.id)
        )
        indexed_total = total
        if mode != "delivery":
            working_ids = _working_file_ids([patient.id])
            query = query.where(ClinicArtifact.id.in_(working_ids))
            total = session.scalar(
                select(func.count()).select_from(working_ids.subquery())
            )
        version = hashlib.sha256(
            _json(
                dict(
                    patientId=patient.label,
                    revision=patient_revision,
                    contractVersion=2,
                    fileViewRevision=1,
                )
            ).encode()
        ).hexdigest()
        common = dict(
            patientId=patient.label,
            mode=mode,
            totalFiles=total,
            totalIndexedFiles=indexed_total,
            indexUpdatedAt=patient_state.updated_at if patient_state else 0,
            indexVersion=version,
            contractVersion=2,
            clientContractVersion=2,
        )
        paged = mode == "archive" and (page == "1" or cursor is not None)
        if if_index_version == version and not paged and mode != "delivery":
            return _envelope(session, **common, files=[], unchanged=True)
        next_cursor = None
        if paged:
            offset = 0
            if cursor is not None:
                try:
                    if not isinstance(cursor, str) or not re.fullmatch(
                        "[A-Za-z0-9_-]{1,2048}", cursor
                    ):
                        raise ValueError()
                    value = json.loads(
                        base64.urlsafe_b64decode(cursor + "=" * (-len(cursor) % 4))
                    )
                    if (
                        value["patientId"] != patient.label
                        or type(value["limit"]) is not int
                        or value["limit"] != limit
                        or type(value["offset"]) is not int
                        or value["offset"] <= 0
                        or not isinstance(value["indexVersion"], str)
                    ):
                        raise ValueError()
                except (ValueError, KeyError, TypeError) as error:
                    raise ValueError("Invalid archive cursor") from error
                if value["indexVersion"] != version:
                    raise CatalogueConflict("Files changed; reload the archive")
                offset = value["offset"]
                if offset >= total or offset % limit:
                    raise ValueError("Invalid archive offset")
            selected = _file_records(
                session, session.scalars(query.offset(offset).limit(limit)), patient
            )
            if offset + len(selected) < total:
                next_cursor = (
                    base64.urlsafe_b64encode(
                        _json(
                            dict(
                                patientId=patient.label,
                                indexVersion=version,
                                offset=offset + len(selected),
                                limit=limit,
                            )
                        ).encode()
                    )
                    .decode()
                    .rstrip("=")
                )
            return _envelope(
                session,
                **common,
                files=selected,
                returnedFiles=len(selected),
                limit=limit,
                nextCursor=next_cursor,
                truncated=next_cursor is not None,
                technicalDeferred=False,
            )
        if mode in ("initial", "archive"):
            technical_kinds = (
                "council-export",
                "data-pack",
                "vision-transcript",
                "technical",
            )
            technical_filter = _document_kind().in_(technical_kinds)
            reviewable_filter = or_(_document_kind().is_(None), ~technical_filter)
            technical_count = session.scalar(
                select(func.count())
                .select_from(ClinicArtifact)
                .where(ClinicArtifact.patient_uuid == patient.id, technical_filter)
            )
            reviewable_count = total - technical_count
            selected_artifacts = list(
                session.scalars(query.where(reviewable_filter).limit(limit))
            )
            remaining = min(80, max(0, limit - len(selected_artifacts)))
            if remaining:
                selected_artifacts.extend(
                    session.scalars(query.where(technical_filter).limit(remaining))
                )
            if mode == "initial":
                heroes = []
                for kind in ("patient-summary", "video"):
                    heroes.extend(
                        session.scalars(query.where(_document_kind() == kind).limit(25))
                    )
                selected_artifacts = list(
                    {a.id: a for a in heroes + selected_artifacts}.values()
                )
            selected = _file_records(session, selected_artifacts, patient)
            return _envelope(
                session,
                **common,
                files=selected,
                returnedFiles=len(selected),
                limit=limit,
                truncated=len(selected) < total,
                technicalDeferred=technical_count > 80,
                deferredArchive=len(selected) < total,
                reviewableFiles=reviewable_count,
                technicalFiles=technical_count,
            )
        if mode == "delivery":
            if (
                not isinstance(relative_path, str)
                or not 0 < len(relative_path) <= 2048
                or re.search(r"[\\:\x00-\x1f\x7f]", relative_path)
                or any(p in ("", ".", "..") for p in relative_path.split("/"))
                or not re.fullmatch("[a-f0-9]{64}", sha256 or "")
            ):
                raise ValueError("Invalid relativePath or sha256")
            files = [
                _artifact_json(session, artifact, patient)
                for artifact in session.scalars(
                    query.where(ClinicArtifact.sha256 == sha256)
                )
            ]
            names = {relative_path, relative_path.rsplit("/", 1)[-1]}
            files_matching = [
                f
                for f in files
                if f["sha256"] == sha256
                and f["hashVerified"]
                and names.intersection(
                    (
                        f["originalName"],
                        f["logicalName"],
                        f["displayName"],
                        f["provenance"].get("relativePath"),
                    )
                )
            ]
            selected = files_matching[:25]
            return _envelope(
                session,
                **common,
                files=selected,
                returnedFiles=len(selected),
                truncated=len(files_matching) > len(selected),
            )
        files = [
            _artifact_json(session, artifact, patient)
            for artifact in session.scalars(query)
        ]
        selected = files
        payload = dict(
            files=selected,
            returnedFiles=len(selected),
            limit=limit,
            truncated=next_cursor is not None if paged else len(selected) < len(files),
            technicalDeferred=False
            if paged
            else sum(_technical(f) for f in files) > 80,
            deferredArchive=len(selected) < len(files),
            reviewableFiles=sum(not _technical(f) for f in files),
            technicalFiles=sum(_technical(f) for f in files),
        )
        if paged:
            payload["nextCursor"] = next_cursor
        return _envelope(session, **common, **payload)


def report_dates(patient_ids):
    if (
        not isinstance(patient_ids, list)
        or len(patient_ids) > 500
        or not all(isinstance(p, str) for p in patient_ids)
    ):
        raise ValueError("At most 500 patientIds are required")
    with storage.session_scope() as session:
        session.execute(text("BEGIN"))
        dates = {}
        for patient_id in dict.fromkeys(patient_ids):
            patient = _patient(session, patient_id)
            _state(session, patient.id)
            dates[patient.label] = session.scalar(
                select(func.max(ClinicArtifact.session_date)).where(
                    ClinicArtifact.patient_uuid == patient.id
                )
            )
        return _envelope(session, patientReportDates=dates)
