"""Shared clinical catalogue. Internal APIs accept producer identities, never public paths.

Standalone writes use BEGIN IMMEDIATE. The existing ORM producer hook participates
in the caller's transaction. Byte verification never creates a paid operation.
"""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime
import hashlib
import json
import mimetypes
import os
import re
import time
import uuid
from weakref import WeakSet

from sqlalchemy import event, func, inspect, select, text, update
from sqlalchemy.orm import Session
from sqlalchemy.dialects.sqlite import insert
from .clinic_models import ClinicPatientCatalogState

from . import storage
from .clinic_catalogue_reads import (
    _allowed_path,
    _artifact_json,
    _fingerprint,
    _json,
    _millis,
)
from .clinic_models import CatalogueConflict, CatalogueNotFound, CatalogueUnavailable
from .clinic_models import (
    ClinicArtifact,
    ClinicCatalogState,
    ClinicLocation,
    ClinicPatientAlias,
    ClinicProjection,
)
from .patient_identity import parse_canonical_patient_id


_initialized_engines = WeakSet()


def _now():
    return time.time_ns() // 1_000_000


@contextmanager
def _write():
    with storage.session_scope() as session:
        session.execute(text("BEGIN IMMEDIATE"))
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise


def initialize_catalogue():
    """Only a DB without preexisting patient/source rows starts imported."""
    with _write() as session:
        if session.get(ClinicCatalogState, 1) is None:
            populated = any(
                session.scalar(select(func.count()).select_from(model))
                for model in (storage.Patient, storage.Report, storage.PatientFile)
            )
            session.add(
                ClinicCatalogState(
                    id=1, revision=0, import_complete=not populated, updated_at=_now()
                )
            )

    _initialized_engines.add(storage.engine)


def _bump(session, patient_uuid=None):
    session.execute(
        update(ClinicCatalogState)
        .where(ClinicCatalogState.id == 1)
        .values(revision=ClinicCatalogState.revision + 1, updated_at=_now())
    )
    revision = session.scalar(
        select(ClinicCatalogState.revision).where(ClinicCatalogState.id == 1)
    )
    if revision is None:
        raise CatalogueUnavailable("Catalogue state is missing")
    patients = (
        {patient_uuid} if isinstance(patient_uuid, str) else (patient_uuid or set())
    )
    for patient in patients:
        _touch_patient_revision(session, patient, revision)
    return revision


def _touch_patient_revision(session, patient_uuid, revision):
    statement = insert(ClinicPatientCatalogState).values(
        patient_uuid=patient_uuid, revision=revision, updated_at=_now()
    )
    session.execute(
        statement.on_conflict_do_update(
            index_elements=["patient_uuid"],
            set_={"revision": revision, "updated_at": _now()},
        )
    )


def complete_catalogue_import(manifest: dict):
    """Local importer attests an explicit complete inventory; no startup auto-import.

    Manifest must enumerate unresolved legacy rows and account for all existing
    Report/PatientFile sources. The importer owns external tree/object census.
    """
    if (
        not isinstance(manifest, dict)
        or not manifest.get("inventoryId")
        or not isinstance(manifest.get("legacyPatientIds"), list)
    ):
        raise ValueError("An explicit inventory manifest is required")
    with _write() as session:
        labels = [
            p.label
            for p in session.scalars(select(storage.Patient))
            if parse_canonical_patient_id(p.label)
        ]
        if len(set(labels)) != len(labels):
            raise CatalogueConflict(
                "Duplicate canonical patients require explicit resolution"
            )
        legacy = {
            p.id
            for p in session.scalars(select(storage.Patient))
            if not parse_canonical_patient_id(p.label)
        }
        if legacy != set(manifest["legacyPatientIds"]):
            raise CatalogueConflict("Legacy inventory does not match existing patients")
        for model, kind in (
            (storage.Report, "report"),
            (storage.PatientFile, "patient-file"),
        ):
            for row in session.scalars(select(model)):
                if not session.scalar(
                    select(ClinicArtifact.id).where(
                        ClinicArtifact.source_kind == kind,
                        ClinicArtifact.source_id == row.id,
                    )
                ):
                    raise CatalogueUnavailable("Existing sources remain unimported")
        if session.scalar(
            select(ClinicProjection.id)
            .where(ClinicProjection.artifact_id.is_(None))
            .limit(1)
        ):
            raise CatalogueUnavailable("Source projections remain incomplete")
        state = session.get(ClinicCatalogState, 1)
        payload = _json(manifest)
        if state.import_complete and state.import_manifest == payload:
            return
        state.import_complete = True
        state.import_manifest = payload
        _bump(session)


def _hash_chunks(chunks):
    digest, size = hashlib.sha256(), 0
    for chunk in chunks:
        if not isinstance(chunk, bytes):
            raise ValueError("Byte readback is required")
        digest.update(chunk)
        size += len(chunk)
    return digest.hexdigest(), size


def _read_local(path):
    try:
        path = _allowed_path(path)
        with path.open("rb") as stream:
            before = stream.fileno()
            stat = os.fstat(before)
            digest, size = _hash_chunks(iter(lambda: stream.read(1024 * 1024), b""))
            after = os.fstat(before)
            current = path.stat()
            if (stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns) != (
                after.st_ino,
                after.st_size,
                after.st_mtime_ns,
                after.st_ctime_ns,
            ) or (
                after.st_ino,
                after.st_size,
                after.st_mtime_ns,
                after.st_ctime_ns,
            ) != (
                current.st_ino,
                current.st_size,
                current.st_mtime_ns,
                current.st_ctime_ns,
            ):
                raise CatalogueUnavailable("Local source changed during verification")
        return (
            str(path),
            digest,
            size,
            _json(
                [
                    after.st_dev,
                    after.st_ino,
                    after.st_size,
                    after.st_mtime_ns,
                    after.st_ctime_ns,
                ]
            ),
        )
    except OSError as error:
        raise CatalogueUnavailable("Local source is unreadable") from error


def _source(session, kind, source_id):
    return session.scalar(
        select(ClinicArtifact).where(
            ClinicArtifact.source_kind == kind, ClinicArtifact.source_id == source_id
        )
    )


def _location(session, artifact, kind, key, patient_alias, verified, fingerprint=None):
    location = session.scalar(
        select(ClinicLocation).where(
            ClinicLocation.artifact_id == artifact.id,
            ClinicLocation.kind == kind,
            ClinicLocation.key == key,
        )
    )
    others = list(
        session.scalars(
            select(ClinicLocation).where(
                ClinicLocation.kind == kind,
                ClinicLocation.key == key,
                ClinicLocation.artifact_id != artifact.id,
                ClinicLocation.active.is_(True),
            )
        )
    )
    affected = set()
    for other in others:
        previous = session.get(ClinicArtifact, other.artifact_id)
        if (previous.sha256, previous.size) != (artifact.sha256, artifact.size):
            other.active = False
            other.verified = False
            affected.add(previous.patient_uuid)
    if location is None:
        location = ClinicLocation(
            id=str(uuid.uuid4()),
            artifact_id=artifact.id,
            kind=kind,
            key=key,
            patient_alias=patient_alias,
            verified=verified,
            active=True,
            verified_at=_now() if verified else None,
            fingerprint=fingerprint,
        )
        session.add(location)
        return affected | {artifact.patient_uuid}
    changed = (
        any(not other.active for other in others)
        or not location.active
        or (verified and (not location.verified or location.fingerprint != fingerprint))
    )
    if changed:
        location.active = True
        location.verified = verified
        location.verified_at = _now() if verified else None
        location.fingerprint = fingerprint
    return affected | ({artifact.patient_uuid} if changed else set())


def _register(
    session,
    *,
    patient_uuid,
    source_kind,
    source_id,
    original_name,
    logical_family,
    local_path=None,
    sha256=None,
    size=None,
    file_key=None,
    content_type=None,
    document_kind=None,
    session_date=None,
    generated_at=None,
    provenance=None,
    uploaded_at=None,
    uploaded_by=None,
    _verified_local=None,
):
    patient = session.get(storage.Patient, patient_uuid)
    if patient is None:
        # The caller may be flushing a new Patient and its first source together.
        patient = next(
            (
                p
                for p in session.new
                if isinstance(p, storage.Patient) and p.id == patient_uuid
            ),
            None,
        )
    if patient is None:
        raise CatalogueNotFound("Patient not found")
    if not all(
        isinstance(v, str) and v
        for v in (source_kind, source_id, original_name, logical_family)
    ):
        raise ValueError(
            "Stable producer identity, original name and family are required"
        )
    if file_key is not None and (
        not isinstance(file_key, str)
        or not file_key
        or len(file_key) > 2048
        or re.search(r"[/\\\x00-\x1f\x7f]", file_key)
        or file_key.startswith("$")
    ):
        raise ValueError("Invalid stable file key")
    if uploaded_at is not None and (type(uploaded_at) is not int or uploaded_at < 0):
        raise ValueError("Original upload time must be Unix milliseconds")
    verified = local_path is not None
    if verified:
        local_path, actual_hash, actual_size, fingerprint = (
            _verified_local or _read_local(local_path)
        )
        if _fingerprint(local_path) != fingerprint:
            raise CatalogueUnavailable("Local source changed before registration")
        if (sha256 is not None and sha256 != actual_hash) or (
            size is not None and size != actual_size
        ):
            raise CatalogueConflict("Source bytes differ from the binding")
        sha256, size = actual_hash, actual_size
    if (
        not isinstance(sha256, str)
        or not re.fullmatch("[a-f0-9]{64}", sha256)
        or type(size) is not int
        or size < 0
    ):
        raise ValueError("Exact SHA256 and size are required")
    if session_date is not None:
        if (
            datetime.strptime(session_date, "%Y-%m-%d").strftime("%Y-%m-%d")
            != session_date
        ):
            raise ValueError("Invalid session date")
    if generated_at is not None and (type(generated_at) is not int or generated_at < 0):
        raise ValueError("Generation time must be original Unix milliseconds")
    if provenance is not None and not isinstance(provenance, dict):
        raise ValueError("Producer provenance must be an object")
    provenance_json = _json(provenance or {})
    artifact = _source(session, source_kind, source_id)
    material = dict(
        patient_uuid=patient_uuid,
        original_name=original_name,
        logical_family=logical_family,
        sha256=sha256,
        size=size,
        content_type=content_type
        or mimetypes.guess_type(original_name)[0]
        or "application/octet-stream",
        document_kind=document_kind,
        session_date=session_date,
        generated_at=generated_at,
        provenance_json=provenance_json,
        uploaded_at=uploaded_at,
        uploaded_by=uploaded_by,
    )
    if artifact:
        if any(getattr(artifact, key) != value for key, value in material.items()) or (
            file_key is not None and artifact.file_key != file_key
        ):
            raise CatalogueConflict(
                "Producer identity is already bound to different material"
            )
        affected = set()
    else:
        version = (
            session.scalar(
                select(func.max(ClinicArtifact.version)).where(
                    ClinicArtifact.patient_uuid == patient_uuid,
                    ClinicArtifact.logical_family == logical_family,
                )
            )
            or 0
        ) + 1
        # Include other unflushed source registrations in the same producer transaction.
        version = max(
            [version]
            + [
                a.version + 1
                for a in session.new
                if isinstance(a, ClinicArtifact)
                and a.patient_uuid == patient_uuid
                and a.logical_family == logical_family
            ]
        )
        artifact_id = str(uuid.uuid4())
        artifact = ClinicArtifact(
            id=artifact_id,
            source_kind=source_kind,
            source_id=source_id,
            version=version,
            file_key=file_key or artifact_id,
            registered_at=_now(),
            archived=False,
            **material,
        )
        session.add(artifact)
        affected = {patient_uuid}
    if local_path:
        affected.update(
            _location(
                session,
                artifact,
                "local",
                str(local_path),
                patient.label,
                True,
                fingerprint,
            )
        )
    projection = session.scalar(
        select(ClinicProjection).where(
            ClinicProjection.source_kind == source_kind,
            ClinicProjection.source_id == source_id,
        )
    )
    if projection is not None and projection.artifact_id != artifact.id:
        projection.artifact_id, projection.error = artifact.id, None
        affected.add(patient_uuid)
    return artifact, affected


def register_artifact(**kwargs):
    """Internal source admission; concurrent exact replay retains file ID/version/revision.

    source_kind + source_id is immutable across retries. A changed mutable source
    requires its next real producer identity. No filename or hash-only deduplication.
    Local bytes are verified here; remote-only metadata remains unverified.
    """
    if "_verified_local" in kwargs:
        raise ValueError("Source verification is internal")
    if kwargs.get("local_path") is not None:
        kwargs["_verified_local"] = _read_local(kwargs["local_path"])
    with _write() as session:
        artifact, affected = _register(session, **kwargs)
        if affected:
            _bump(session, affected)
        session.flush()
        return _artifact_json(
            session, artifact, session.get(storage.Patient, artifact.patient_uuid)
        )


def _remote_key(key):
    if (
        not isinstance(key, str)
        or len(key) > 2048
        or re.search(r"[\\\x00-\x1f\x7f]", key)
    ):
        raise ValueError("Invalid remote binding")
    parts = key.split("/")
    if (
        len(parts) != 4
        or parts[0] != "patients"
        or parts[2] != "files"
        or not parts[1]
        or not parts[3]
        or parts[3] in (".", "..")
        or parts[3].startswith("$")
    ):
        raise ValueError("Expected a patient file object key")
    return parts[1]


def add_remote_location(file_id, key):
    """Register remote inventory without granting verified receipt status."""
    alias = _remote_key(key)
    with _write() as session:
        artifact = session.get(ClinicArtifact, file_id)
        if artifact is None:
            raise CatalogueNotFound("File not found")
        patient = session.get(storage.Patient, artifact.patient_uuid)
        known = session.get(ClinicPatientAlias, alias)
        if alias != patient.label and (
            known is None or known.patient_uuid != patient.id
        ):
            raise CatalogueConflict("Remote key belongs to an unbound patient alias")
        affected = _location(session, artifact, "netlify", key, alias, False)
        if affected:
            _bump(session, affected)


def verify_remote_location(file_id, key, readback):
    """Internal byte worker supplies a callable yielding actual remote response bytes.

    Listing/client hashes and timestamps cannot grant verification. Readback is
    outside the short write transaction; the exact artifact binding is rechecked.
    """
    _remote_key(key)
    digest, size = _hash_chunks(readback())
    with _write() as session:
        artifact = session.get(ClinicArtifact, file_id)
        if artifact is None:
            raise CatalogueNotFound("File not found")
        matches = (digest, size) == (artifact.sha256, artifact.size)
        location = session.scalar(
            select(ClinicLocation).where(
                ClinicLocation.artifact_id == file_id,
                ClinicLocation.kind == "netlify",
                ClinicLocation.key == key,
                ClinicLocation.active.is_(True),
            )
        )
        if location is None:
            raise CatalogueConflict(
                "Remote location must be registered before readback"
            )
        if location.verified != matches:
            location.verified, location.verified_at = (
                matches,
                _now() if matches else None,
            )
            _bump(session, artifact.patient_uuid)
    if not matches:
        raise CatalogueConflict("Remote bytes differ from the artifact")


def resolve_projection(source_kind, source_id):
    """Replay one durable incomplete existing producer projection."""
    with _write() as session:
        projection = session.scalar(
            select(ClinicProjection).where(
                ClinicProjection.source_kind == source_kind,
                ClinicProjection.source_id == source_id,
            )
        )
        if projection is None:
            artifact = _source(session, source_kind, source_id)
            if artifact is None:
                raise CatalogueNotFound("Projection not found")
        else:
            artifact, affected = _register(
                session, **json.loads(projection.payload_json)
            )
            if affected:
                _bump(session, affected)
        session.flush()
        return _artifact_json(
            session, artifact, session.get(storage.Patient, artifact.patient_uuid)
        )


def _producer_hook(session, flush_context, instances):
    # Offline identity migrations intentionally use an uninitialized old schema.
    # Only a successfully initialized authority participates in this projection.
    if (
        session.get_bind() is not storage.engine
        or storage.engine not in _initialized_engines
    ):
        return
    patients = [
        p
        for p in session.new | session.dirty
        if isinstance(p, storage.Patient)
        and (p in session.new or session.is_modified(p, include_collections=False))
    ]
    sources = [
        p for p in session.new if isinstance(p, (storage.Report, storage.PatientFile))
    ]
    removed = [p for p in session.deleted if isinstance(p, storage.PatientFile)]
    if not patients and not sources and not removed:
        return
    # Acquire SQLite's writer lock before catalogue queries and version selection.
    revision = _bump(session)
    for patient in patients:
        if not patient.id:
            patient.id = str(uuid.uuid4())
        if revision is not None:
            _touch_patient_revision(session, patient.id, revision)
        history = inspect(patient).attrs.label.history
        for label in set(list(history.deleted) + [patient.label]):
            if not parse_canonical_patient_id(label):
                continue
            alias = session.get(ClinicPatientAlias, label)
            if alias is None:
                session.add(ClinicPatientAlias(alias=label, patient_uuid=patient.id))
            elif alias.patient_uuid != patient.id:
                # Existing low-level legacy writers can retain duplicate labels.
                # Preserve the conflict even after both rows are later relabelled.
                alias.ambiguous = True
                if revision is not None:
                    _touch_patient_revision(session, alias.patient_uuid, revision)
    for source in sources:
        if revision is not None:
            _touch_patient_revision(session, source.patient_id, revision)
        if not source.id:
            source.id = str(uuid.uuid4())
        kind = "report" if isinstance(source, storage.Report) else "patient-file"
        if source.created_at is None:
            source.created_at = storage._utcnow()
        args = dict(
            patient_uuid=source.patient_id,
            source_kind=kind,
            source_id=source.id,
            original_name=source.filename,
            logical_family=f"{kind}:{source.filename}",
            local_path=source.stored_path,
            content_type=source.mime_type,
            document_kind="source-report" if kind == "report" else None,
            provenance={kind + "Id": source.id},
            uploaded_at=_millis(source.created_at),
        )
        # Trusted original intake metadata joins the source in this same flush.
        args.update(session.info.get("clinic_source_metadata", {}).get(source.id, {}))
        try:
            _, affected = _register(session, **args)
            for patient_uuid in affected:
                _touch_patient_revision(session, patient_uuid, revision)
        except (CatalogueUnavailable, CatalogueNotFound, ValueError) as error:
            if isinstance(error, CatalogueConflict):
                raise
            session.add(
                ClinicProjection(
                    id=str(uuid.uuid4()),
                    patient_uuid=source.patient_id,
                    source_kind=kind,
                    source_id=source.id,
                    payload_json=_json(args),
                    error=type(error).__name__,
                    artifact_id=None,
                )
            )
    for source in removed:
        if revision is not None:
            _touch_patient_revision(session, source.patient_id, revision)
        artifact = _source(session, "patient-file", source.id)
        if artifact:
            artifact.archived = True


event.listen(Session, "before_flush", _producer_hook)


def register_patient_alias(patient_uuid: str, alias: str):
    """Importer binds an observed historical key to an existing patient row.

    This neither allocates nor renames a patient. Existing current/retired keys
    cannot be stolen from another patient, including noncanonical legacy rows.
    """
    if (
        not isinstance(alias, str)
        or not 0 < len(alias) <= 256
        or re.search(r"[/\\\x00-\x1f\x7f]", alias)
    ):
        raise ValueError("Invalid historical patient alias")
    with _write() as session:
        if session.get(storage.Patient, patient_uuid) is None:
            raise CatalogueNotFound("Patient not found")
        occupants = list(
            session.scalars(
                select(storage.Patient.id).where(storage.Patient.label == alias)
            )
        )
        current = session.get(ClinicPatientAlias, alias)
        if any(p != patient_uuid for p in occupants) or (
            current and (current.patient_uuid != patient_uuid or current.ambiguous)
        ):
            raise CatalogueConflict("Historical key belongs to another patient")
        if current is None:
            session.add(ClinicPatientAlias(alias=alias, patient_uuid=patient_uuid))
            _bump(session, patient_uuid)
