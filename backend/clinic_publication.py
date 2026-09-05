"""Free byte projection of original database-issued artifacts; no generation."""

import base64
import json
import os
import selectors
import shutil
import subprocess
import time
from sqlalchemy import select
from . import storage
from .clinic_catalogue import _write, _location, _bump, verify_remote_location
from .clinic_catalogue_reads import _patient, _envelope, _state, _json
from .clinic_models import (
    ClinicArtifact,
    ClinicLocation,
    ClinicPublication,
    CatalogueNotFound,
    CatalogueConflict,
    CatalogueUnavailable,
)


def _publication_item(session, artifact):
    patient = session.get(storage.Patient, artifact.patient_uuid)
    binding = session.get(ClinicPublication, artifact.id)
    selected = (
        session.scalar(
            select(ClinicLocation).where(
                ClinicLocation.artifact_id == artifact.id,
                ClinicLocation.kind == "netlify",
                ClinicLocation.key == binding.remote_key,
                ClinicLocation.active.is_(True),
            )
        )
        if binding
        else None
    )
    return dict(
        fileId=artifact.id,
        patientId=patient.label,
        source=dict(kind=artifact.source_kind, id=artifact.source_id),
        fileKey=artifact.file_key,
        remoteKey=binding.remote_key if binding else None,
        sha256=artifact.sha256,
        size=artifact.size,
        catalogRevision=_state(session).revision,
        verified=bool(selected and selected.verified),
    )


def publication_items(patient_id, *, limit=100, cursor=None):
    try:
        limit = int(limit)
        if not 1 <= limit <= 500:
            raise ValueError()
    except (ValueError, TypeError):
        raise ValueError("Invalid publication limit") from None
    with storage.session_scope() as s:
        from sqlalchemy import text

        s.execute(text("BEGIN"))
        patient = _patient(s, patient_id)
        revision = _state(s, patient.id).revision
        after = ""
        if cursor:
            try:
                if len(cursor) > 2048:
                    raise ValueError()
                c = json.loads(base64.urlsafe_b64decode(cursor.encode()))
                if (
                    set(c) != {"patientId", "revision", "after"}
                    or c["patientId"] != patient.label
                ):
                    raise ValueError()
                if c["revision"] != revision:
                    raise CatalogueConflict("Publication revision changed")
                after = c["after"]
                if not isinstance(after, str):
                    raise ValueError()
            except (ValueError, TypeError, KeyError) as e:
                if isinstance(e, CatalogueConflict):
                    raise
                raise ValueError("Invalid publication cursor") from e
        rows = list(
            s.scalars(
                select(ClinicArtifact)
                .where(
                    ClinicArtifact.patient_uuid == patient.id, ClinicArtifact.id > after
                )
                .order_by(ClinicArtifact.id)
                .limit(limit + 1)
            )
        )
        next_cursor = (
            base64.urlsafe_b64encode(
                _json(
                    dict(
                        patientId=patient.label,
                        revision=revision,
                        after=rows[limit - 1].id,
                    )
                ).encode()
            ).decode()
            if len(rows) > limit
            else None
        )
        return _envelope(
            s,
            items=[_publication_item(s, a) for a in rows[:limit]],
            nextCursor=next_cursor,
        )


def prepare_publication(file_id):
    with _write() as s:
        artifact = s.get(ClinicArtifact, file_id)
        if artifact is None:
            raise CatalogueNotFound("Artifact not found")
        patient = s.get(storage.Patient, artifact.patient_uuid)
        _patient(s, patient.label)
        item = _publication_item(s, artifact)
        if item["remoteKey"] is None:
            key = f"patients/{patient.label}/files/{artifact.file_key}"
            occupied = s.scalar(
                select(ClinicLocation).where(
                    ClinicLocation.kind == "netlify",
                    ClinicLocation.key == key,
                    ClinicLocation.artifact_id != artifact.id,
                )
            )
            if occupied is not None:
                raise CatalogueConflict(
                    "Publication key already binds another artifact"
                )
            s.add(ClinicPublication(artifact_id=artifact.id, remote_key=key))
            affected = _location(s, artifact, "netlify", key, patient.label, False)
            if affected:
                _bump(s, affected)
            s.flush()
        return _envelope(s, item=_publication_item(s, artifact))


def _helper_bytes(
    command, *, cwd, key, size, timeout=300, request_payload=None, stop_event=None
):
    """Fixed trusted helper, bounded streaming; nonzero exit never completes."""
    proc = subprocess.Popen(
        command,
        cwd=cwd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    selector = selectors.DefaultSelector()
    deadline = time.monotonic() + timeout
    count = 0
    errors = bytearray()
    try:
        request = (
            _json(
                request_payload
                if request_payload is not None
                else dict(schemaVersion=1, key=key, maxBytes=size)
            ).encode()
            + b"\n"
        )
        if len(request) > 16384:
            raise CatalogueUnavailable("Readback request exceeds limit")
        proc.stdin.write(request)
        proc.stdin.close()
        for stream in (proc.stdout, proc.stderr):
            os.set_blocking(stream.fileno(), False)
            selector.register(stream, selectors.EVENT_READ)
        while selector.get_map():
            if stop_event is not None and stop_event.is_set():
                raise CatalogueUnavailable("Strong readback cancelled")
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise CatalogueUnavailable("Strong readback timed out")
            for entry, _ in selector.select(min(remaining, 0.1)):
                chunk = os.read(entry.fileobj.fileno(), 65536)
                if not chunk:
                    selector.unregister(entry.fileobj)
                    continue
                if entry.fileobj is proc.stderr:
                    errors.extend(chunk)
                    if len(errors) > 4096:
                        raise CatalogueUnavailable("Strong readback failed")
                else:
                    count += len(chunk)
                    if count > size:
                        raise ReadbackOversize("Strong readback exceeds original size")
                    yield chunk
        while proc.poll() is None:
            if stop_event is not None and stop_event.is_set():
                raise CatalogueUnavailable("Strong readback cancelled")
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise CatalogueUnavailable("Strong readback timed out")
            try:
                proc.wait(timeout=min(remaining, 0.1))
            except subprocess.TimeoutExpired:
                continue
        if proc.returncode != 0:
            raise CatalogueUnavailable("Strong readback failed")
    except (OSError, subprocess.SubprocessError) as error:
        raise CatalogueUnavailable("Strong readback unavailable") from error
    finally:
        selector.close()
        if proc.poll() is None:
            proc.kill()
        proc.wait()
        for stream in (proc.stdin, proc.stdout, proc.stderr):
            stream.close()


def strong_readback(key, size, *, stop_event=None):
    from .portal_sync import portal_sync_repo

    root = portal_sync_repo().resolve()
    helper = root / "scripts/qeeg_clinic_blob_readback.mjs"
    node = shutil.which("node")
    if not node or not helper.is_file():
        raise CatalogueUnavailable("Trusted byte helper unavailable")
    yield from _helper_bytes(
        [node, str(helper)], cwd=root, key=key, size=size, stop_event=stop_event
    )


def verify_publication(file_id, key, *, stop_event=None):
    with storage.session_scope() as s:
        artifact = s.get(ClinicArtifact, file_id)
        if artifact is None:
            raise CatalogueNotFound("Artifact not found")
        location = s.scalar(
            select(ClinicLocation).where(
                ClinicLocation.artifact_id == file_id,
                ClinicLocation.kind == "netlify",
                ClinicLocation.key == key,
                ClinicLocation.active.is_(True),
            )
        )
        if location is None:
            raise CatalogueConflict("Remote location is not registered")
        size = artifact.size
    try:

        def readback():
            return (
                strong_readback(key, size, stop_event=stop_event)
                if stop_event is not None
                else strong_readback(key, size)
            )

        verify_remote_location(file_id, key, readback)
    except ReadbackOversize:
        # Observed bytes exceeded the original size: this is positive mismatch
        # evidence, so an old durable receipt must be revoked as well.
        with _write() as s:
            location = s.scalar(
                select(ClinicLocation).where(
                    ClinicLocation.artifact_id == file_id,
                    ClinicLocation.kind == "netlify",
                    ClinicLocation.key == key,
                )
            )
            if location is not None and location.verified:
                location.verified = False
                location.verified_at = None
                _bump(s, s.get(ClinicArtifact, file_id).patient_uuid)
        raise CatalogueConflict("Remote bytes exceed original artifact") from None
    with storage.session_scope() as s:
        return _envelope(s, item=_publication_item(s, s.get(ClinicArtifact, file_id)))


class ReadbackOversize(CatalogueUnavailable):
    """Strong readback positively observed more bytes than the source permits."""
