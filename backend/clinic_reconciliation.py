"""Immutable read-only census and explicit replayable free catalogue import.

Callers supply complete remote SDK census pages and actual strong byte reads.
Original cross-source bindings are evidence, never inferred from equal filenames.
"""

from pathlib import Path
import hashlib
import json
import os
import re
import uuid
from sqlalchemy import select, inspect
from . import storage
from .clinic_catalogue import (
    register_artifact,
    add_remote_location,
    verify_remote_location,
    _write,
    _bump,
)
from .clinic_catalogue_reads import _json, _millis
from .clinic_models import (
    ClinicArtifact,
    ClinicProjection,
    CatalogueConflict,
    CatalogueUnavailable,
)
from .clinic_intake import _immutable, upload_lock
from .patient_identity import parse_canonical_patient_id

ROOT_OPERATIONAL_FILES = frozenset(
    {
        ".DS_Store",
        ".qeeg_portal_netlify_sync.lock",
        ".qeeg_portal_sync_state.json",
        "_README.txt",
        ".qeeg_portal_local_pipeline_state.json",
        ".qeeg_portal_sync_watch_state.json",
        ".qeeg_portal_netlify_sync.spawn.lock",
    }
)


def _digest(chunks, limit):
    digest = hashlib.sha256()
    size = 0
    for chunk in chunks:
        if not isinstance(chunk, bytes):
            raise ValueError("Actual source bytes required")
        size += len(chunk)
        if size > limit:
            raise CatalogueUnavailable("Source exceeds explicit inventory byte limit")
        digest.update(chunk)
    return digest.hexdigest(), size


def _source_path(stored):
    p = Path(stored)
    return p if p.is_absolute() else Path(__file__).resolve().parents[1] / p


def _inventory_root(inventory_id):
    from .clinic_intake import require_key

    require_key(inventory_id)
    # Key characters may contain separators; the database inventory ID remains
    # original while its directory uses a collision-resistant exact key digest.
    return (
        Path(storage.DATA_DIR)
        / "clinic_inventory"
        / hashlib.sha256(inventory_id.encode()).hexdigest()
    )


def validate_remote_census(events):
    """Require every ordered SDK page and its successful terminal census."""
    page = 0
    keys = set()
    complete = False
    for event in events:
        if complete or not isinstance(event, dict):
            raise CatalogueUnavailable("Malformed remote census")
        if event.get("type") == "page":
            page += 1
            if (
                set(event) != {"type", "page", "keys"}
                or type(event["page"]) is not int
                or event["page"] != page
                or not isinstance(event["keys"], list)
            ):
                raise CatalogueUnavailable("Malformed remote page")
            for key in event["keys"]:
                if (
                    not isinstance(key, str)
                    or not key
                    or len(key.encode()) > 2048
                    or key in keys
                ):
                    raise CatalogueUnavailable("Repeated or invalid remote key")
                keys.add(key)
                yield key
        elif event.get("type") == "complete":
            if (
                set(event) != {"type", "pages", "keyCount"}
                or type(event["pages"]) is not int
                or type(event["keyCount"]) is not int
                or event["pages"] != page
                or event["keyCount"] != len(keys)
            ):
                raise CatalogueUnavailable("Incomplete remote census")
            complete = True
        else:
            raise CatalogueUnavailable("Unknown remote census event")
    if not complete:
        raise CatalogueUnavailable("Remote census lacks successful completion")


def _retained_remote_evidence(records):
    """Validate explicit original global namespaces, never a chart exclusion."""
    result = {}
    for row in records:
        if (not isinstance(row, dict) or not {"key", "category", "sha256", "size"}.issubset(row)
                or set(row) - {"key", "category", "sha256", "size", "metadata"}
                or not isinstance(row["key"], str) or not row["key"].isprintable()
                or not re.fullmatch(r"[a-f0-9]{64}", str(row["sha256"]))
                or type(row["size"]) is not int or row["size"] < 0):
            raise ValueError("Invalid retained remote evidence")
        key, category = row["key"], row["category"]
        if key in result:
            raise ValueError("Duplicate retained remote evidence")
        if category == "legacy-video-library":
            if not re.fullmatch(r"videos/[^/\\]+\.mp4", key) or "/.." in key:
                raise ValueError("Retention requires an original root library video")
        elif category == "legacy-unfiled-upload":
            parts = key.split("/")
            if len(parts) != 2 or parts[0] != "uploads" or str(uuid.UUID(parts[1])) != parts[1]:
                raise ValueError("Retention requires an original root upload UUID")
            meta = row.get("metadata")
            if (not isinstance(meta, dict) or meta.get("id") != parts[1]
                    or type(meta.get("size")) is not int or meta["size"] != row["size"]
                    or type(meta.get("uploadedAt")) is not int or meta["uploadedAt"] < 0
                    or any(not isinstance(meta.get(name), str) or not meta[name]
                           for name in ("filename", "contentType", "uploadedBy"))):
                raise ValueError("Original unfiled upload metadata proof required")
        else:
            raise ValueError("Unknown retained remote category")
        result[key] = row
    return result


def _global_remote_key(key):
    if key.startswith("videos/"):
        return True
    return key.startswith("uploads/") and len(key.split("/")) == 2 and not key.endswith(".json")


def build_inventory(
    inventory_id,
    *,
    remote_events,
    remote_readback,
    max_file_bytes,
    bindings=None,
    local_aliases=None,
    legacy_upload_records=(),
    retained_remote_objects=(),
):
    """Read every original row, local tree entry and complete remote object census.

    bindings maps an exact remote key or local absolute path to an explicit
    original {patientUuid,sourceKind,sourceId} or {patientUuid,fileId} binding.
    Metadata and identity bytes are retained raw, never used to rename a chart.
    """
    if type(max_file_bytes) is not int or max_file_bytes <= 0:
        raise ValueError("Explicit file byte limit required")
    legacy_upload_records = list(legacy_upload_records)
    retained_remote_objects = list(retained_remote_objects)
    retained_objects = _retained_remote_evidence(retained_remote_objects)
    bindings = bindings or {}
    local_aliases = local_aliases or []
    if any(key in bindings for key in retained_objects):
        raise CatalogueConflict("Retained global object also has a chart binding")
    root = _inventory_root(inventory_id)
    with upload_lock("inventory:" + inventory_id):
        manifest_path = root / "manifest.json"
        if manifest_path.exists():
            # A completed inventory is immutable, including its input bindings.
            prior = json.loads(manifest_path.read_bytes())
            if (
                prior["bindings"] != bindings
                or prior["aliases"] != local_aliases
                or prior["legacyUploads"] != legacy_upload_records
                or prior.get("retainedRemoteObjects", []) != retained_remote_objects
            ):
                raise CatalogueConflict("Inventory binding evidence changed")
            return prior
        root.mkdir(parents=True, exist_ok=True)
        rows_path = root / "rows.ndjson"
        pending_rows = root / (".rows-" + str(uuid.uuid4()))
        row_stream = pending_rows.open("wb")
        row_count = 0

        def record_row(row):
            nonlocal row_count
            row_stream.write(_json(row).encode() + b"\n")
            row_stream.flush()
            os.fsync(row_stream.fileno())
            row_count += 1

        errors = []
        legacy = []
        seen_paths = set()
        with storage.session_scope() as s:
            patients = list(s.scalars(select(storage.Patient)))
            patient_rows = [
                {
                    c.key: getattr(p, c.key).isoformat()
                    if hasattr(getattr(p, c.key), "isoformat")
                    else getattr(p, c.key)
                    for c in inspect(storage.Patient).columns
                }
                for p in patients
            ]
            legacy = [p.id for p in patients if not parse_canonical_patient_id(p.label)]
            for model, kind in (
                (storage.Report, "report"),
                (storage.PatientFile, "patient-file"),
            ):
                for source in s.scalars(select(model).order_by(model.id)):
                    path = _source_path(source.stored_path)
                    row = dict(
                        sourceRow={
                            c.key: getattr(source, c.key).isoformat()
                            if hasattr(getattr(source, c.key), "isoformat")
                            else getattr(source, c.key)
                            for c in inspect(model).columns
                        },
                        kind="engine-source",
                        sourceKind=kind,
                        sourceId=source.id,
                        patientUuid=source.patient_id,
                        storedPath=source.stored_path,
                        resolvedPath=str(path),
                        originalName=source.filename,
                        contentType=source.mime_type,
                        uploadedAt=_millis(source.created_at),
                    )
                    seen_paths.add(str(path.resolve()))
                    try:
                        with path.open("rb") as stream:
                            row["sha256"], row["size"] = _digest(
                                iter(lambda: stream.read(65536), b""), max_file_bytes
                            )
                        row["status"] = "available"
                    except FileNotFoundError:
                        row["status"] = (
                            "retained_missing"
                            if source.patient_id in legacy
                            else "missing"
                        )
                        row["reason"] = "missing_source_bytes"
                        if row["status"] == "missing":
                            errors.append(
                                dict(
                                    sourceId=source.id,
                                    reason="missing canonical source",
                                )
                            )
                    except (OSError, CatalogueUnavailable) as error:
                        row.update(status="error", reason=type(error).__name__)
                        errors.append(
                            dict(sourceId=source.id, reason=type(error).__name__)
                        )
                    record_row(row)
        from .portal_sync import portal_patients_dir

        portal = portal_patients_dir()
        if portal.exists():

            def walk_error(error):
                raise error

            try:
                for directory, dirs, files in os.walk(
                    portal, onerror=walk_error, followlinks=False
                ):
                    for name in dirs:
                        path = Path(directory) / name
                        if path.is_symlink():
                            errors.append(
                                dict(path=str(path), reason="symlink_directory")
                            )
                            record_row(
                                dict(
                                    kind="local-directory",
                                    path=str(path),
                                    status="error",
                                    reason="symlink_directory",
                                )
                            )
                    for name in files:
                        path = Path(directory) / name
                        if str(path.resolve()) in seen_paths:
                            continue
                        row = dict(
                            kind="local-file",
                            path=str(path),
                            binding=bindings.get(str(path)),
                        )
                        try:
                            with path.open("rb") as stream:
                                if (
                                    path.parent == portal
                                    and name in ROOT_OPERATIONAL_FILES
                                    and not path.is_symlink()
                                ):
                                    raw = stream.read(max_file_bytes + 1)
                                    row["sha256"], row["size"] = _digest(
                                        [raw], max_file_bytes
                                    )
                                    object_name = (
                                        "operational-"
                                        + hashlib.sha256(str(path).encode()).hexdigest()
                                        + ".bin"
                                    )
                                    _immutable(root / "objects" / object_name, raw)
                                    row.update(
                                        kind="root-operational",
                                        rawOperational=object_name,
                                    )
                                    stream.seek(0)
                                row["sha256"], row["size"] = _digest(
                                    iter(lambda: stream.read(65536), b""),
                                    max_file_bytes,
                                )
                            row["status"] = "available"
                        except (OSError, CatalogueUnavailable) as error:
                            row.update(status="error", reason=type(error).__name__)
                            errors.append(
                                dict(path=str(path), reason=type(error).__name__)
                            )
                        record_row(row)
            except OSError as error:
                errors.append(dict(source="local-tree", reason=type(error).__name__))
        retained_seen = set()
        try:
            for key in validate_remote_census(remote_events):
                row = dict(kind="remote-object", key=key, binding=bindings.get(key))
                if key in retained_objects:
                    retained_seen.add(key)
                    row["retainedEvidence"] = retained_objects[key]
                try:
                    if _global_remote_key(key) and key not in retained_objects:
                        raise CatalogueUnavailable("Original global object retention evidence missing")
                    if key.endswith(".json") and "/files/" not in key:
                        chunks = remote_readback(key, 8 * 1024 * 1024)
                        raw = bytearray()
                        for chunk in chunks:
                            if not isinstance(chunk, bytes):
                                raise ValueError("Actual object bytes required")
                            raw.extend(chunk)
                            if len(raw) > 8 * 1024 * 1024:
                                raise CatalogueUnavailable("Metadata exceeds limit")
                        json.loads(raw)
                        object_name = hashlib.sha256(key.encode()).hexdigest() + ".json"
                        _immutable(root / "objects" / object_name, bytes(raw))
                        row.update(
                            rawObject=object_name,
                            sha256=hashlib.sha256(raw).hexdigest(),
                            size=len(raw),
                        )
                    else:
                        row["sha256"], row["size"] = _digest(
                            remote_readback(key, max_file_bytes), max_file_bytes
                        )
                    if key in retained_objects:
                        proof = retained_objects[key]
                        if (row["sha256"], row["size"]) != (proof["sha256"], proof["size"]):
                            raise CatalogueUnavailable("Retained remote bytes differ from original evidence")
                    row["status"] = "available"
                except (OSError, ValueError, CatalogueUnavailable) as error:
                    row.update(status="error", reason=type(error).__name__)
                    errors.append(dict(key=key, reason=type(error).__name__))
                record_row(row)
        except Exception as error:
            errors.append(dict(source="remote-census", reason=type(error).__name__))
        for key in sorted(set(retained_objects) - retained_seen):
            errors.append(dict(key=key, reason="retained_object_missing_from_census"))
        row_stream.close()
        with pending_rows.open("rb") as stream:
            rows_hash, _ = _digest(iter(lambda: stream.read(65536), b""), 2**63 - 1)
        if rows_path.exists():
            with rows_path.open("rb") as stream:
                previous_hash, _ = _digest(
                    iter(lambda: stream.read(65536), b""), 2**63 - 1
                )
            if previous_hash != rows_hash:
                raise CatalogueConflict(
                    "Interrupted inventory rows differ; use a new inventory ID"
                )
        else:
            os.link(pending_rows, rows_path)
        pending_rows.unlink()
        manifest = dict(
            schemaVersion=1,
            inventoryId=inventory_id,
            legacyPatientIds=sorted(legacy),
            patients=patient_rows,
            rowsFile="rows.ndjson",
            rowsSha256=rows_hash,
            rowCount=row_count,
            bindings=bindings,
            aliases=local_aliases,
            legacyUploads=list(legacy_upload_records),
            retainedRemoteObjects=retained_remote_objects,
            errors=errors,
            complete=not errors,
        )
        # A failed attempt remains immutable evidence too. Retrying requires a
        # new explicit inventory ID, so a failed source never becomes empty.
        _immutable(manifest_path, _json(manifest).encode())
        return manifest


def _original_args(row):
    kind = row["sourceKind"]
    return dict(
        patient_uuid=row["patientUuid"],
        source_kind=kind,
        source_id=row["sourceId"],
        original_name=row["originalName"],
        logical_family=f"{kind}:{row['originalName']}",
        local_path=row["resolvedPath"],
        content_type=row["contentType"],
        document_kind="source-report" if kind == "report" else None,
        uploaded_at=row["uploadedAt"],
        provenance={kind + "Id": row["sourceId"]},
        **(
            {"sha256": row["sha256"], "size": row["size"]}
            if row["status"] == "available"
            else {}
        ),
    )


def retain_missing_source(inventory, row):
    if row["status"] != "retained_missing":
        raise ValueError("Expected retained missing source")
    evidence = dict(
        sourceKind=row["sourceKind"],
        sourceId=row["sourceId"],
        patientUuid=row["patientUuid"],
        storedPath=row["storedPath"],
        reason="missing_source_bytes",
        inventoryEvidenceId=inventory["inventoryId"],
    )
    with _write() as s:
        patient = s.get(storage.Patient, row["patientUuid"])
        model = storage.Report if row["sourceKind"] == "report" else storage.PatientFile
        source = s.get(model, row["sourceId"])
        if (
            patient is None
            or parse_canonical_patient_id(patient.label)
            or source is None
            or source.patient_id != patient.id
            or source.stored_path != row["storedPath"]
        ):
            raise CatalogueConflict("Retained source no longer matches legacy history")
        try:
            _source_path(source.stored_path).stat()
        except FileNotFoundError:
            pass
        else:
            raise CatalogueConflict("Retained source is now available")
        projection = s.scalar(
            select(ClinicProjection).where(
                ClinicProjection.source_kind == row["sourceKind"],
                ClinicProjection.source_id == row["sourceId"],
            )
        )
        if projection is None:
            projection = ClinicProjection(
                id=str(uuid.uuid4()),
                patient_uuid=patient.id,
                source_kind=row["sourceKind"],
                source_id=row["sourceId"],
                payload_json=_json(_original_args(row)),
                error="missing_source_bytes",
                artifact_id=None,
            )
            s.add(projection)
            _bump(s, patient.id)
        if projection.artifact_id is not None:
            raise CatalogueConflict("Retained missing source already has an artifact")
    return evidence


def validate_retained_sources(session, manifest):
    entries = manifest.get("retainedUnresolvedSources", [])
    if not isinstance(entries, list):
        raise ValueError("Invalid retained source evidence")
    result = set()
    for entry in entries:
        if (
            not isinstance(entry, dict)
            or set(entry)
            != {
                "sourceKind",
                "sourceId",
                "patientUuid",
                "storedPath",
                "reason",
                "inventoryEvidenceId",
            }
            or entry["reason"] != "missing_source_bytes"
        ):
            raise ValueError("Invalid retained source evidence")
        inventory = json.loads(
            (
                _inventory_root(entry["inventoryEvidenceId"]) / "manifest.json"
            ).read_bytes()
        )
        if (
            not inventory["complete"]
            or inventory["inventoryId"] != manifest["inventoryId"]
        ):
            raise CatalogueUnavailable("Retained evidence census is incomplete")
        matches = [
            r
            for r in inventory_rows(inventory)
            if r.get("kind") == "engine-source"
            and all(
                r.get(k) == entry[k]
                for k in ("sourceKind", "sourceId", "patientUuid", "storedPath")
            )
            and r.get("status") == "retained_missing"
        ]
        model = {"report": storage.Report, "patient-file": storage.PatientFile}.get(
            entry["sourceKind"]
        )
        source = session.get(model, entry["sourceId"]) if model else None
        patient = session.get(storage.Patient, entry["patientUuid"])
        projection = session.scalar(
            select(ClinicProjection).where(
                ClinicProjection.source_kind == entry["sourceKind"],
                ClinicProjection.source_id == entry["sourceId"],
            )
        )
        if (
            len(matches) != 1
            or source is None
            or patient is None
            or parse_canonical_patient_id(patient.label)
            or source.patient_id != patient.id
            or source.stored_path != entry["storedPath"]
            or projection is None
            or projection.patient_uuid != patient.id
            or projection.artifact_id is not None
        ):
            raise CatalogueConflict(
                "Retained evidence does not bind exact missing legacy row"
            )
        identity = (entry["sourceKind"], entry["sourceId"])
        if identity in result:
            raise ValueError("Duplicate retained source evidence")
        result.add(identity)
    return result


def _bound_artifact(binding):
    if not isinstance(binding, dict):
        raise CatalogueUnavailable("Original source ownership remains unresolved")
    with storage.session_scope() as s:
        if "fileId" in binding:
            artifact = s.get(ClinicArtifact, binding["fileId"])
        else:
            artifact = s.scalar(
                select(ClinicArtifact).where(
                    ClinicArtifact.source_kind == binding["sourceKind"],
                    ClinicArtifact.source_id == binding["sourceId"],
                )
            )
        if artifact is None or artifact.patient_uuid != binding["patientUuid"]:
            raise CatalogueConflict(
                "Original source ownership does not match catalogue"
            )
        return artifact


def _add_exact_location(row, remote_readback):
    from .clinic_catalogue import _read_local, _location

    binding = row["binding"]
    if isinstance(binding, dict) and binding.get("sourceKind") == "netlify-history":
        if (
            row["kind"] != "remote-object"
            or binding.get("sourceId") != row["key"]
            or not isinstance(binding.get("metadata"), dict)
        ):
            raise CatalogueConflict(
                "Remote-only source needs its exact original key and metadata evidence"
            )
        metadata = binding["metadata"]
        register_artifact(
            patient_uuid=binding["patientUuid"],
            source_kind="netlify-history",
            source_id=row["key"],
            original_name=metadata["originalName"],
            logical_family=metadata["logicalFamily"],
            sha256=row["sha256"],
            size=row["size"],
            content_type=metadata.get("contentType"),
            document_kind=metadata.get("documentKind"),
            session_date=metadata.get("sessionDate"),
            generated_at=metadata.get("generatedAt"),
            uploaded_at=metadata.get("uploadedAt"),
            uploaded_by=metadata.get("uploadedBy"),
            provenance=dict(originalRemoteKey=row["key"], originalMetadata=metadata),
        )
    artifact = _bound_artifact(binding)
    if (row["sha256"], row["size"]) != (artifact.sha256, artifact.size):
        raise CatalogueConflict("Inventory bytes do not match original artifact")
    if row["kind"] == "remote-object" and row["key"].startswith("uploads/pending/"):
        # Pending ingress is original upload evidence, never a patient replica.
        return artifact.id
    if row["kind"] == "remote-object":
        from .clinic_catalogue import _remote_key
        from .clinic_models import ClinicLocation

        alias = _remote_key(row["key"])
        with storage.session_scope() as session:
            occupied = session.scalar(
                select(ClinicLocation).where(
                    ClinicLocation.kind == "netlify",
                    ClinicLocation.key == row["key"],
                    ClinicLocation.artifact_id != artifact.id,
                    ClinicLocation.active.is_(True),
                )
            )
            if occupied is not None:
                raise CatalogueConflict(
                    "Original remote object has conflicting source ownership"
                )
        import_file_alias(
            dict(
                patientUuid=artifact.patient_uuid,
                fileId=artifact.id,
                patientAlias=alias,
                relativePath=row["key"].rsplit("/", 1)[-1],
                sha256=artifact.sha256,
                evidence=dict(
                    originalRemoteKey=row["key"], originalSourceBinding=binding
                ),
            )
        )
        add_remote_location(artifact.id, row["key"])
        verify_remote_location(
            artifact.id, row["key"], lambda: remote_readback(row["key"], artifact.size)
        )
    else:
        path, digest, size, fingerprint = _read_local(row["path"])
        if (digest, size) != (artifact.sha256, artifact.size):
            raise CatalogueConflict("Local inventory bytes changed")
        with _write() as s:
            a = s.get(ClinicArtifact, artifact.id)
            patient = s.get(storage.Patient, a.patient_uuid)
            affected = _location(s, a, "local", path, patient.label, True, fingerprint)
            if affected:
                _bump(s, affected)
    return artifact.id


def import_file_alias(evidence):
    """Import explicit original journal/path/hash evidence without merging charts."""
    from .clinic_catalogue import _location
    from .clinic_models import ClinicPatientAlias

    required = {
        "patientUuid",
        "fileId",
        "patientAlias",
        "relativePath",
        "sha256",
        "evidence",
    }
    if (
        not isinstance(evidence, dict)
        or set(evidence) != required
        or not isinstance(evidence["evidence"], dict)
        or not evidence["evidence"]
    ):
        raise ValueError("Original byte-attested alias evidence required")
    relative = evidence["relativePath"]
    alias = evidence["patientAlias"]
    if (
        not isinstance(relative, str)
        or Path(relative).is_absolute()
        or any(p in ("", "..", ".") for p in relative.split("/"))
        or not isinstance(alias, str)
        or not alias
        or any(c in alias for c in "/\\\0")
    ):
        raise ValueError("Invalid original relative alias")
    artifact = _bound_artifact(evidence)
    if artifact.sha256 != evidence["sha256"]:
        raise CatalogueConflict("Alias bytes differ from original artifact")
    # Immutable evidence survives even when both collision charts relabel.
    evidence_path = (
        Path(storage.DATA_DIR)
        / "clinic_alias_evidence"
        / hashlib.sha256(_json(evidence).encode()).hexdigest()
    )
    _immutable(evidence_path, _json(evidence).encode())
    with _write() as s:
        prior = s.get(ClinicPatientAlias, alias)
        affected = set()
        if prior is None:
            prior = ClinicPatientAlias(
                alias=alias, patient_uuid=artifact.patient_uuid, ambiguous=False
            )
            s.add(prior)
            affected.add(artifact.patient_uuid)
        occupants = set(
            s.scalars(select(storage.Patient.id).where(storage.Patient.label == alias))
        ) | {prior.patient_uuid, artifact.patient_uuid}
        if len(occupants) > 1 and not prior.ambiguous:
            prior.ambiguous = True
            affected.update(occupants)
        a = s.get(ClinicArtifact, artifact.id)
        affected.update(
            _location(s, a, "legacy-reference", relative, alias, False, artifact.sha256)
        )
        if affected:
            _bump(s, affected)
    return artifact.id


def import_inventory(inventory_id, *, remote_readback, activate=False):
    """Explicit per-unit replay. A progress file cannot stand in for DB receipts."""
    root = _inventory_root(inventory_id)
    with upload_lock("inventory-import:" + inventory_id):
        inventory = json.loads((root / "manifest.json").read_bytes())
        if inventory["inventoryId"] != inventory_id or not inventory["complete"]:
            raise CatalogueUnavailable("Source census is incomplete")
        outcomes = []
        retained = []
        errors = []

        retained_objects = _retained_remote_evidence(inventory.get("retainedRemoteObjects", []))
        retained_seen = set()
        journal = root / ("progress-" + str(uuid.uuid4()) + ".ndjson")
        with journal.open("xb") as stream:
            stream.write((_json(dict(type="start", inventoryId=inventory_id,
                                    rowsSha256=inventory["rowsSha256"])) + "\n").encode())
            stream.flush()
            os.fsync(stream.fileno())
        sequence = 0
        counts = (0, 0, 0)

        def checkpoint():
            payload = _json(dict(inventoryId=inventory_id, outcomes=outcomes,
                                retainedUnresolvedSources=retained, errors=errors,
                                journalFile=journal.name, journalSequence=sequence))
            path = root / "progress.json"
            tmp = root / (".progress-" + str(uuid.uuid4()))
            with tmp.open("w") as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(tmp, path)
            fd = os.open(root, os.O_RDONLY)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)

        def save():
            nonlocal sequence, counts
            sequence += 1
            delta = dict(outcomes=outcomes[counts[0]:], retainedUnresolvedSources=retained[counts[1]:],
                         errors=errors[counts[2]:])
            with journal.open("ab") as stream:
                stream.write((_json(dict(inventoryId=inventory_id, sequence=sequence, delta=delta)) + "\n").encode())
                stream.flush()
                os.fsync(stream.fileno())
            counts = (len(outcomes), len(retained), len(errors))
            # Geometric checkpoints bound total full-snapshot bytes to O(units).
            if sequence & (sequence - 1) == 0:
                checkpoint()

        checkpoint()

        # Original engine source IDs first, so proven copies attach to them.
        for index, row in enumerate(inventory_rows(inventory)):
            try:
                if row["status"] == "retained_missing":
                    retained.append(retain_missing_source(inventory, row))
                    result = "retained_missing"
                elif row["status"] != "available":
                    raise CatalogueUnavailable("Original inventory source unavailable")
                elif row.get("retainedEvidence"):
                    proof = retained_objects.get(row.get("key"))
                    if proof != row["retainedEvidence"] or row.get("binding") or row["kind"] != "remote-object":
                        raise CatalogueConflict("Retained remote inventory evidence changed")
                    if _digest(remote_readback(row["key"], row["size"]), row["size"]) != (proof["sha256"], proof["size"]):
                        raise CatalogueConflict("Retained remote original bytes changed")
                    retained_seen.add(row["key"])
                    result = "retained_remote"
                elif row["kind"] == "root-operational":
                    from .portal_sync import portal_patients_dir

                    path = Path(row["path"])
                    if (
                        path.parent != portal_patients_dir()
                        or path.name not in ROOT_OPERATIONAL_FILES
                        or path.is_symlink()
                    ):
                        raise CatalogueConflict("Original operational path changed")
                    original = (root / "objects" / row["rawOperational"]).read_bytes()
                    if _digest([original], row["size"]) != (row["sha256"], row["size"]):
                        raise CatalogueConflict("Original operational evidence changed")
                    with path.open("rb") as stream:
                        observed = _digest(
                            iter(lambda: stream.read(65536), b""), row["size"]
                        )
                    if observed != (row["sha256"], row["size"]):
                        raise CatalogueConflict("Original operational bytes changed")
                    result = "retained_operational"
                elif row["kind"] == "engine-source":
                    with storage.session_scope() as s:
                        existing = s.scalar(
                            select(ClinicArtifact).where(
                                ClinicArtifact.source_kind == row["sourceKind"],
                                ClinicArtifact.source_id == row["sourceId"],
                            )
                        )
                    if existing:
                        if (existing.patient_uuid, existing.sha256, existing.size) != (
                            row["patientUuid"],
                            row["sha256"],
                            row["size"],
                        ):
                            raise CatalogueConflict(
                                "Original source changed since inventory"
                            )
                        result = existing.id
                    else:
                        result = register_artifact(**_original_args(row))["fileId"]
                elif row.get("rawObject"):
                    # Raw historic metadata remains immutable inventory evidence.
                    # Typed upload/feedback import follows after file bindings.
                    result = "retained_metadata"
                elif row.get("binding"):
                    result = _add_exact_location(row, remote_readback)
                else:
                    raise CatalogueUnavailable(
                        "Original file ownership/source binding remains unresolved"
                    )
                outcomes.append(
                    dict(
                        row=index,
                        status="retained"
                        if result.startswith("retained_")
                        else "registered",
                        result=result,
                    )
                )
            except Exception as error:
                detail = dict(
                    row=index,
                    status="unresolved",
                    reason=type(error).__name__,
                    detail=str(error),
                )
                outcomes.append(detail)
                errors.append(detail)
            save()
        for index, evidence in enumerate(inventory["aliases"]):
            try:
                result = import_file_alias(evidence)
                outcomes.append(dict(alias=index, status="registered", fileId=result))
            except Exception as error:
                errors.append(
                    dict(alias=index, reason=type(error).__name__, detail=str(error))
                )
            save()
        from .clinic_upload_import import (
            import_legacy_record,
            import_submission_evidence,
        )

        for record in inventory["legacyUploads"]:
            try:
                import_legacy_record(record)
            except Exception as error:
                errors.append(
                    dict(uploadId=record.get("uploadId"), reason=type(error).__name__)
                )
            save()
        metadata = {}
        metadata_hashes = {}
        for r in inventory_rows(inventory):
            if r.get("rawObject"):
                raw = (root / "objects" / r["rawObject"]).read_bytes()
                if hashlib.sha256(raw).hexdigest() != r["sha256"]:
                    raise CatalogueConflict("Original metadata evidence changed")
                metadata[r["key"]] = json.loads(raw)
                metadata_hashes[r["key"]] = r["sha256"]
        for key, record in metadata.items():
            try:
                if key.startswith("uploads/submissions/"):
                    if not isinstance(record, dict):
                        raise ValueError("Invalid submission evidence")
                    markers = [
                        v
                        for v in metadata.values()
                        if isinstance(v, dict)
                        and v.get("kind") == "new_patient_upload"
                        and v.get("uploadId") == record.get("submissionId")
                    ]
                    if len(markers) > 1 and any(m != markers[0] for m in markers[1:]):
                        raise CatalogueConflict("Original pending marker is ambiguous")
                    marker = markers[0] if markers else None
                    import_submission_evidence(record, marker=marker)
                    original = _original_submission_binding(
                        record, inventory, remote_readback
                    )
                    if original is not None:
                        files, registered = original
                        import_submission_evidence(
                            record, marker=marker, files=files, registered=registered
                        )
                elif (
                    isinstance(record, dict)
                    and record.get("kind") == "new_patient_upload"
                ):
                    import_legacy_record(dict(record, status="pending"))
                elif "/feedback/" in key:
                    _import_feedback_object(
                        key, record, inventory, metadata_hashes[key]
                    )
            except Exception as error:
                errors.append(
                    dict(key=key, reason=type(error).__name__, detail=str(error))
                )
            save()
        if retained_seen != set(retained_objects):
            errors.append(dict(reason="retained_remote_evidence_not_closed"))
            save()
        checkpoint()
        if errors:
            raise CatalogueUnavailable(
                f"Inventory retains {len(errors)} unresolved units"
            )
        manifest = dict(
            inventoryId=inventory_id,
            legacyPatientIds=inventory["legacyPatientIds"],
            retainedUnresolvedSources=retained,
        )
        if activate:
            from .clinic_catalogue import complete_catalogue_import

            complete_catalogue_import(manifest)
        return dict(**manifest, outcomes=outcomes, errors=errors, activated=activate)


def _import_feedback_object(key, record, inventory, original_hash):
    from .clinic_feedback import record_feedback

    if not isinstance(record, dict) or not {
        "action",
        "submittedBy",
        "submittedAt",
    } <= set(record):
        raise ValueError("Invalid original feedback object")
    binding = inventory["bindings"].get(key)
    artifact = _bound_artifact(binding)
    with storage.session_scope() as s:
        patient = s.get(storage.Patient, artifact.patient_uuid)
    if not parse_canonical_patient_id(patient.label):
        raise CatalogueUnavailable("Legacy feedback chart remains unresolved")
    original_id = (
        record.get("eventId")
        or "legacy-feedback:"
        + hashlib.sha256((key + "\n" + original_hash).encode()).hexdigest()
    )
    record_feedback(
        key=original_id,
        patient_id=patient.label,
        file_id=artifact.id,
        version=artifact.version,
        action=record["action"],
        notes=record.get("notes") or "",
        actor=record["submittedBy"],
        created_at=record["submittedAt"],
    )
    if "/.archive/feedback/" in key:
        # Original archived object is an import projection, not a new feedback
        # event. Apply once; later staff unarchive actions survive import replay.
        with _write() as session:
            source_id = key + ":" + original_hash
            prior = session.scalar(
                select(ClinicProjection).where(
                    ClinicProjection.source_kind == "legacy-archive",
                    ClinicProjection.source_id == source_id,
                )
            )
            if prior is None:
                session.add(
                    ClinicProjection(
                        id=str(uuid.uuid4()),
                        patient_uuid=artifact.patient_uuid,
                        source_kind="legacy-archive",
                        source_id=source_id,
                        payload_json=_json(dict(originalKey=key, sha256=original_hash)),
                        artifact_id=artifact.id,
                    )
                )
                session.get(ClinicArtifact, artifact.id).archived = True
                _bump(session, artifact.patient_uuid)
            elif prior.artifact_id != artifact.id:
                raise CatalogueConflict("Original archive ownership changed")


def inventory_rows(inventory):
    path = _inventory_root(inventory["inventoryId"]) / "rows.ndjson"
    with path.open("rb") as stream:
        digest, _ = _digest(iter(lambda: stream.read(65536), b""), 2**63 - 1)
        if digest != inventory["rowsSha256"]:
            raise CatalogueConflict("Original inventory rows changed")
        stream.seek(0)
        count = 0
        for line in stream:
            count += 1
            yield json.loads(line)
        if count != inventory["rowCount"]:
            raise CatalogueConflict("Original inventory row census changed")


def remote_inventory_events():
    """Fixed root-coordinated SDK helper; its successful exit precedes completion."""
    import shutil
    from .portal_sync import portal_sync_repo
    from .clinic_publication import _helper_bytes

    root = portal_sync_repo().resolve()
    node = shutil.which("node")
    helper = root / "scripts/qeeg_clinic_blob_inventory.mjs"
    if not node or not helper.is_file():
        raise CatalogueUnavailable("Trusted census helper unavailable")
    pending = bytearray()
    terminal = None
    for chunk in _helper_bytes(
        [node, str(helper)],
        cwd=root,
        key="",
        size=2**63 - 1,
        request_payload={"schemaVersion": 1},
    ):
        pending.extend(chunk)
        while b"\n" in pending:
            raw, _, remaining = pending.partition(b"\n")
            pending = bytearray(remaining)
            if len(raw) > 1048576:
                raise CatalogueUnavailable("Census line exceeds limit")
            event = json.loads(raw)
            if terminal is not None:
                raise CatalogueUnavailable("Census emitted after completion")
            if event.get("type") == "complete":
                terminal = event
            else:
                yield event
        if len(pending) > 1048576:
            raise CatalogueUnavailable("Census line exceeds limit")
    if pending or terminal is None:
        raise CatalogueUnavailable("Incomplete census helper response")
    yield terminal


def _original_submission_binding(receipt, inventory, remote_readback):
    """Original ordered response keys must explicitly bind original source rows."""
    items = receipt["manifest"]["files"]
    stored = receipt["response"]["uploaded"]
    bindings = [inventory["bindings"].get(item["fileKey"]) for item in stored]
    if any(binding is None for binding in bindings):
        return None
    artifacts = [_bound_artifact(binding) for binding in bindings]
    if (
        any(a.source_kind not in ("report", "patient-file") for a in artifacts)
        or len({a.patient_uuid for a in artifacts}) != 1
    ):
        raise CatalogueConflict("Original upload source ownership differs")
    files = []
    for artifact, item, original in zip(artifacts, items, stored):
        if (artifact.sha256, artifact.size) != (item["sha256"], item["size"]):
            raise CatalogueConflict("Original submission source bytes differ")
        raw = bytearray()
        for chunk in remote_readback(original["fileKey"], item["size"]):
            if not isinstance(chunk, bytes):
                raise CatalogueUnavailable("Actual original upload bytes required")
            raw.extend(chunk)
            if len(raw) > item["size"]:
                raise CatalogueConflict("Original submission byte size differs")
        if (
            len(raw) != item["size"]
            or hashlib.sha256(raw).hexdigest() != item["sha256"]
        ):
            raise CatalogueConflict("Original submission bytes differ")
        files.append((item["originalName"], bytes(raw), item["contentType"]))
    with storage.session_scope() as session:
        patient = session.get(storage.Patient, artifacts[0].patient_uuid)
        if not parse_canonical_patient_id(patient.label):
            raise CatalogueUnavailable("Original upload chart remains noncanonical")
        registered = dict(
            patientId=patient.label, sourceIds=[a.source_id for a in artifacts]
        )
    return files, registered
