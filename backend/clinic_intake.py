"""Durable free intake. Exact staging bytes and original source receipts survive retries."""

from __future__ import annotations
from contextlib import contextmanager
from datetime import datetime
import fcntl
import hashlib
import io
import json
import os
from pathlib import Path
import re
import tempfile
import uuid
from sqlalchemy import select
from . import storage, reports, patient_files
from .clinic_catalogue import _write, _bump, _now
from .clinic_catalogue_reads import _patient, _envelope, _json
from .clinic_models import (
    ClinicArtifact,
    CatalogueConflict,
    CatalogueNotFound,
    CatalogueUnavailable,
)
from .clinic_records import (
    ClinicUpload,
    ClinicUploadItem,
    ClinicMutation,
    ClinicLegacyUpload,
)
from .patient_intake import (
    IdentityInput,
    find_patient_by_identity,
    identity_key,
    IdentityNameConflict,
)
from .patient_identity import allocate_canonical_patient_id, normalize_birthdate


def require_key(value):
    if not isinstance(value, str) or not re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9._:-]{0,127}", value
    ):
        raise ValueError("A stable Idempotency-Key is required")
    return value


@contextmanager
def upload_lock(key):
    root = Path(storage.DATA_DIR) / "clinic_intake" / "locks"
    root.mkdir(parents=True, exist_ok=True)
    with (root / (hashlib.sha256(key.encode()).hexdigest() + ".lock")).open(
        "a+b"
    ) as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def _immutable(path, data):
    missing = []
    parent = path.parent
    while not parent.exists():
        missing.append(parent)
        parent = parent.parent
    for directory in reversed(missing):
        directory.mkdir(exist_ok=True)
        with _directory_fd(directory.parent) as fd:
            os.fsync(fd)
    if path.exists():
        if path.read_bytes() != data:
            raise CatalogueConflict("Immutable upload bytes changed")
        return
    fd, name = tempfile.mkstemp(prefix=".staging-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(name, path)
        except FileExistsError:
            if path.read_bytes() != data:
                raise CatalogueConflict("Immutable upload bytes changed")
        with _directory_fd(path.parent) as directory:
            os.fsync(directory)
    finally:
        os.unlink(name)


@contextmanager
def _directory_fd(path):
    fd = os.open(path, os.O_RDONLY)
    try:
        yield fd
    finally:
        os.close(fd)


def _identity(value, resolution=None):
    if not isinstance(value, dict) or set(value) - {
        "firstName",
        "lastName",
        "firstInitial",
        "lastInitial",
        "birthdate",
    }:
        raise ValueError("Invalid identity fields")
    if any(v is not None and not isinstance(v, str) for v in value.values()):
        raise ValueError("Identity fields must be text")
    r = resolution or {}
    return IdentityInput(
        first_name=value.get("firstName"),
        last_name=value.get("lastName"),
        first_initial=value.get("firstInitial"),
        last_initial=value.get("lastInitial"),
        birthdate=value.get("birthdate"),
        attach_to=r.get("attachTo"),
        force_new=r.get("forceNew", False),
    )


def _resolution(value):
    if value is None or value == {}:
        return {}
    if not isinstance(value, dict):
        raise ValueError("Invalid resolution")
    if set(value) == {"forceNew"} and value["forceNew"] is True:
        return value
    if (
        set(value) == {"attachTo"}
        and isinstance(value["attachTo"], str)
        and value["attachTo"]
    ):
        return value
    raise ValueError("Choose attachTo or forceNew")


def _manifest(
    *, identity, files, file_meta, actor, patient_id, resolution, analysis_intent
):
    if not isinstance(files, list) or not 1 <= len(files) <= 100:
        raise ValueError("Supply 1 to 100 files")
    if not isinstance(file_meta, list) or len(file_meta) != len(files):
        raise ValueError("Each ordered file needs fileMeta")
    resolution = _resolution(resolution)
    parsed = _identity(identity, resolution)
    if (not patient_id or parsed.force_new) and not parsed.attach_to:
        identity_key(parsed)
    items = []
    for index, ((name, data, mime), meta) in enumerate(zip(files, file_meta)):
        if (
            not isinstance(name, str)
            or not name
            or len(name.encode()) > 1024
            or any(not c.isprintable() for c in name)
        ):
            raise ValueError("Invalid original filename")
        if not isinstance(data, bytes) or not data:
            raise ValueError("Upload bytes are required")
        if not isinstance(mime, str) or not mime:
            raise ValueError("Content type required")
        if not isinstance(meta, dict):
            raise ValueError("Invalid fileMeta")
        if meta.get("sessionDate"):
            if not isinstance(meta["sessionDate"], str):
                raise ValueError("Invalid sessionDate")
            if (
                datetime.strptime(meta["sessionDate"], "%Y-%m-%d").strftime("%Y-%m-%d")
                != meta["sessionDate"]
            ):
                raise ValueError("Invalid sessionDate")
        if meta.get("reportBirthdate"):
            dob = normalize_birthdate(meta["reportBirthdate"])
            if identity.get("birthdate") and dob != normalize_birthdate(
                identity["birthdate"]
            ):
                raise CatalogueConflict("Report birthdate differs from upload identity")
        items.append(
            dict(
                itemIndex=index,
                originalName=name,
                sha256=hashlib.sha256(data).hexdigest(),
                size=len(data),
                contentType=mime,
                metadata=meta,
            )
        )
    if analysis_intent is not None:
        a = analysis_intent
        required = {
            "operationId",
            "confirmed",
            "reportItemIndexes",
            "specialInstructions",
        }
        if not isinstance(a, dict) or set(a) not in (
            required,
            required | {"expectedPolicyFingerprint"},
        ):
            raise ValueError("Invalid confirmed analysis intent")
        if "expectedPolicyFingerprint" in a and (
            not isinstance(a["expectedPolicyFingerprint"], str)
            or not re.fullmatch(r"[0-9a-f]{64}", a["expectedPolicyFingerprint"])
        ):
            raise ValueError("Invalid expected analysis policy fingerprint")
        require_key(a["operationId"])
        indexes = a["reportItemIndexes"]
        if (
            a["confirmed"] is not True
            or not isinstance(a["specialInstructions"], str)
            or len(a["specialInstructions"]) > 100000
            or not isinstance(indexes, list)
            or not indexes
            or any(type(i) is not int or i < 0 or i >= len(items) for i in indexes)
            or len(set(indexes)) != len(indexes)
            or any(
                items[i]["metadata"].get("documentKind") != "report" for i in indexes
            )
        ):
            raise ValueError("Confirm exact ordered report items")
    return dict(
        identity=identity,
        patientId=patient_id,
        resolution=resolution,
        uploadedBy=actor,
        items=items,
        analysisIntent=analysis_intent,
    )


def submit_upload(
    *,
    key,
    identity,
    files,
    file_meta,
    actor=None,
    patient_id=None,
    resolution=None,
    analysis_intent=None,
    upload_id=None,
    uploaded_at=None,
    registered=None,
    principal=None,
):
    key = require_key(key)
    if principal not in (None, "workbench", "thrylen-service"):
        raise ValueError("Invalid trusted principal")
    manifest = _manifest(
        identity=identity,
        files=files,
        file_meta=file_meta,
        actor=actor,
        patient_id=patient_id,
        resolution=resolution,
        analysis_intent=analysis_intent,
    )
    with upload_lock(key):
        with storage.session_scope() as s:
            existing = s.scalar(
                select(ClinicUpload).where(ClinicUpload.admission_key == key)
            )
            if existing and existing.manifest_json != _json(manifest):
                raise CatalogueConflict(
                    "Submission key already binds different material"
                )
            if existing and registered:
                saved_ids = list(
                    s.scalars(
                        select(ClinicUploadItem.source_id)
                        .where(ClinicUploadItem.upload_id == existing.id)
                        .order_by(ClinicUploadItem.position)
                    )
                )
                if (
                    existing.patient_uuid != _patient(s, registered["patientId"]).id
                    or saved_ids != registered["sourceIds"]
                ):
                    raise CatalogueConflict("Original registered binding changed")
            chosen = existing.id if existing else (upload_id or key)
        require_key(chosen)
        root = (
            Path(storage.DATA_DIR)
            / "clinic_intake"
            / "submissions"
            / hashlib.sha256(key.encode()).hexdigest()
        )
        policy = None
        policy_binding = None
        if (
            analysis_intent
            and not existing
            and "expectedPolicyFingerprint" in analysis_intent
        ):
            from .clinic_analysis_intents import prepare_policy_binding

            # Verify exactly what was shown before persisting upload material or intent.
            policy, policy_binding = prepare_policy_binding(
                root, expected_fingerprint=analysis_intent["expectedPolicyFingerprint"]
            )
        for item, (_, data, _) in zip(manifest["items"], files):
            _immutable(root / (str(item["itemIndex"]) + ".bytes"), data)
        _immutable(root / "manifest.json", _json(manifest).encode())
        if analysis_intent:
            from .clinic_analysis_intents import (
                prepare_policy_binding,
                read_policy_binding,
            )

            if existing:
                policy_binding = json.loads(existing.analysis_json)["policyBinding"]
                snapshot, _ = read_policy_binding(policy_binding)
                policy = snapshot["publicPolicy"]
            elif policy_binding is None:
                policy, policy_binding = prepare_policy_binding(root)
        with _write() as s:
            existing = s.get(ClinicUpload, chosen)
            if existing is None:
                analysis = None
                if analysis_intent:
                    from .clinic_records import ClinicOperation

                    op = analysis_intent["operationId"]
                    if (
                        s.get(ClinicOperation, op)
                        or s.get(storage.AnalysisInputReservation, op)
                        or s.scalar(
                            select(storage.Run.id).where(storage.Run.operation_id == op)
                        )
                    ):
                        raise CatalogueConflict(
                            "Analysis operation already has an original owner"
                        )
                    analysis = {
                        **analysis_intent,
                        "policy": policy,
                        "policyBinding": policy_binding,
                    }
                    if any(
                        json.loads(u.analysis_json)["operationId"]
                        == analysis["operationId"]
                        for u in s.scalars(
                            select(ClinicUpload).where(
                                ClinicUpload.analysis_json.is_not(None)
                            )
                        )
                    ):
                        raise CatalogueConflict(
                            "Analysis operation already belongs to another upload"
                        )
                adopted_patient = (
                    _patient(s, registered["patientId"]) if registered else None
                )
                if registered and (
                    set(registered) != {"patientId", "sourceIds"}
                    or len(registered["sourceIds"]) != len(manifest["items"])
                    or len(set(registered["sourceIds"])) != len(manifest["items"])
                ):
                    raise ValueError("Exact original ordered source bindings required")
                s.add(
                    ClinicUpload(
                        id=chosen,
                        admission_key=key,
                        patient_uuid=adopted_patient.id if adopted_patient else None,
                        manifest_json=_json(manifest),
                        uploaded_at=uploaded_at if uploaded_at is not None else _now(),
                        uploaded_by=actor,
                        uploaded_principal=principal,
                        analysis_json=_json(analysis) if analysis else None,
                    )
                )
                for item in manifest["items"]:
                    sid = (
                        registered["sourceIds"][item["itemIndex"]]
                        if registered
                        else str(uuid.uuid4())
                    )
                    source_kind = (
                        "report"
                        if item["metadata"].get("documentKind") == "report"
                        else "patient-file"
                    )
                    artifact = None
                    if registered:
                        source = s.get(
                            storage.Report
                            if source_kind == "report"
                            else storage.PatientFile,
                            sid,
                        )
                        artifact = s.scalar(
                            select(ClinicArtifact).where(
                                ClinicArtifact.source_id == sid,
                                ClinicArtifact.source_kind == source_kind,
                            )
                        )
                        if (
                            not source
                            or source.patient_id != adopted_patient.id
                            or not artifact
                            or artifact.sha256 != item["sha256"]
                            or artifact.size != item["size"]
                            or hashlib.sha256(
                                Path(source.stored_path).read_bytes()
                            ).hexdigest()
                            != item["sha256"]
                        ):
                            raise CatalogueConflict(
                                "Original source binding differs or is missing"
                            )
                    s.add(
                        ClinicUploadItem(
                            id=chosen + ":" + str(item["itemIndex"]),
                            upload_id=chosen,
                            position=item["itemIndex"],
                            metadata_json=_json(item),
                            staging_path=str(
                                root / (str(item["itemIndex"]) + ".bytes")
                            ),
                            source_id=sid,
                            source_kind=source_kind,
                            status="registered" if registered else "pending",
                            artifact_id=artifact.id if artifact else None,
                        )
                    )
                _bump(s)
        return _resume_locked(chosen)


def _bind_patient(upload_id):
    with _write() as s:
        u = s.get(ClinicUpload, upload_id)
        if u is None:
            raise CatalogueNotFound("Upload not found")
        if u.patient_uuid:
            return u.patient_uuid
        m = json.loads(u.manifest_json)
        answer = json.loads(u.resolution_json) if u.resolution_json else m["resolution"]
        identity = _identity(m["identity"], answer)
        try:
            if answer.get("attachTo"):
                exact = _patient(s, answer["attachTo"])
                identity = _identity(m["identity"], {"attachTo": exact.label})
                patient, keep = find_patient_by_identity(s, identity)
            elif m["patientId"] and not answer.get("forceNew"):
                from .patient_identity import parse_canonical_patient_id
                from .patient_intake import stored_full_name

                target = _patient(s, m["patientId"])
                parsed = parse_canonical_patient_id(target.label)
                merged = dict(
                    firstName=target.first_name,
                    lastName=target.last_name,
                    firstInitial=target.first_initial or parsed.first_initial,
                    lastInitial=target.last_initial or parsed.last_initial,
                    birthdate=target.birthdate or parsed.birthdate,
                )
                report_dobs = {
                    i["metadata"].get("reportBirthdate")
                    for i in m["items"]
                    if i["metadata"].get("reportBirthdate")
                }
                if len(report_dobs) > 1:
                    raise ValueError("Report items have different dates of birth")
                if report_dobs and "birthdate" not in m["identity"]:
                    merged["birthdate"] = next(iter(report_dobs))
                merged.update(m["identity"])
                if "firstName" in m["identity"] and "firstInitial" not in m["identity"]:
                    merged.pop("firstInitial", None)
                if "lastName" in m["identity"] and "lastInitial" not in m["identity"]:
                    merged.pop("lastInitial", None)
                identity = _identity(merged)
                patient, keep = find_patient_by_identity(
                    s, identity, target_patient=target
                )
                if patient is None or patient.id != target.id:
                    raise IdentityNameConflict(
                        dict(
                            conflict="identity_name_mismatch",
                            incoming_name=" ".join(
                                filter(None, [identity.first_name, identity.last_name])
                            ),
                            candidates=[
                                dict(
                                    patient_id=target.label,
                                    name=stored_full_name(target),
                                )
                            ],
                            detail="The supplied identity differs from this chart. Choose the same person or someone different.",
                        )
                    )
                if not m["identity"]:
                    keep = True
            else:
                patient, keep = find_patient_by_identity(s, identity)
        except IdentityNameConflict as error:
            u.status = "needs_operator_answer"
            u.conflict_json = _json(error.payload)
            _bump(s)
            return None
        if patient:
            _patient(s, patient.label)
        if patient is None or not keep:
            first, last, dob = identity_key(identity)
            canonical = allocate_canonical_patient_id(
                s,
                first_initial=first,
                last_initial=last,
                birthdate=dob,
                exclude_patient_uuid=patient.id if patient else None,
                commit=False,
            )
            fields = dict(
                label=canonical,
                birthdate=dob,
                first_initial=first,
                last_initial=last,
                first_name=identity.first_name,
                last_name=identity.last_name,
                commit=False,
            )
            patient = (
                storage.update_patient(s, patient.id, **fields)
                if patient
                else storage.create_patient(s, **fields)
            )
        u.patient_uuid = patient.id
        u.conflict_json = None
        u.status = "pending"
        _bump(s, patient.id)
        return patient.id


def _file_item(item_id, patient_uuid):
    with storage.session_scope() as s:
        item = s.get(ClinicUploadItem, item_id)
        if item.status == "registered":
            return
        meta = json.loads(item.metadata_json)
        source_id = item.source_id
        kind = item.source_kind
        data = Path(item.staging_path).read_bytes()
    if hashlib.sha256(data).hexdigest() != meta["sha256"] or len(data) != meta["size"]:
        raise CatalogueUnavailable("Staged upload byte binding changed")
    if kind == "report":
        path, extracted, mime, _ = reports.save_report_upload(
            patient_id=patient_uuid,
            report_id=source_id,
            filename=meta["originalName"],
            provided_mime_type=meta["contentType"],
            file_bytes=data,
        )
    else:
        path, mime, size = patient_files.save_patient_file_upload(
            patient_id=patient_uuid,
            file_id=source_id,
            filename=meta["originalName"],
            provided_mime_type=meta["contentType"],
            src=io.BytesIO(data),
        )
    # fsync all extraction/source bytes before the authoritative source receipt.
    for output in path.parent.rglob("*"):
        if output.is_file():
            with output.open("rb") as stream:
                os.fsync(stream.fileno())
    with _directory_fd(path.parent) as directory:
        os.fsync(directory)
    with _write() as s:
        item = s.get(ClinicUploadItem, item_id)
        u = s.get(ClinicUpload, item.upload_id)
        s.info["clinic_source_metadata"] = {
            source_id: dict(
                uploaded_at=u.uploaded_at,
                uploaded_by=u.uploaded_by,
                session_date=meta["metadata"].get("sessionDate") or None,
                provenance={
                    kind + "Id": source_id,
                    "uploadId": u.id,
                    "itemIndex": item.position,
                    "metadata": meta["metadata"],
                },
            )
        }
        if kind == "report":
            storage.create_report(
                s,
                report_id=source_id,
                patient_id=patient_uuid,
                filename=meta["originalName"],
                mime_type=mime,
                stored_path=path,
                extracted_text_path=extracted,
                commit=False,
            )
        else:
            storage.create_patient_file(
                s,
                file_id=source_id,
                patient_id=patient_uuid,
                filename=meta["originalName"],
                mime_type=mime,
                stored_path=path,
                size_bytes=size,
                commit=False,
            )
        artifact = s.scalar(
            select(ClinicArtifact).where(
                ClinicArtifact.source_kind == kind,
                ClinicArtifact.source_id == source_id,
            )
        )
        if artifact is None:
            raise CatalogueUnavailable("Source catalogue registration incomplete")
        item.artifact_id = artifact.id
        item.status = "registered"
        item.error = None


def _resume_locked(upload_id):
    patient_uuid = _bind_patient(upload_id)
    if patient_uuid:
        with storage.session_scope() as s:
            ids = list(
                s.scalars(
                    select(ClinicUploadItem.id)
                    .where(ClinicUploadItem.upload_id == upload_id)
                    .order_by(ClinicUploadItem.position)
                )
            )
        for item_id in ids:
            try:
                _file_item(item_id, patient_uuid)
            except Exception as error:
                with _write() as s:
                    item = s.get(ClinicUploadItem, item_id)
                    item.status = "failed"
                    item.error = type(error).__name__ + ": Free filing needs retry"
                    _bump(s, patient_uuid)
        with _write() as s:
            u = s.get(ClinicUpload, upload_id)
            statuses = list(
                s.scalars(
                    select(ClinicUploadItem.status).where(
                        ClinicUploadItem.upload_id == upload_id
                    )
                )
            )
            status = (
                "registered" if all(x == "registered" for x in statuses) else "failed"
            )
            if u.status != status:
                u.status = status
                _bump(s, patient_uuid)
    return get_upload(upload_id)


def resume_upload(upload_id):
    with storage.session_scope() as s:
        u = s.get(ClinicUpload, upload_id)
        if not u:
            raise CatalogueNotFound("Upload not found")
        key = u.admission_key
    with upload_lock(key):
        return _resume_locked(upload_id)


def _upload_json(s, u):
    m = json.loads(u.manifest_json)
    items = [
        dict(
            **json.loads(i.metadata_json),
            itemId=i.id,
            status=i.status,
            sourceId=i.source_id,
            fileId=i.artifact_id,
            error=i.error,
        )
        for i in s.scalars(
            select(ClinicUploadItem)
            .where(ClinicUploadItem.upload_id == u.id)
            .order_by(ClinicUploadItem.position)
        )
    ]
    patient = s.get(storage.Patient, u.patient_uuid) if u.patient_uuid else None
    analysis = json.loads(u.analysis_json) if u.analysis_json else None
    if analysis:
        from .clinic_analysis_intents import read_policy_binding

        _, compatible = read_policy_binding(analysis.pop("policyBinding"))
        selected = [items[i] for i in analysis["reportItemIndexes"]]
        run = s.scalar(
            select(storage.Run).where(
                storage.Run.operation_id == analysis["operationId"]
            )
        )
        if run and (
            run.patient_id != u.patient_uuid
            or json.loads(run.source_report_ids_json or "[]")
            not in ([i["sourceId"] for i in selected], [])
            or (
                not json.loads(run.source_report_ids_json or "[]")
                and run.report_id not in {i["sourceId"] for i in selected}
            )
        ):
            raise CatalogueConflict(
                "Original analysis operation source binding differs"
            )
        analysis = {
            **analysis,
            "reportIds": [
                i["sourceId"] for i in selected if i["status"] == "registered"
            ],
            "runId": run.id if run else None,
            "status": "incompatible_policy"
            if not compatible and not run
            else run.status
            if run
            else (
                "needs_operator_answer"
                if u.status == "needs_operator_answer"
                else "ready"
                if all(i["status"] == "registered" for i in selected)
                else "pending_registration"
            ),
        }
    return dict(
        uploadId=u.id,
        status=u.status,
        patientId=patient.label if patient else None,
        identity=m["identity"],
        conflict=json.loads(u.conflict_json) if u.conflict_json else None,
        items=items,
        uploadedAt=u.uploaded_at,
        uploadedBy=u.uploaded_by,
        analysis=analysis,
    )


def _legacy_json(row):
    record = json.loads(row.record_json)
    return dict(
        uploadId=row.id,
        status=record.get("status", "uncertain"),
        patientId=record.get("patientId"),
        identity=record.get("identity", {}),
        conflict=record.get("conflict"),
        items=[],
        uploadedAt=record.get("uploadedAt"),
        uploadedBy=record.get("uploadedBy"),
        analysis=None,
        error=record.get("error"),
        legacy=True,
    )


def get_upload(upload_id):
    with storage.session_scope() as s:
        u = s.get(ClinicUpload, upload_id)
        if u:
            return _envelope(s, upload=_upload_json(s, u))
        legacy = s.get(ClinicLegacyUpload, upload_id)
        if legacy:
            return _envelope(s, upload=_legacy_json(legacy))
        raise CatalogueNotFound("Upload not found")


def list_uploads():
    with storage.session_scope() as s:
        uploads = [
            _upload_json(s, u)
            for u in s.scalars(
                select(ClinicUpload).order_by(
                    ClinicUpload.uploaded_at.desc(), ClinicUpload.id
                )
            )
        ]
        admitted = {u["uploadId"] for u in uploads}
        uploads.extend(
            _legacy_json(u)
            for u in s.scalars(
                select(ClinicLegacyUpload).order_by(ClinicLegacyUpload.id)
            )
            if u.id not in admitted
        )
        return _envelope(s, uploads=uploads)


def resolve_upload(upload_id, *, key, resolution, actor=None):
    key = require_key(key)
    resolution = _resolution(resolution)
    if not resolution:
        raise ValueError("A resolution is required")
    material = _json(
        dict(kind="resolution", uploadId=upload_id, resolution=resolution, actor=actor)
    )
    with _write() as s:
        u = s.get(ClinicUpload, upload_id)
        if not u:
            legacy = s.get(ClinicLegacyUpload, upload_id)
            if not legacy:
                raise CatalogueNotFound("Upload not found")
            prior = s.get(ClinicMutation, key)
            if prior:
                if prior.material_json != material:
                    raise CatalogueConflict("Resolution key changed")
                return _envelope(s, upload=_legacy_json(legacy))
            record = json.loads(legacy.record_json)
            if record.get("patientId") or record.get("status") == "uncertain":
                raise CatalogueConflict(
                    "Original upload requires source reconciliation"
                )
            if resolution.get("attachTo"):
                _patient(s, resolution["attachTo"])
            legacy.record_json = _json(
                {**record, "resolution": resolution, "status": "pending"}
            )
            s.add(
                ClinicMutation(
                    key=key,
                    material_json=material,
                    result_json=_json({"uploadId": upload_id}),
                )
            )
            _bump(s)
            return _envelope(s, upload=_legacy_json(legacy))
        admission = u.admission_key
    with upload_lock(admission):
        with _write() as s:
            prior = s.get(ClinicMutation, key)
            if prior and prior.material_json != material:
                raise CatalogueConflict("Resolution key changed")
            u = s.get(ClinicUpload, upload_id)
            if not prior:
                if u.patient_uuid:
                    raise CatalogueConflict("Upload already has a patient binding")
                if resolution.get("attachTo"):
                    _patient(s, resolution["attachTo"])
                u.resolution_json = _json(resolution)
                s.add(
                    ClinicMutation(
                        key=key,
                        material_json=material,
                        result_json=_json(dict(uploadId=upload_id)),
                    )
                )
                _bump(s)
        return _resume_locked(upload_id)


def promote_upload(upload_id, portal_dir):
    """Retryable exact byte projection; persist chosen path before any copy."""
    with storage.session_scope() as s:
        u = s.get(ClinicUpload, upload_id)
        if not u:
            raise CatalogueNotFound("Upload not found")
        key = u.admission_key
    with upload_lock(key):
        with storage.session_scope() as s:
            u = s.get(ClinicUpload, upload_id)
            if u.status != "registered":
                raise CatalogueConflict("Upload items remain unfinished")
            label = s.get(storage.Patient, u.patient_uuid).label
            ids = list(
                s.scalars(
                    select(ClinicUploadItem.id)
                    .where(ClinicUploadItem.upload_id == upload_id)
                    .order_by(ClinicUploadItem.position)
                )
            )
        outputs = []
        for item_id in ids:
            with _write() as s:
                item = s.get(ClinicUploadItem, item_id)
                m = json.loads(item.metadata_json)
                data = Path(item.staging_path).read_bytes()
                if hashlib.sha256(data).hexdigest() != m["sha256"]:
                    raise CatalogueUnavailable("Staging bytes changed")
                if item.projection_path:
                    path = Path(item.projection_path)
                    if path.parent != Path(portal_dir) / label:
                        raise CatalogueConflict(
                            "Projection target differs from original binding"
                        )
                else:
                    name = Path(m["originalName"].replace("\\", "/")).name
                    path = Path(portal_dir) / label / name
                    if path.exists() and path.read_bytes() != data:
                        path = path.with_name(
                            path.stem + "__" + item.source_id + path.suffix
                        )
                    # Every item in a same-name submission reserves its own path before copying.
                    occupied = s.scalar(
                        select(ClinicUploadItem.id).where(
                            ClinicUploadItem.projection_path == str(path),
                            ClinicUploadItem.id != item.id,
                        )
                    )
                    if occupied:
                        path = path.with_name(
                            path.stem + "__" + item.source_id + path.suffix
                        )
                    item.projection_path = str(path)
            _immutable(path, data)
            outputs.append(str(path))
        return outputs
