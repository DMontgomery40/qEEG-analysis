"""Free catalogue projections from original E5 and export producer identities."""

from datetime import datetime
import hashlib
import json
import os
import re
from pathlib import Path
import shutil
import tempfile
from . import storage
from .clinic_catalogue import register_artifact, _read_local
from .clinic_catalogue_reads import _patient
from .clinic_models import CatalogueConflict, CatalogueUnavailable


def retained_producer_path(
    path, source_kind, source_id, *, expected_sha256=None, expected_size=None
):
    """Preserve exact bytes before a mutable latest path can be reused."""
    if expected_sha256 is not None or expected_size is not None:
        if (
            not isinstance(expected_sha256, str)
            or not re.fullmatch(r"[a-f0-9]{64}", expected_sha256)
            or type(expected_size) is not int
            or expected_size < 0
        ):
            raise ValueError("Expected SHA-256 and size must be supplied together")
    path = Path(path)
    identity = json.dumps([source_kind, source_id], separators=(",", ":")).encode()
    directory = (
        Path(storage.DATA_DIR)
        / "clinic_producer_bytes"
        / hashlib.sha256(identity).hexdigest()
    )
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / "original"
    fd, pending = tempfile.mkstemp(dir=directory)
    try:
        with os.fdopen(fd, "wb") as out, path.open("rb") as source:
            before = os.fstat(source.fileno())
            digest, size = hashlib.sha256(), 0
            while chunk := source.read(65536):
                out.write(chunk)
                digest.update(chunk)
                size += len(chunk)
            out.flush()
            os.fsync(out.fileno())
            after = os.fstat(source.fileno())
            current = path.stat()

            def fingerprint(st):
                return (
                    st.st_dev,
                    st.st_ino,
                    st.st_size,
                    st.st_mtime_ns,
                    st.st_ctime_ns,
                )

            if fingerprint(before) != fingerprint(after) or fingerprint(
                after
            ) != fingerprint(current):
                raise CatalogueUnavailable("Producer bytes changed during snapshot")
        # Validate the bytes in this exact private snapshot before the immutable
        # source identity can be reserved. A caller-side/path precheck races.
        if expected_sha256 is not None and (digest.hexdigest(), size) != (
            expected_sha256,
            expected_size,
        ):
            raise CatalogueConflict(
                "Original producer bytes differ from expected binding"
            )
        try:
            os.link(pending, target)
        except FileExistsError:
            if _read_local(pending)[1:3] != _read_local(target)[1:3]:
                raise CatalogueConflict("Original producer bytes changed")
        directory_fd = os.open(directory, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        os.unlink(pending)
    return target


def register_original_output(
    *,
    patient_uuid,
    source_kind,
    source_id,
    path,
    original_name,
    logical_family,
    expected_sha256=None,
    expected_size=None,
    **metadata,
):
    snapshot = retained_producer_path(
        path,
        source_kind,
        source_id,
        expected_sha256=expected_sha256,
        expected_size=expected_size,
    )
    args = dict(
        patient_uuid=patient_uuid,
        source_kind=source_kind,
        source_id=source_id,
        original_name=original_name,
        logical_family=logical_family,
        **metadata,
    )
    artifact = register_artifact(**args, local_path=snapshot)
    # An original E5 manifest may pin a configured portal outside DATA_DIR.
    # Retain that location without granting local serving authority; the exact
    # snapshot inside DATA_DIR remains the downloadable verified source.
    if not Path(path).resolve().is_relative_to(Path(storage.DATA_DIR).resolve()):
        from .clinic_catalogue import _write, _location, _bump
        from .clinic_catalogue_reads import _artifact_json
        from .clinic_models import ClinicArtifact

        with _write() as s:
            row = s.get(ClinicArtifact, artifact["fileId"])
            patient = s.get(storage.Patient, patient_uuid)
            affected = _location(
                s, row, "local", str(Path(path).resolve()), patient.label, False
            )
            if affected:
                _bump(s, affected)
            s.flush()
            return _artifact_json(s, row, patient)
    return register_artifact(**args, local_path=path)


def register_patient_output(owner, data, kind, binding):
    generated = int(datetime.fromisoformat(data["generated_at"]).timestamp() * 1000)
    with owner.file_guard():
        return register_original_output(
            patient_uuid=data["patient_id"],
            source_kind="patient-facing",
            source_id=json.dumps(
                [owner.run_id, "patient_facing", kind], separators=(",", ":")
            ),
            path=binding["path"],
            original_name=Path(binding["path"]).name,
            logical_family="patient-facing:" + kind,
            sha256=binding["sha256"],
            size=binding["size"],
            document_kind="patient-summary" if kind in ("md", "pdf") else "technical",
            generated_at=generated,
            provenance={
                "runId": owner.run_id,
                "obligation": "patient_facing",
                "outputKind": kind,
                "sourceFingerprint": data["source_fingerprint"],
            },
        )


def freeze_council_export(payload):
    """Freeze original selected-source event; a retry timestamp is not an event."""
    from .clinic_intake import _immutable, upload_lock

    identity = [payload["run_id"], payload["selected_artifact_id"]]
    if not identity[1]:
        raise CatalogueUnavailable("Original selected artifact identity required")
    key = hashlib.sha256(json.dumps(identity).encode()).hexdigest()
    path = Path(storage.DATA_DIR) / "clinic_export_events" / key / "event.json"
    with upload_lock("export:" + key):
        _, digest, size, _ = _read_local(payload["selected_artifact"]["content_path"])
        source = dict(sha256=digest, size=size)
        if path.exists():
            saved = json.loads(path.read_bytes())
            if saved["source"] != source:
                raise CatalogueConflict("Original selected export source changed")
            original = saved["payload"]
        else:
            original = payload
            outputs = {}
            for kind in ("md", "pdf"):
                _, output_hash, output_size, _ = _read_local(
                    payload["exports"]["final_" + kind]
                )
                outputs[kind] = [output_hash, output_size]
            saved = dict(source=source, payload=payload, outputs=outputs)
            _immutable(
                path,
                json.dumps(saved, sort_keys=True).encode(),
            )
        # Reuse the original event's accepted byte snapshots on retries. Every
        # path remains a mutable latest location; each distinct selected source
        # retains an independent immutable snapshot and catalogue identity.
        for kind in ("md", "pdf"):
            source_id = json.dumps([*identity, kind], separators=(",", ":"))
            destination = Path(payload["exports"]["final_" + kind])
            snapshot_root = (
                Path(storage.DATA_DIR)
                / "clinic_producer_bytes"
                / hashlib.sha256(
                    json.dumps(
                        ["council-export", source_id], separators=(",", ":")
                    ).encode()
                ).hexdigest()
            )
            snapshot = snapshot_root / "original"
            if not snapshot.exists():
                if list(_read_local(destination)[1:3]) != saved["outputs"][kind]:
                    raise CatalogueConflict(
                        "Original export bytes changed before snapshot recovery"
                    )
                snapshot = retained_producer_path(
                    destination, "council-export", source_id
                )
            if list(_read_local(snapshot)[1:3]) != saved["outputs"][kind]:
                raise CatalogueConflict(
                    "Original export snapshot differs from retained event"
                )
            for target in (destination, payload["exports"].get("portal_final_" + kind)):
                if target:
                    target = Path(target)
                    target.parent.mkdir(parents=True, exist_ok=True)
                    fd, tmp = tempfile.mkstemp(dir=target.parent)
                    try:
                        with os.fdopen(fd, "wb") as out, snapshot.open(
                            "rb"
                        ) as incoming:
                            shutil.copyfileobj(incoming, out, 65536)
                            out.flush()
                            os.fsync(out.fileno())
                        os.replace(tmp, target)
                    finally:
                        if os.path.exists(tmp):
                            os.unlink(tmp)
        return original


def register_council_export(payload, manifest_path):
    """Original selected artifact identifies the persisted export event."""
    with storage.session_scope() as s:
        patient = _patient(s, payload["patient_label"])
        run = s.get(storage.Run, payload["run_id"])
        if run is None or run.patient_id != patient.id:
            raise CatalogueConflict("Original export Run binding differs")
        patient_uuid = patient.id
    selected = payload["selected_artifact_id"]
    if not selected:
        raise CatalogueUnavailable("Original selected artifact identity required")
    event = [payload["run_id"], selected]
    generated = int(datetime.fromisoformat(payload["exported_at"]).timestamp() * 1000)
    outputs = {
        "md": payload["exports"]["final_md"],
        "pdf": payload["exports"]["final_pdf"],
        "meta": str(manifest_path),
    }
    results = {}
    for kind, path in outputs.items():
        results[kind] = register_original_output(
            patient_uuid=patient_uuid,
            source_kind="council-export",
            source_id=json.dumps([*event, kind], separators=(",", ":")),
            path=path,
            original_name=Path(path).name,
            logical_family="council-export:" + kind,
            document_kind="council-export",
            generated_at=generated,
            provenance={
                "runId": payload["run_id"],
                "selectedArtifactId": selected,
                "exportedAt": payload["exported_at"],
                "outputKind": kind,
            },
        )
    return results
