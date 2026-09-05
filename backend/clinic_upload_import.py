"""Narrow import of original upload evidence; uncertainty never admits new work."""

import hashlib
import json
from .clinic_catalogue import _write, _bump
from .clinic_catalogue_reads import _json
from .clinic_models import CatalogueConflict, CatalogueUnavailable
from .clinic_records import ClinicLegacyUpload
from .clinic_intake import require_key, submit_upload


def import_legacy_record(record):
    if not isinstance(record, dict):
        raise ValueError("Invalid legacy upload")
    upload_id = require_key(record.get("uploadId"))
    with _write() as s:
        existing = s.get(ClinicLegacyUpload, upload_id)
        if existing:
            return json.loads(existing.record_json)
        s.add(
            ClinicLegacyUpload(
                id=upload_id, evidence_json=_json(record), record_json=_json(record)
            )
        )
        _bump(s)
    return record


def import_submission_evidence(receipt, *, marker=None, files=None, registered=None):
    """Persist reviewed Thrylen receipt verbatim; files are ordered exact byte tuples.

    Uncertain claim/consumed marker requires reconciliation, not re-publication.
    This importer never invents analysis confirmation for historical uploads.
    """
    if (
        not isinstance(receipt, dict)
        or receipt.get("schemaVersion") != 1
        or receipt.get("phase") not in ("admitted", "publishing", "published")
    ):
        raise ValueError("Invalid original submission receipt")
    key = require_key(receipt.get("submissionId"))
    m = receipt.get("manifest")
    if (
        type(receipt.get("uploadedAt")) is not int
        or receipt["uploadedAt"] < 0
        or not isinstance(m, dict)
        or not isinstance(m.get("files"), list)
        or not m["files"]
        or not isinstance(m.get("identity"), dict)
        or not isinstance(m.get("resolution"), dict)
        or not isinstance(m.get("uploadedBy"), str)
    ):
        raise ValueError("Malformed submission manifest")
    encoded = json.dumps(m, ensure_ascii=False, separators=(",", ":")).encode()
    if hashlib.sha256(encoded).hexdigest() != receipt.get("manifestHash"):
        raise CatalogueConflict("Original submission manifest hash differs")
    for f in m["files"]:
        if (
            not isinstance(f, dict)
            or type(f.get("size")) is not int
            or f["size"] < 1
            or not isinstance(f.get("sha256"), str)
            or len(f["sha256"]) != 64
            or not all(
                isinstance(f.get(k), str) and f[k]
                for k in ("name", "originalName", "contentType")
            )
        ):
            raise ValueError("Malformed original item manifest")
    if "analysisIntent" in m:
        raise CatalogueUnavailable(
            "Historical analysis intent requires explicit reviewed migration"
        )
    expected = receipt.get("response")
    if (
        not isinstance(expected, dict)
        or expected.get("uploadId") != key
        or not isinstance(expected.get("uploaded"), list)
        or len(expected["uploaded"]) != len(m["files"])
    ):
        raise ValueError("Missing original upload response")
    marker_matches = (
        isinstance(marker, dict)
        and marker.get("uploadId") == key
        and marker.get("kind") == "new_patient_upload"
        and marker.get("uploadedAt") == receipt["uploadedAt"]
        and marker.get("uploadedBy") == m["uploadedBy"]
        and marker.get("fileKey") in {f.get("fileKey") for f in expected["uploaded"]}
    )
    uncertain = (
        receipt["phase"] in ("publishing", "published")
        and not marker_matches
        and registered is None
    )
    record = dict(
        uploadId=key,
        status="uncertain" if uncertain else "pending",
        identity=m["identity"],
        originalSubmission=receipt,
    )
    with _write() as s:
        prior = s.get(ClinicLegacyUpload, key)
        if prior:
            legacy = json.loads(prior.record_json)
            if legacy.get("patientId") and registered is None:
                uncertain = True
                record["status"] = "uncertain"
            original = json.loads(prior.evidence_json).get("originalSubmission")
            if original and (
                original["manifest"] != m
                or original["uploadedAt"] != receipt["uploadedAt"]
                or original.get("response") != expected
            ):
                raise CatalogueConflict("Original submission identity changed")
        else:
            s.add(
                ClinicLegacyUpload(
                    id=key, evidence_json=_json(record), record_json=_json(record)
                )
            )
            _bump(s)
    if uncertain or files is None:
        return record
    if len(files) != len(m["files"]):
        raise ValueError("Exact ordered original bytes required")
    for (_, data, _), f in zip(files, m["files"]):
        if hashlib.sha256(data).hexdigest() != f["sha256"] or len(data) != f["size"]:
            raise CatalogueConflict("Original upload item bytes differ")
    return submit_upload(
        key=key,
        upload_id=key,
        registered=registered,
        uploaded_at=receipt["uploadedAt"],
        identity=m["identity"],
        resolution=m["resolution"],
        actor=m["uploadedBy"],
        files=[
            (f["originalName"], data, f["contentType"])
            for (_, data, _), f in zip(files, m["files"])
        ],
        file_meta=[
            {k: f.get(k) for k in ("documentKind", "sessionDate", "reportBirthdate")}
            | {"originalFileKey": stored["fileKey"]}
            for f, stored in zip(m["files"], expected["uploaded"])
        ],
    )
