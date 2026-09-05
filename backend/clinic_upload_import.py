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

    def matching_marker(value):
        return (
            isinstance(value, dict)
            and value.get("uploadId") == key
            and value.get("kind") == "new_patient_upload"
            and value.get("uploadedAt") == receipt["uploadedAt"]
            and value.get("uploadedBy") == m["uploadedBy"]
            and value.get("fileKey") in {f.get("fileKey") for f in expected["uploaded"]}
        )

    with _write() as s:
        prior = s.get(ClinicLegacyUpload, key)
        evidence = json.loads(prior.evidence_json) if prior else {}
        legacy = (
            json.loads(prior.record_json)
            if prior
            else dict(uploadId=key, status="pending", identity=m["identity"])
        )
        if prior:
            original = evidence.get("originalSubmission")
            if original and (
                original["manifest"] != m
                or original["uploadedAt"] != receipt["uploadedAt"]
                or original.get("response") != expected
            ):
                raise CatalogueConflict("Original submission identity changed")
            if (
                legacy.get("patientId")
                and registered is not None
                and registered.get("patientId") != legacy["patientId"]
            ):
                from .clinic_catalogue_reads import _patient

                try:
                    same_patient = (
                        _patient(s, registered.get("patientId")).id
                        == _patient(s, legacy["patientId"]).id
                    )
                except LookupError:
                    same_patient = False
                if not same_patient:
                    raise CatalogueConflict("Original registered chart binding changed")

        # Preserve the first receipt and every distinct observation, including absent
        # or mismatched markers. Replaying old phases cannot erase surviving proof.
        evidence.setdefault("originalSubmission", receipt)
        observations = evidence.setdefault("submissionObservations", [])
        observation = dict(receipt=receipt, marker=marker)
        if observation not in observations:
            observations.append(observation)
        marker_seen = any(matching_marker(o["marker"]) for o in observations)
        publishing_seen = any(
            o["receipt"]["phase"] in ("publishing", "published") for o in observations
        )
        previous_status = (
            legacy.get("preReconciliationStatus", "uncertain")
            if legacy["status"] == "uncertain"
            else legacy["status"]
        )
        uncertain = bool(legacy.get("patientId")) or (
            (publishing_seen or previous_status == "uncertain") and not marker_seen
        )
        record = {
            **legacy,
            "originalSubmission": evidence["originalSubmission"],
            "preReconciliationStatus": previous_status,
            "status": "uncertain" if uncertain else previous_status,
        }
        if prior:
            if prior.evidence_json != _json(evidence) or prior.record_json != _json(
                record
            ):
                prior.evidence_json = _json(evidence)
                prior.record_json = _json(record)
                _bump(s)
        else:
            s.add(
                ClinicLegacyUpload(
                    id=key, evidence_json=_json(evidence), record_json=_json(record)
                )
            )
            _bump(s)
    if files is None or (uncertain and registered is None):
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
