from backend.tests.clinic_test_helpers import forbid_clinic_paid  # noqa: F401
import hashlib
import json
from backend import pipeline_uploads
from backend.clinic_upload_import import import_submission_evidence
from backend.tests.test_clinic_intake import counts


def original(phase="published"):
    m = dict(
        identity={"firstName": "Ada", "lastName": "Baker", "birthdate": "02-02-1900"},
        resolution={},
        uploadedBy="Doctor",
        files=[
            dict(
                name="same.txt",
                originalName="same.txt",
                size=3,
                contentType="text/plain",
                documentKind=None,
                sessionDate=None,
                reportBirthdate=None,
                sha256=hashlib.sha256(b"one").hexdigest(),
            )
        ],
    )
    return dict(
        schemaVersion=1,
        submissionId="original-id",
        uploadedAt=1234,
        manifest=m,
        manifestHash=hashlib.sha256(
            json.dumps(m, separators=(",", ":")).encode()
        ).hexdigest(),
        phase=phase,
        response=dict(
            uploadId="original-id",
            uploaded=[dict(fileKey="uploads/pending/original-id/000__same.txt")],
        ),
    )


def test_published_import_preserves_original_identifiers_and_bytes(temp_data_dir):
    receipt = original()
    marker = dict(
        kind="new_patient_upload",
        uploadId="original-id",
        uploadedAt=1234,
        uploadedBy="Doctor",
        fileKey="uploads/pending/original-id/000__same.txt",
    )
    a = import_submission_evidence(
        receipt, marker=marker, files=[("same.txt", b"one", "text/plain")]
    )["upload"]
    assert a["uploadId"] == "original-id" and a["uploadedAt"] == 1234
    assert (
        import_submission_evidence(
            receipt, marker=marker, files=[("same.txt", b"one", "text/plain")]
        )["upload"]
        == a
    )
    assert counts() == (1, 1, 0, 1, 0)


def test_ambiguous_publication_never_recreates_consumed_work(temp_data_dir):
    result = import_submission_evidence(
        original("publishing"), files=[("same.txt", b"one", "text/plain")]
    )
    assert result["status"] == "uncertain"
    assert counts() == (0, 0, 0, 0, 0)


def test_legacy_json_is_read_once_and_registered_cannot_reset(
    temp_data_dir, monkeypatch
):
    monkeypatch.setattr(
        pipeline_uploads, "uploads_dir", lambda: temp_data_dir / "legacy"
    )
    path = temp_data_dir / "legacy" / "old.json"
    path.parent.mkdir()
    path.write_text(
        json.dumps(dict(uploadId="old", status="registered", patientId="AB_02-02-1900"))
    )
    assert pipeline_uploads.read_upload("old")["status"] == "registered"
    path.write_text(json.dumps(dict(uploadId="old", status="pending")))
    pipeline_uploads.record_seen(upload_id="old", identity={})
    assert pipeline_uploads.read_upload("old")["status"] == "registered"


def test_uncertain_import_is_visible_on_shared_upload_reads(temp_data_dir):
    from backend import clinic_intake

    import_submission_evidence(original("publishing"))
    row = clinic_intake.get_upload("original-id")["upload"]
    assert row["status"] == "uncertain"
    assert clinic_intake.list_uploads()["uploads"] == [row]


def test_legacy_registered_marker_cannot_allocate_again(temp_data_dir):
    from backend.clinic_upload_import import import_legacy_record

    import_legacy_record(
        dict(uploadId="original-id", status="registered", patientId="AB_02-02-1900")
    )
    receipt = original()
    marker = dict(
        kind="new_patient_upload",
        uploadId="original-id",
        uploadedAt=1234,
        uploadedBy="Doctor",
        fileKey="uploads/pending/original-id/000__same.txt",
    )
    result = import_submission_evidence(
        receipt, marker=marker, files=[("same.txt", b"one", "text/plain")]
    )
    assert result["status"] == "uncertain"
    assert counts() == (0, 0, 0, 0, 0)


def test_legacy_conflict_can_be_answered_through_shared_api_domain(temp_data_dir):
    from backend.clinic_upload_import import import_legacy_record
    from backend.clinic_intake import resolve_upload

    import_legacy_record(
        dict(
            uploadId="old",
            status="needs_operator_answer",
            identity={"firstName": "Ada"},
            conflict={"candidates": []},
        )
    )
    result = resolve_upload(
        "old", key="answer", resolution={"forceNew": True}, actor="Doctor"
    )
    assert result["upload"]["status"] == "pending"
    assert pipeline_uploads.pending_resolution("old") == {"forceNew": True}
    assert (
        resolve_upload(
            "old", key="answer", resolution={"forceNew": True}, actor="Doctor"
        )
        == result
    )
    assert counts() == (0, 0, 0, 0, 0)
