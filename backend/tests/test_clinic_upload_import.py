from backend.tests.clinic_test_helpers import forbid_clinic_paid  # noqa: F401
import hashlib
import json
import pytest
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


def original_marker():
    return dict(
        kind="new_patient_upload",
        uploadId="original-id",
        uploadedAt=1234,
        uploadedBy="Doctor",
        fileKey="uploads/pending/original-id/000__same.txt",
    )


def durable_legacy_snapshot():
    from backend import storage
    from backend.clinic_intake import get_upload
    from backend.clinic_records import ClinicLegacyUpload

    with storage.session_scope() as session:
        row = session.get(ClinicLegacyUpload, "original-id")
        evidence = json.loads(row.evidence_json)
        record = json.loads(row.record_json)
    return dict(evidence=evidence, record=record, shared=get_upload("original-id"))


@pytest.mark.parametrize("answer_kind", ["forceNew", "attachTo"])
def test_independent_worker_uploads_share_answers_without_sharing_mutations(
    temp_data_dir, tmp_path, monkeypatch, answer_kind
):
    from backend import clinic_intake, reports, storage
    from backend.tests.test_portal_pipeline_worker import (
        _UploadClient,
        _upload_job,
        _fake_save_report_upload,
        _no_paid_work,
    )
    from scripts import portal_pipeline_worker as worker

    _no_paid_work(monkeypatch, worker)
    monkeypatch.setattr(
        reports, "save_report_upload", _fake_save_report_upload(temp_data_dir)
    )
    target = clinic_intake.submit_upload(
        key="seed",
        identity=_upload_job()["identity"],
        files=[("seed.txt", b"seed", "text/plain")],
        file_meta=[{}],
    )["upload"]["patientId"]
    answer = {"forceNew": True} if answer_kind == "forceNew" else {"attachTo": target}
    result_ids = []
    for index, name in enumerate(("Barry", "Beth")):
        upload_id = f"independent-{index}"
        payload = _upload_job()
        payload.update(
            uploadId=upload_id, fileKey=f"uploads/pending/{upload_id}/scan.pdf"
        )
        payload["identity"]["firstName"] = name
        job_key = f"pipeline/jobs/{upload_id}/upload.json"
        client = _UploadClient(
            {job_key: payload}, blobs={payload["fileKey"]: name.encode()}
        )
        args = dict(
            client=client,
            portal_dir=tmp_path / "portal",
            status_dir=tmp_path / "status",
            job_key=job_key,
            payload=payload,
        )
        assert (
            worker.process_new_patient_upload(**args).status == "needs_operator_answer"
        )
        record = pipeline_uploads.read_upload(upload_id)
        pipeline_uploads.write_upload({**record, "resolution": answer})
        accepted = worker.process_new_patient_upload(**args)
        assert accepted.status == "registered", accepted.note
        result_ids.append(accepted.patient_id)
        original_upload = clinic_intake.get_upload(upload_id)
        for _ in range(2):
            client._jobs[job_key] = payload
            replay = worker.process_new_patient_upload(**args)
            assert replay.patient_id == accepted.patient_id
            assert replay.status in ("registered", "already_registered")
            assert clinic_intake.get_upload(upload_id) == original_upload
    assert result_ids == (
        [target + "_2", target + "_3"]
        if answer_kind == "forceNew"
        else [target, target]
    )
    with storage.session_scope() as session:
        patients = storage.list_patients(session)
        assert len(patients) == (3 if answer_kind == "forceNew" else 1)
        assert sum(len(storage.list_reports(session, p.id)) for p in patients) == 2


@pytest.mark.parametrize(
    "legacy_status", ["pending", "needs_operator_answer", "registered"]
)
@pytest.mark.parametrize("phase", ["admitted", "publishing", "published"])
def test_existing_legacy_receipt_and_reconciliation_survive_process_replacement(
    temp_data_dir, legacy_status, phase
):
    import os
    import subprocess
    import sys
    from backend.clinic_upload_import import import_legacy_record

    legacy = dict(
        uploadId="original-id",
        status=legacy_status,
        identity=original()["manifest"]["identity"],
        conflict={"original": "parked"},
    )
    if legacy_status == "registered":
        legacy["patientId"] = "AB_02-02-1900"
    import_legacy_record(legacy)
    receipt = original(phase)
    expected_status = "uncertain"
    returned = import_submission_evidence(receipt)
    snapshot = durable_legacy_snapshot()
    assert returned["status"] == expected_status
    assert snapshot["shared"]["upload"]["status"] == expected_status
    assert snapshot["record"]["originalSubmission"] == receipt
    assert snapshot["record"].get("patientId") == legacy.get("patientId")
    assert snapshot["record"]["conflict"] == legacy["conflict"]
    assert all(snapshot["evidence"][key] == value for key, value in legacy.items())
    assert snapshot["evidence"]["originalSubmission"] == receipt
    assert snapshot["evidence"]["submissionObservations"] == [
        dict(receipt=receipt, marker=None)
    ]
    assert import_submission_evidence(receipt) == returned
    assert durable_legacy_snapshot() == snapshot
    paired = temp_data_dir.parent / (temp_data_dir.name + "-read-paired")
    paired.mkdir()
    (paired / "data").symlink_to(temp_data_dir, target_is_directory=True)
    code = """
import json
from backend import storage
from backend.tests.test_clinic_upload_import import durable_legacy_snapshot
storage.init_db()
print(json.dumps(durable_legacy_snapshot(),sort_keys=True))
"""
    child = subprocess.run(
        [sys.executable, "-c", code],
        env={
            **os.environ,
            "DATA_DIR": str(paired / "data"),
            "QEEG_ANALYSIS_ROOT": str(paired),
        },
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert child.returncode == 0, child.stderr
    assert json.loads(child.stdout) == snapshot
    assert counts() == (0, 0, 0, 0, 0)


def test_import_preserves_phase_marker_history_and_rejects_changed_material(
    temp_data_dir,
):
    import copy
    from backend.clinic_models import CatalogueConflict
    from backend.clinic_upload_import import import_legacy_record

    legacy = dict(
        uploadId="original-id",
        status="needs_operator_answer",
        identity={"firstName": "Ada"},
        conflict={"candidates": []},
        resolution={"forceNew": True},
    )
    import_legacy_record(legacy)
    first = original("admitted")
    import_submission_evidence(first)
    published = original("published")
    import_submission_evidence(published)
    assert durable_legacy_snapshot()["shared"]["upload"]["status"] == "uncertain"
    marker = original_marker()
    import_submission_evidence(published, marker=marker)
    state = durable_legacy_snapshot()
    assert state["record"]["status"] == "needs_operator_answer"
    assert state["record"]["resolution"] == legacy["resolution"]
    assert state["evidence"]["originalSubmission"] == first
    assert state["evidence"]["submissionObservations"] == [
        dict(receipt=first, marker=None),
        dict(receipt=published, marker=None),
        dict(receipt=published, marker=marker),
    ]
    # Each call describes the current marker. History does not prove it survives.
    import_submission_evidence(first)
    current = durable_legacy_snapshot()
    assert current["evidence"] == state["evidence"]
    assert current["record"]["status"] == "uncertain"
    assert current["record"]["resolution"] == legacy["resolution"]
    state = current
    for field in ("manifest", "uploadedAt", "response"):
        changed = copy.deepcopy(published)
        if field == "manifest":
            changed[field]["uploadedBy"] = "Another"
            changed["manifestHash"] = hashlib.sha256(
                json.dumps(changed[field], separators=(",", ":")).encode()
            ).hexdigest()
        elif field == "uploadedAt":
            changed[field] += 1
        else:
            changed[field]["uploaded"][0]["fileKey"] += "changed"
        with pytest.raises(CatalogueConflict):
            import_submission_evidence(changed, marker=marker)
        assert durable_legacy_snapshot() == state


def test_registered_import_cannot_replace_original_chart_evidence(temp_data_dir):
    from backend.clinic_models import CatalogueConflict
    from backend.clinic_upload_import import import_legacy_record

    import_legacy_record(
        dict(uploadId="original-id", status="registered", patientId="AB_02-02-1900")
    )
    import_submission_evidence(original(), marker=original_marker())
    before = durable_legacy_snapshot()
    with pytest.raises(CatalogueConflict):
        import_submission_evidence(
            original(),
            marker=original_marker(),
            registered={"patientId": "ZZ_01-01-1900", "sourceIds": ["unrelated"]},
        )
    assert durable_legacy_snapshot() == before
    assert before["shared"]["upload"]["patientId"] == "AB_02-02-1900"
    assert before["shared"]["upload"]["status"] == "uncertain"


def test_reconciliation_keeps_later_operator_answer_and_original_conflict(
    temp_data_dir,
):
    from backend.clinic_intake import resolve_upload
    from backend.clinic_upload_import import import_legacy_record

    conflict = {"candidates": [{"patient_id": "AB_02-02-1900", "name": "Ada Baker"}]}
    import_legacy_record(
        dict(
            uploadId="original-id",
            status="needs_operator_answer",
            identity=original()["manifest"]["identity"],
            conflict=conflict,
        )
    )
    receipt = original()
    import_submission_evidence(receipt)
    mismatched = {**original_marker(), "uploadedBy": "Another"}
    assert (
        import_submission_evidence(receipt, marker=mismatched)["status"] == "uncertain"
    )
    assert (
        import_submission_evidence(receipt, marker=original_marker())["status"]
        == "needs_operator_answer"
    )
    answer = {"forceNew": True}
    resolve_upload("original-id", key="operator-answer", resolution=answer)
    assert import_submission_evidence(receipt)["status"] == "uncertain"
    assert (
        import_submission_evidence(receipt, marker=original_marker())["status"]
        == "pending"
    )
    state = durable_legacy_snapshot()
    assert state["record"]["resolution"] == answer
    assert state["evidence"]["conflict"] == conflict
    assert state["evidence"]["submissionObservations"] == [
        dict(receipt=receipt, marker=None),
        dict(receipt=receipt, marker=mismatched),
        dict(receipt=receipt, marker=original_marker()),
    ]


@pytest.mark.parametrize("force_new", [False, True])
@pytest.mark.parametrize("prior_allocation", [False, True])
@pytest.mark.parametrize("current_marker_kind", ["absent", "mismatched"])
def test_current_marker_loss_blocks_fresh_admission_until_exact_reconciliation(
    temp_data_dir, force_new, prior_allocation, current_marker_kind
):
    import os
    import subprocess
    import sys
    from backend import clinic_intake
    from backend.clinic_models import CatalogueConflict

    receipt = original("publishing")
    receipt["manifest"]["resolution"] = {"forceNew": True} if force_new else {}
    receipt["manifestHash"] = hashlib.sha256(
        json.dumps(receipt["manifest"], separators=(",", ":")).encode()
    ).hexdigest()
    marker = original_marker()
    assert import_submission_evidence(receipt, marker=marker)["status"] == "pending"

    def original_source():
        return clinic_intake.submit_upload(
            key="separate-original-consumer",
            identity=receipt["manifest"]["identity"],
            resolution=receipt["manifest"]["resolution"],
            actor="Doctor",
            files=[("same.txt", b"one", "text/plain")],
            file_meta=[{}],
        )["upload"]

    source = original_source() if prior_allocation else None
    before = counts()
    current = (
        None if current_marker_kind == "absent" else {**marker, "uploadedBy": "Another"}
    )
    # An earlier observed marker could have been consumed by a separate legacy
    # registration. Replacement sees the current evidence and must not allocate.
    paired = temp_data_dir.parent / (temp_data_dir.name + "-marker-paired")
    paired.mkdir()
    (paired / "data").symlink_to(temp_data_dir, target_is_directory=True)
    code = """
import json, sys
from backend import storage
from backend.llm_client import AsyncOpenAICompatClient
from backend.paid_transport import PaidAsyncTransport, PaidSyncTransport
calls=[]
def forbidden(*a,**k):
    calls.append('paid')
    raise AssertionError('Import must not call providers')
for name in ('chat_completions','responses','list_models'):
    setattr(AsyncOpenAICompatClient,name,forbidden)
PaidAsyncTransport.handle_async_request=forbidden
PaidSyncTransport.handle_request=forbidden
from backend.clinic_upload_import import import_submission_evidence
from backend.tests.test_clinic_intake import counts
storage.init_db()
value=json.load(sys.stdin)
result=import_submission_evidence(value['receipt'],marker=value['marker'],files=[('same.txt',b'one','text/plain')])
print(json.dumps({'result':result,'counts':counts(),'calls':calls}))
"""
    child = subprocess.run(
        [sys.executable, "-c", code],
        input=json.dumps(dict(receipt=receipt, marker=current)),
        env={
            **os.environ,
            "DATA_DIR": str(paired / "data"),
            "QEEG_ANALYSIS_ROOT": str(paired),
        },
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert child.returncode == 0, child.stderr
    result = json.loads(child.stdout)
    assert result["result"].get("status") == "uncertain"
    assert tuple(result["counts"]) == before
    assert result["calls"] == []
    state = durable_legacy_snapshot()
    assert state["shared"]["upload"]["status"] == "uncertain"
    assert state["evidence"]["submissionObservations"] == [
        dict(receipt=receipt, marker=marker),
        dict(receipt=receipt, marker=current),
    ]
    assert (
        import_submission_evidence(
            receipt, marker=current, files=[("same.txt", b"one", "text/plain")]
        )["status"]
        == "uncertain"
    )
    assert counts() == before
    if source is None:
        source = original_source()
    original_counts = counts()
    registered = dict(
        patientId=source["patientId"], sourceIds=[source["items"][0]["sourceId"]]
    )
    with pytest.raises(CatalogueConflict):
        import_submission_evidence(
            receipt,
            marker=current,
            files=[("same.txt", b"one", "text/plain")],
            registered={**registered, "sourceIds": ["missing-source"]},
        )
    assert counts() == original_counts
    accepted = import_submission_evidence(
        receipt,
        marker=current,
        files=[("same.txt", b"one", "text/plain")],
        registered=registered,
    )["upload"]
    assert accepted["patientId"] == source["patientId"]
    assert [i["sourceId"] for i in accepted["items"]] == registered["sourceIds"]
    assert accepted["status"] == "registered"
    assert counts() == original_counts
    # The original common binding now permits safe resume despite marker absence.
    replay = import_submission_evidence(
        receipt, marker=current, files=[("same.txt", b"one", "text/plain")]
    )["upload"]
    assert replay == accepted
    assert counts() == original_counts


def test_marker_loss_reuses_common_patient_binding_after_interrupted_item_filing(
    temp_data_dir, monkeypatch
):
    from backend import patient_files

    original_save = patient_files.save_patient_file_upload

    def interrupted(**kwargs):
        raise OSError("filing interrupted after durable patient binding")

    monkeypatch.setattr(patient_files, "save_patient_file_upload", interrupted)
    receipt = original("published")
    receipt["manifest"]["resolution"] = {"forceNew": True}
    receipt["manifestHash"] = hashlib.sha256(
        json.dumps(receipt["manifest"], separators=(",", ":")).encode()
    ).hexdigest()
    first = import_submission_evidence(
        receipt, marker=original_marker(), files=[("same.txt", b"one", "text/plain")]
    )["upload"]
    assert first["patientId"] == "AB_02-02-1900"
    assert first["status"] == "failed"
    assert counts() == (1, 1, 0, 0, 0)
    monkeypatch.setattr(patient_files, "save_patient_file_upload", original_save)
    resumed = import_submission_evidence(
        receipt, marker=None, files=[("same.txt", b"one", "text/plain")]
    )["upload"]
    assert resumed["patientId"] == first["patientId"]
    assert resumed["items"][0]["sourceId"] == first["items"][0]["sourceId"]
    assert resumed["status"] == "registered"
    assert counts() == (1, 1, 0, 1, 0)


def test_marker_loss_does_not_treat_unbound_common_intent_as_allocation_proof(
    temp_data_dir, monkeypatch
):
    from backend import clinic_intake
    from backend.clinic_records import ClinicUpload
    from backend import storage

    bind = clinic_intake._bind_patient

    def interrupted(upload_id):
        raise OSError("interrupted before patient binding")

    monkeypatch.setattr(clinic_intake, "_bind_patient", interrupted)
    receipt = original("published")
    files = [("same.txt", b"one", "text/plain")]
    with pytest.raises(OSError, match="before patient binding"):
        import_submission_evidence(receipt, marker=original_marker(), files=files)
    with storage.session_scope() as session:
        assert session.get(ClinicUpload, "original-id").patient_uuid is None
    assert counts() == (0, 0, 0, 0, 0)
    monkeypatch.setattr(clinic_intake, "_bind_patient", bind)
    assert import_submission_evidence(receipt, files=files)["status"] == "uncertain"
    assert counts() == (0, 0, 0, 0, 0)


@pytest.mark.parametrize(
    "registered",
    [
        {},
        False,
        {"patientId": "AB_02-02-1900"},
        {"patientId": "AB_02-02-1900", "sourceIds": []},
        {"patientId": "AB_02-02-1900", "sourceIds": [""]},
    ],
)
def test_empty_or_incomplete_source_proof_cannot_bypass_current_marker_uncertainty(
    temp_data_dir, registered
):
    receipt = original("published")
    import_submission_evidence(receipt, marker=original_marker())
    with pytest.raises(ValueError):
        import_submission_evidence(
            receipt,
            marker=None,
            files=[("same.txt", b"one", "text/plain")],
            registered=registered,
        )
    assert counts() == (0, 0, 0, 0, 0)
