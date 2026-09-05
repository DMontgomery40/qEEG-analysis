from backend.tests.clinic_test_helpers import forbid_clinic_paid  # noqa: F401
import importlib
import pytest
from sqlalchemy import select
from backend import storage, clinic_catalogue, clinic_catalogue_reads as reads
from backend.clinic_models import CatalogueConflict, CatalogueNotFound


def artifact(root, source="source"):
    with storage.session_scope() as s:
        patients = storage.list_patients(s)
        patient = (
            patients[0]
            if patients
            else storage.create_patient(
                s,
                label="AB_02-02-1900",
                first_name="Ada",
                last_name="Baker",
                first_initial="A",
                last_initial="B",
                birthdate="02-02-1900",
            )
        )
    path = root / source
    path.write_bytes(source.encode())
    return clinic_catalogue.register_artifact(
        patient_uuid=patient.id,
        source_kind="renderer",
        source_id=source,
        logical_family="video",
        original_name="video.mp4",
        local_path=path,
    )


def test_feedback_history_approval_and_archival_preserve_bytes(temp_data_dir):
    m = importlib.import_module("backend.clinic_feedback")
    old = artifact(temp_data_dir, "older")
    new = artifact(temp_data_dir, "newer")
    args = dict(
        patient_id=old["patientId"],
        file_id=old["fileId"],
        version=old["version"],
        actor="Doctor",
    )
    rejected = m.record_feedback(
        key="reject", action="reject", notes="Please revise.", **args
    )
    assert (
        m.record_feedback(key="reject", action="reject", notes="Please revise.", **args)
        == rejected
    )
    with pytest.raises(CatalogueConflict):
        m.record_feedback(key="reject", action="approve", notes="", **args)
    m.record_feedback(
        key="approve",
        patient_id=new["patientId"],
        file_id=new["fileId"],
        version=new["version"],
        action="approve",
        actor="Doctor",
    )
    m.record_feedback(key="note", action="notes", notes="Follow up.", **args)
    rows = reads.patient_files(old["patientId"])["files"]
    byid = {r["fileId"]: r for r in rows}
    assert byid[old["fileId"]]["archived"] is True
    assert byid[old["fileId"]]["feedback"]["action"] == "reject"
    assert byid[old["fileId"]]["feedback"]["submittedBy"] == "Doctor"
    assert (temp_data_dir / "older").read_bytes() == b"older"
    m.record_notification("reject", status="failed", detail="notifier unavailable")
    assert m.feedback_history(old["fileId"])[0]["notification"]["status"] == "failed"
    assert len(m.feedback_history(old["fileId"])) == 2


def test_patch_receipt_cannot_allocate_again_after_lost_ack(temp_data_dir):
    a = artifact(temp_data_dir)
    module = importlib.import_module("backend.clinic_patient_updates")
    updated = module.patch_patient(
        a["patientId"],
        key="rename",
        changes={"firstName": "Zoë", "notes": "Enjoys music"},
        actor="Doctor",
    )
    assert updated["patient"]["patientId"] == "ZB_02-02-1900"
    assert (
        module.patch_patient(
            a["patientId"],
            key="rename",
            changes={"firstName": "Zoë", "notes": "Enjoys music"},
            actor="Doctor",
        )
        == updated
    )
    assert reads.roster(a["patientId"])["patient"]["patientId"] == "ZB_02-02-1900"
    with pytest.raises(CatalogueConflict):
        module.patch_patient(
            a["patientId"], key="rename", changes={"firstName": "Zed"}, actor="Doctor"
        )


def test_job_projection_rejects_unknown_and_preserves_terminal(temp_data_dir):
    a = artifact(temp_data_dir)
    jobs = importlib.import_module("backend.clinic_jobs")
    with pytest.raises(CatalogueNotFound):
        jobs.update_operation(
            "unknown",
            producer="workbench",
            generation=1,
            sequence=1,
            payload={"status": "complete"},
        )
    jobs.register_operation(
        "original-renderer-id",
        patient_id=a["patientId"],
        producer="workbench",
        kind="video",
        original={"conversationId": "original-conversation"},
    )
    assert (
        jobs.update_operation(
            "original-renderer-id",
            producer="workbench",
            generation=2,
            sequence=5,
            payload={"status": "complete", "fileId": a["fileId"]},
        )
        is True
    )
    assert (
        jobs.update_operation(
            "original-renderer-id",
            producer="workbench",
            generation=1,
            sequence=99,
            payload={"status": "running"},
        )
        is False
    )
    with pytest.raises(CatalogueConflict):
        jobs.update_operation(
            "original-renderer-id",
            producer="workbench",
            generation=2,
            sequence=5,
            payload={"status": "failed"},
        )
    with pytest.raises(CatalogueConflict):
        jobs.update_operation(
            "original-renderer-id",
            producer="other",
            generation=3,
            sequence=1,
            payload={"status": "failed"},
        )
    rows = jobs.patient_jobs(a["patientId"])["jobs"]
    assert rows[0]["operationId"] == "original-renderer-id"
    assert rows[0]["status"] == "complete"


def test_engine_jobs_read_original_stage_and_post_receipts_without_file_inference(
    temp_data_dir,
):
    from backend.clinic_jobs import patient_jobs
    from backend.tests.test_clinic_intake import submit

    u = submit(
        files=[("report.txt", b"facts", "text/plain")],
        file_meta=[{"documentKind": "report"}],
    )["upload"]
    with storage.session_scope() as s:
        p = s.scalar(
            select(storage.Patient).where(storage.Patient.label == u["patientId"])
        )
        run = storage.create_run(
            s,
            patient_id=p.id,
            report_id=u["items"][0]["sourceId"],
            council_model_ids=["original-model"],
            consolidator_model_id="original-model",
        )
        s.add(
            storage.StageReceipt(
                run_id=run.id,
                stage_num=1,
                receipt_path="original-stage-receipt",
                receipt_hash="a" * 64,
                execution_manifest_hash="b" * 64,
                input_fingerprint="c" * 64,
                policy_version="original-policy",
                owner_token="owner",
                owner_generation=1,
            )
        )
        s.add(
            storage.PostObligation(
                run_id=run.id,
                kind="patient_facing",
                manifest_path="original-post-manifest",
                manifest_hash="d" * 64,
                owner_token="owner",
                owner_generation=1,
                state="pending",
            )
        )
        s.commit()
    (temp_data_dir / "finished.mp4").write_bytes(b"not a job receipt")
    job = patient_jobs(u["patientId"])["jobs"][0]
    assert job["runId"] == run.id and job["status"] == "created"
    assert job["stages"][0]["receiptHash"] == "a" * 64
    assert job["post"][0]["state"] == "pending"
