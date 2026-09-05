"""Durable filing uses ordered source identity, never a filename or retry count."""

from backend.tests.clinic_test_helpers import forbid_clinic_paid  # noqa: F401
from concurrent.futures import ThreadPoolExecutor
import hashlib
import importlib

import pytest
from sqlalchemy import select, func
from backend import storage
from backend.clinic_models import CatalogueConflict


def intake():
    return importlib.import_module("backend.clinic_intake")


def submit(key="queue-1", **kwargs):
    args = dict(
        key=key,
        identity={"firstName": "Ada", "lastName": "Baker", "birthdate": "02-02-1900"},
        files=[
            ("same.txt", b"first", "text/plain"),
            ("same.txt", b"second", "text/plain"),
        ],
        file_meta=[{}, {}],
        actor="Staff",
    )
    args.update(kwargs)
    return intake().submit_upload(**args)


def counts():
    with storage.session_scope() as s:
        return tuple(
            s.scalar(select(func.count()).select_from(m))
            for m in (
                storage.Patient,
                storage.PatientIdReservation,
                storage.Report,
                storage.PatientFile,
                storage.Run,
            )
        )


def test_ordered_bytes_survive_same_names_and_replays(temp_data_dir):
    first = submit()["upload"]
    assert first["status"] == "registered"
    assert first["patientId"] == "AB_02-02-1900"
    assert len({x["sourceId"] for x in first["items"]}) == 2
    assert submit()["upload"] == first
    second = submit("queue-2")["upload"]
    assert second["patientId"] == first["patientId"]
    assert counts() == (1, 1, 0, 4, 0)
    for item, expected in zip(first["items"], [b"first", b"second"]):
        assert item["sha256"] == hashlib.sha256(expected).hexdigest()
    assert intake().get_upload(first["uploadId"])["upload"] == first


@pytest.mark.parametrize(
    "change",
    [
        dict(
            files=[
                ("same.txt", b"changed", "text/plain"),
                ("same.txt", b"second", "text/plain"),
            ]
        ),
        dict(file_meta=[{"sessionDate": "2026-09-01"}, {}]),
        dict(
            identity={
                "firstName": "Anne",
                "lastName": "Baker",
                "birthdate": "02-02-1900",
            }
        ),
        dict(actor="Other"),
    ],
)
def test_admission_key_binds_all_material(temp_data_dir, change):
    submit()
    with pytest.raises(CatalogueConflict):
        submit(**change)
    assert counts() == (1, 1, 0, 2, 0)


def test_concurrent_lost_ack_reuses_one_binding(temp_data_dir):
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(lambda _: submit()["upload"], range(8)))
    assert all(r == results[0] for r in results)
    assert counts() == (1, 1, 0, 2, 0)


@pytest.mark.parametrize(
    "answer,want",
    [
        ({"attachTo": "AB_02-02-1900"}, "AB_02-02-1900"),
        ({"forceNew": True}, "AB_02-02-1900_2"),
    ],
)
def test_conflict_resolution_is_original_submission(temp_data_dir, answer, want):
    submit()
    pending = submit(
        "other",
        identity={"firstName": "Anne", "lastName": "Baker", "birthdate": "02-02-1900"},
    )["upload"]
    assert pending["status"] == "needs_operator_answer"
    resolved = intake().resolve_upload(
        pending["uploadId"], key="answer", resolution=answer, actor="Staff"
    )["upload"]
    assert resolved["patientId"] == want
    assert (
        intake().resolve_upload(
            pending["uploadId"], key="answer", resolution=answer, actor="Staff"
        )["upload"]
        == resolved
    )
    with storage.session_scope() as s:
        assert (
            s.scalar(
                select(storage.Patient).where(storage.Patient.label == "AB_02-02-1900")
            ).first_name
            == "Ada"
        )


def test_allocator_binding_failure_rolls_back_together(temp_data_dir, monkeypatch):
    intake()
    original = storage.create_patient

    def fail(*args, **kwargs):
        original(*args, **kwargs)
        raise RuntimeError("death after Patient flush")

    monkeypatch.setattr(storage, "create_patient", fail)
    with pytest.raises(RuntimeError):
        submit()
    assert counts() == (0, 0, 0, 0, 0)
    monkeypatch.setattr(storage, "create_patient", original)
    assert submit()["upload"]["patientId"] == "AB_02-02-1900"


def test_failed_extraction_preserves_other_items_and_retry(temp_data_dir, monkeypatch):
    from backend import reports

    real = reports.save_report_upload

    def fail(**kwargs):
        raise RuntimeError("free extraction interrupted")

    monkeypatch.setattr(reports, "save_report_upload", fail)
    args = dict(file_meta=[{"documentKind": "report"}, {}])
    failed = submit(**args)["upload"]
    assert [x["status"] for x in failed["items"]] == ["failed", "registered"]
    monkeypatch.setattr(reports, "save_report_upload", real)
    assert submit(**args)["upload"]["status"] == "registered"
    assert counts() == (1, 1, 1, 1, 0)


@pytest.mark.parametrize(
    "meta",
    [
        [],
        [{}],
        ["bad", {}],
        [{"documentKind": "report", "sessionDate": "not-date"}, {}],
    ],
)
def test_malformed_manifest_has_no_clinical_effect(temp_data_dir, meta):
    with pytest.raises(ValueError):
        submit(file_meta=meta)
    assert counts() == (0, 0, 0, 0, 0)


def test_confirmed_intent_is_bound_without_paid_admission(temp_data_dir):
    result = submit(
        file_meta=[{"documentKind": "report"}, {}],
        analysis_intent={
            "operationId": "original-analysis",
            "confirmed": True,
            "reportItemIndexes": [0],
            "specialInstructions": "Compare.",
        },
    )["upload"]
    assert result["analysis"]["operationId"] == "original-analysis"
    assert result["analysis"]["status"] == "ready"
    assert result["analysis"]["reportIds"] == [result["items"][0]["sourceId"]]
    assert counts() == (1, 1, 1, 1, 0)


@pytest.mark.parametrize("boundary", ["create_patient", "create_report"])
def test_actual_process_death_replacement_reuses_original_binding(
    temp_data_dir, boundary
):
    import os
    import subprocess
    import sys

    code = """
import os
from backend import storage
from backend.clinic_intake import submit_upload
storage.init_db()
original=getattr(storage,os.environ['CRASH_BOUNDARY'])
def crash(*a,**kw):
    original(*a,**kw)
    os._exit(71)
setattr(storage,os.environ['CRASH_BOUNDARY'],crash)
submit_upload(key='process-key',identity={'firstName':'Ada','lastName':'Baker','birthdate':'02-02-1900'},files=[('source.txt',b'facts','text/plain')],file_meta=[{'documentKind':'report'}],actor='Staff')
"""
    paired = temp_data_dir.parent / (temp_data_dir.name + "-paired")
    paired.mkdir()
    (paired / "data").symlink_to(temp_data_dir, target_is_directory=True)
    env = {
        **os.environ,
        "DATA_DIR": str(paired / "data"),
        "QEEG_ANALYSIS_ROOT": str(paired),
        "CRASH_BOUNDARY": boundary,
    }
    child = subprocess.run(
        [sys.executable, "-c", code], env=env, capture_output=True, timeout=15
    )
    assert child.returncode == 71, child.stderr.decode()
    if boundary == "create_patient":
        assert counts() == (0, 0, 0, 0, 0)
    else:
        assert counts() == (1, 1, 0, 0, 0)
    result = submit(
        "process-key",
        files=[("source.txt", b"facts", "text/plain")],
        file_meta=[{"documentKind": "report"}],
    )["upload"]
    assert result["patientId"] == "AB_02-02-1900"
    assert counts() == (1, 1, 1, 0, 0)


def test_all_reserved_collision_ordinals_survive_force_new_replays(temp_data_dir):
    for ordinal in range(1, 14):
        u = submit("collision-" + str(ordinal), resolution={"forceNew": True})["upload"]
        assert u["patientId"] == "AB_02-02-1900" + (
            "" if ordinal == 1 else "_" + str(ordinal)
        )
        assert (
            submit("collision-" + str(ordinal), resolution={"forceNew": True})["upload"]
            == u
        )
    assert counts() == (13, 13, 0, 26, 0)


def test_explicit_ambiguous_alias_cannot_select_first_patient(temp_data_dir):
    with storage.session_scope() as s:
        storage.create_patient(s, label="AB_02-02-1900")
        storage.create_patient(s, label="AB_02-02-1900")
    with pytest.raises(CatalogueConflict):
        submit(patient_id="AB_02-02-1900")
    assert counts() == (2, 0, 0, 0, 0)


def test_adopt_original_registered_sources_never_allocates_force_new_again(
    temp_data_dir,
):
    prior = submit("old")["upload"]
    registered = {
        "patientId": prior["patientId"],
        "sourceIds": [x["sourceId"] for x in prior["items"]],
    }
    adopted = submit(
        "original-legacy", resolution={"forceNew": True}, registered=registered
    )["upload"]
    assert adopted["patientId"] == prior["patientId"]
    assert [x["sourceId"] for x in adopted["items"]] == registered["sourceIds"]
    assert counts() == (1, 1, 0, 2, 0)


def test_projection_equal_names_across_concurrent_submissions_preserves_every_byte(
    temp_data_dir,
):
    a = submit("a", files=[("same.txt", b"A", "text/plain")], file_meta=[{}])["upload"]
    b = submit("b", files=[("same.txt", b"B", "text/plain")], file_meta=[{}])["upload"]
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(
            pool.map(
                lambda u: intake().promote_upload(
                    u["uploadId"], temp_data_dir / "portal"
                ),
                [a, b],
            )
        )
    from pathlib import Path

    assert {Path(paths[0]).read_bytes() for paths in results} == {b"A", b"B"}
    for u, paths in zip([a, b], results):
        assert intake().promote_upload(u["uploadId"], temp_data_dir / "portal") == paths


def test_confirmed_upload_cannot_borrow_existing_operation_identity(temp_data_dir):
    from backend.clinic_jobs import register_operation

    p = submit()["upload"]["patientId"]
    register_operation(
        "already-used",
        patient_id=p,
        producer="workbench",
        kind="video",
        original={"conversationId": "chat"},
    )
    with pytest.raises(CatalogueConflict):
        submit(
            "analysis",
            file_meta=[{"documentKind": "report"}, {}],
            analysis_intent={
                "operationId": "already-used",
                "confirmed": True,
                "reportItemIndexes": [0],
                "specialInstructions": "",
            },
        )


@pytest.mark.parametrize("indexes", [[True], [0, 0], [-1], [2], [1]])
def test_analysis_confirmation_requires_exact_selected_report_items(
    temp_data_dir, indexes
):
    with pytest.raises(ValueError):
        submit(
            file_meta=[{"documentKind": "report"}, {}],
            analysis_intent={
                "operationId": "analysis",
                "confirmed": True,
                "reportItemIndexes": indexes,
                "specialInstructions": "",
            },
        )
    assert counts() == (0, 0, 0, 0, 0)


def test_actual_run_reference_must_match_original_upload_chart(temp_data_dir):
    u = submit(
        file_meta=[{"documentKind": "report"}, {}],
        analysis_intent={
            "operationId": "intent",
            "confirmed": True,
            "reportItemIndexes": [0],
            "specialInstructions": "",
        },
    )["upload"]
    with storage.session_scope() as s:
        other = storage.create_patient(s, label="XY_01-01-1900")
        run = storage.create_run(
            s,
            patient_id=other.id,
            report_id=u["items"][0]["sourceId"],
            council_model_ids=[],
            consolidator_model_id="fake",
        )
        run.operation_id = "intent"
        s.commit()
    with pytest.raises(CatalogueConflict):
        intake().get_upload(u["uploadId"])


def test_original_upload_policy_survives_settings_and_prompt_drift(
    temp_data_dir, monkeypatch
):
    from backend.council import execution

    monkeypatch.setenv("QEEG_STAGE1_MAX_TOKENS", "777")
    intent = {
        "operationId": "frozen",
        "confirmed": True,
        "reportItemIndexes": [0],
        "specialInstructions": "Original",
    }
    u = submit(file_meta=[{"documentKind": "report"}, {}], analysis_intent=intent)[
        "upload"
    ]
    policy = importlib.import_module("backend.clinic_analysis_intents")
    first = policy.confirmed_analysis_binding(u["uploadId"])
    monkeypatch.setenv("QEEG_STAGE1_MAX_TOKENS", "999")
    monkeypatch.setattr(policy, "_snapshot_prompts", lambda: {"new": "changed"})
    replay = submit(file_meta=[{"documentKind": "report"}, {}], analysis_intent=intent)[
        "upload"
    ]
    assert replay["analysis"]["operationId"] == "frozen"
    bound = policy.confirmed_analysis_binding(u["uploadId"])
    assert bound["policySnapshot"]["settings"]["QEEG_STAGE1_MAX_TOKENS"] == "777"
    assert bound["policySnapshot"]["prompts"] == first["policySnapshot"]["prompts"]
    monkeypatch.setattr(execution, "_recipe", lambda: {"changed": "recipe"})
    assert (
        intake().get_upload(u["uploadId"])["upload"]["analysis"]["status"]
        == "incompatible_policy"
    )


def test_unreadable_original_upload_policy_is_explicit_and_preserves_bytes(
    temp_data_dir, monkeypatch
):
    policy = importlib.import_module("backend.clinic_analysis_intents")

    def broken():
        raise OSError("policy source unreadable")

    monkeypatch.setattr(policy, "_snapshot_prompts", broken)
    with pytest.raises(OSError):
        submit(
            file_meta=[{"documentKind": "report"}, {}],
            analysis_intent={
                "operationId": "broken",
                "confirmed": True,
                "reportItemIndexes": [0],
                "specialInstructions": "",
            },
        )
    assert counts() == (0, 0, 0, 0, 0)
    assert any(
        p.read_bytes() == b"first"
        for p in (temp_data_dir / "clinic_intake" / "submissions").glob("*/*.bytes")
    )


def test_known_chart_conflicting_identity_needs_explicit_answer(temp_data_dir):
    first = submit()["upload"]
    conflict = submit(
        "known",
        patient_id=first["patientId"],
        identity={"firstName": "Anne", "lastName": "Baker", "birthdate": "02-02-1900"},
    )["upload"]
    assert conflict["status"] == "needs_operator_answer"
    resolved = intake().resolve_upload(
        "known", key="known-answer", resolution={"forceNew": True}
    )["upload"]
    assert resolved["patientId"] == "AB_02-02-1900_2"
    assert counts() == (2, 2, 0, 4, 0)


@pytest.mark.parametrize(
    "identity",
    [{}, {"firstName": "Ada", "lastName": "Baker", "birthdate": "02-02-1900"}],
)
def test_known_legacy_chart_with_missing_normalized_identity_files_without_splitting(
    temp_data_dir, identity
):
    with storage.session_scope() as s:
        storage.create_patient(s, label="AB_02-02-1900")
    u = submit(patient_id="AB_02-02-1900", identity=identity)["upload"]
    assert u["status"] == "registered" and u["patientId"] == "AB_02-02-1900"
    assert counts()[0] == 1


def test_known_chart_birthdate_mismatch_is_resolved_without_renaming(temp_data_dir):
    p = submit()["upload"]["patientId"]
    u = submit(
        "wrong-dob",
        patient_id=p,
        identity={"firstName": "Ada", "lastName": "Baker", "birthdate": "03-03-1900"},
    )["upload"]
    assert u["status"] == "needs_operator_answer"
    answer = intake().resolve_upload(
        u["uploadId"], key="same", resolution={"attachTo": p}
    )["upload"]
    assert answer["patientId"] == p
    with storage.session_scope() as s:
        assert storage.list_patients(s)[0].birthdate == "02-02-1900"


def test_known_chart_report_birthdate_is_checked_before_filing(temp_data_dir):
    p = submit()["upload"]["patientId"]
    u = submit(
        "report-dob",
        patient_id=p,
        identity={},
        file_meta=[{"documentKind": "report", "reportBirthdate": "03-03-1900"}, {}],
    )["upload"]
    assert u["status"] == "needs_operator_answer"
    assert counts() == (1, 1, 0, 2, 0)


def test_missing_accepted_private_policy_fails_loudly_without_losing_filing(
    temp_data_dir,
):
    from backend.clinic_models import CatalogueUnavailable

    u = submit(
        file_meta=[{"documentKind": "report"}, {}],
        analysis_intent={
            "operationId": "policy-file",
            "confirmed": True,
            "reportItemIndexes": [0],
            "specialInstructions": "",
        },
    )["upload"]
    next(
        (temp_data_dir / "clinic_intake" / "submissions").glob("*/analysis-policy.json")
    ).unlink()
    with pytest.raises(CatalogueUnavailable):
        intake().get_upload(u["uploadId"])
    with pytest.raises(CatalogueUnavailable):
        submit(
            file_meta=[{"documentKind": "report"}, {}],
            analysis_intent={
                "operationId": "policy-file",
                "confirmed": True,
                "reportItemIndexes": [0],
                "specialInstructions": "",
            },
        )
    assert counts() == (1, 1, 1, 1, 0)
