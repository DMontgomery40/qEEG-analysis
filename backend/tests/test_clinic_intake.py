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


@pytest.mark.parametrize("dates", [("2/2/1900", "02/02/1900"), ("02-02-1900", "2/2/1900")])
def test_equivalent_report_dates_on_known_chart_file_once(temp_data_dir, dates):
    patient = submit()["upload"]["patientId"]
    args = dict(patient_id=patient, identity={}, file_meta=[
        {"documentKind": "report", "reportBirthdate": date} for date in dates
    ])
    first = submit("mixed-spelling", **args)["upload"]
    assert first["status"] == "registered"
    assert first["patientId"] == patient
    assert submit("mixed-spelling", **args)["upload"] == first
    assert counts() == (1, 1, 2, 2, 0)


def test_different_report_dates_on_known_chart_remain_rejected(temp_data_dir):
    patient = submit()["upload"]["patientId"]
    with pytest.raises(ValueError, match="different dates of birth"):
        submit("different-dates", patient_id=patient, identity={}, file_meta=[
            {"documentKind": "report", "reportBirthdate": date}
            for date in ("2/2/1900", "03/03/1900")
        ])
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


@pytest.mark.parametrize("drift", ["settings", "prompts", "models", "presentation"])
def test_confirmed_policy_fingerprint_rejects_new_drift_but_replays_original(
    temp_data_dir, monkeypatch, drift
):
    from backend import clinic_analysis_intents as policy, config, clinic_naming
    from types import SimpleNamespace

    shown = policy.public_current_policy()
    intent = {
        "operationId": "confirmed-policy",
        "confirmed": True,
        "reportItemIndexes": [0],
        "specialInstructions": "",
        "expectedPolicyFingerprint": shown["analysisPolicyFingerprint"],
    }
    original = submit(
        file_meta=[{"documentKind": "report"}, {}], analysis_intent=intent
    )["upload"]
    binding = policy.confirmed_analysis_binding(original["uploadId"])
    assert binding["policyHash"] == shown["analysisPolicyFingerprint"]
    if drift == "settings":
        monkeypatch.setenv("QEEG_STAGE1_MAX_TOKENS", "98765")
    elif drift == "prompts":
        monkeypatch.setattr(policy, "_snapshot_prompts", lambda: {"new": "changed"})
    elif drift == "models":
        monkeypatch.setattr(config, "COUNCIL_MODELS", [SimpleNamespace(id="new-model")])
    else:
        monkeypatch.setitem(clinic_naming.POLICY, "tts", {"voice": "changed"})
    assert (
        policy.public_current_policy()["analysisPolicyFingerprint"]
        != shown["analysisPolicyFingerprint"]
    )
    replay = submit(file_meta=[{"documentKind": "report"}, {}], analysis_intent=intent)[
        "upload"
    ]
    assert replay["uploadId"] == original["uploadId"]
    assert (
        policy.confirmed_analysis_binding(original["uploadId"])["policySnapshot"]
        == binding["policySnapshot"]
    )
    before = counts()
    with pytest.raises(CatalogueConflict, match="policy changed"):
        submit(
            "new-policy",
            file_meta=[{"documentKind": "report"}, {}],
            analysis_intent={**intent, "operationId": "new-operation"},
        )
    assert counts() == before
    rejected_root = (
        temp_data_dir
        / "clinic_intake"
        / "submissions"
        / hashlib.sha256(b"new-policy").hexdigest()
    )
    assert not rejected_root.exists()


@pytest.mark.parametrize("fingerprint", [None, "", "z" * 64, "A" * 64, 123])
def test_invalid_expected_policy_fingerprint_is_not_admitted(
    temp_data_dir, fingerprint
):
    with pytest.raises(ValueError):
        submit(
            file_meta=[{"documentKind": "report"}, {}],
            analysis_intent={
                "operationId": "invalid-policy",
                "confirmed": True,
                "reportItemIndexes": [0],
                "specialInstructions": "",
                "expectedPolicyFingerprint": fingerprint,
            },
        )
    assert counts() == (0, 0, 0, 0, 0)


def test_identical_report_bytes_file_once_on_the_same_chart(temp_data_dir):
    # 2026-09-29: one Wellness Basic PDF became two engine reports (hub upload
    # at 18:04, chat drop at 18:07) and five paid runs landed on the copy.
    same = b"facts about the same session"
    first = submit(
        key="dup-1",
        files=[("wellness.txt", same, "text/plain"), ("note.txt", b"second", "text/plain")],
        file_meta=[{"documentKind": "report"}, {}],
    )["upload"]
    second = submit(
        key="dup-2",
        files=[("7a3d970d__wellness.txt", same, "text/plain"), ("note2.txt", b"third", "text/plain")],
        file_meta=[{"documentKind": "report"}, {}],
    )["upload"]
    assert second["items"][0]["status"] == "registered"
    assert second["items"][0]["sourceId"] == first["items"][0]["sourceId"]
    assert counts()[2] == 1, "the same bytes on the same chart are one report"
    assert counts()[3] == 2, "different patient files still file separately"
    # a different patient with the same bytes is a different report
    other = submit(
        key="dup-3",
        identity={"firstName": "Bea", "lastName": "Carter", "birthdate": "03-03-1901"},
        files=[("wellness.txt", same, "text/plain"), ("note3.txt", b"fourth", "text/plain")],
        file_meta=[{"documentKind": "report"}, {}],
    )["upload"]
    assert other["items"][0]["sourceId"] != first["items"][0]["sourceId"]
    assert counts()[2] == 2


def test_blocked_admission_backs_off_and_names_the_reason(temp_data_dir):
    # 2026-09-29: the scan retried a confirmed hub upload once a second for
    # 72 minutes with a full traceback each time, and the hub showed nothing.
    import asyncio
    from backend import clinic_analysis_intents as intents
    from backend.clinic_models import CatalogueUnavailable

    result = submit(
        file_meta=[{"documentKind": "report"}, {}],
        analysis_intent={
            "operationId": "op-blocked",
            "confirmed": True,
            "reportItemIndexes": [0],
            "specialInstructions": "",
        },
    )["upload"]
    calls = []

    class Store:
        @property
        def engine(self):
            return storage.engine

    class Runtime:
        store = Store()

        async def admission(self, fn, *args):
            calls.append(fn.__name__)
            raise CatalogueUnavailable("Original confirmed models are unavailable")

    intents._ADMISSION_BACKOFF.clear()
    for _ in range(4):
        asyncio.run(intents.activate_confirmed_uploads(Runtime()))
    assert calls == ["admit_confirmed_upload"] * 2, "one immediate retry, then the backoff holds"
    assert intents._ADMISSION_BACKOFF[result["uploadId"]]["attempts"] == 2
    assert intents.admission_block(result["uploadId"]) == "Original confirmed models are unavailable"
    def record():
        payload = intake().get_upload(result["uploadId"])
        return (payload.get("upload") or payload)["analysis"]

    shown = record()
    assert shown["status"] == "blocked"
    assert "models are unavailable" in shown["blockedReason"]
    # An engine restart empties memory; the block is on the upload row, so the
    # hub still sees why it stopped (it read `ready` until the next failure).
    intents._ADMISSION_BACKOFF.clear()
    assert record()["status"] == "blocked"
    assert "models are unavailable" in record()["blockedReason"]

    class Healthy(Runtime):
        async def admission(self, fn, *args):
            calls.append(fn.__name__)
            return None

    asyncio.run(intents.activate_confirmed_uploads(Healthy()))
    assert calls[-1] == "admit_confirmed_upload", "a restart is one fresh try"
    assert record()["status"] == "ready", "a stage that succeeds clears the stored block"


def test_policy_offers_only_models_the_engine_has(temp_data_dir, monkeypatch):
    # 2026-09-29: the hub offered a council pinned to openai/gpt-5.6-terra, an
    # id the engine never discovered, so the confirmed upload never ran.
    from backend import config, clinic_analysis_intents as policy
    from backend.clinic_models import CatalogueUnavailable

    council = [m.id for m in config.COUNCIL_MODELS]
    monkeypatch.setattr(
        config, "DISCOVERED_MODEL_IDS", set(council + [config.DEFAULT_CONSOLIDATOR])
    )
    assert policy.public_current_policy()["policy"]["analysis"]["councilModelIds"] == council
    monkeypatch.setattr(
        config, "DISCOVERED_MODEL_IDS", set(council[1:] + [config.DEFAULT_CONSOLIDATOR])
    )
    with pytest.raises(CatalogueUnavailable, match="on David's end") as refused:
        policy.public_current_policy()
    assert refused.value.models == (council[0],)
    with pytest.raises(CatalogueUnavailable):
        submit(
            file_meta=[{"documentKind": "report"}, {}],
            analysis_intent={
                "operationId": "op-unrunnable",
                "confirmed": True,
                "reportItemIndexes": [0],
                "specialInstructions": "",
            },
        )
    assert counts() == (0, 0, 0, 0, 0), "nothing is filed on a policy the engine cannot run"
    monkeypatch.setattr(config, "DISCOVERED_MODEL_IDS", set())
    assert policy.public_current_policy()["policy"]["analysis"]["councilModelIds"] == council


def test_two_matching_charts_park_the_upload_with_both_candidates(temp_data_dir):
    # Before 2026-10-02 this was a 400 the hub could not answer; the clinic is
    # asked one plain question with both charts offered, like a name mismatch.
    with storage.session_scope() as s:
        storage.create_patient(
            s, label="AB_02-02-1900", first_initial="A", last_initial="B", birthdate="02-02-1900"
        )
        storage.create_patient(
            s, label="AB_02-02-1900_2", first_initial="A", last_initial="B", birthdate="02-02-1900",
            first_name="Ada", last_name="Baker",
        )
    parked = submit("two-fit")["upload"]
    assert parked["status"] == "needs_operator_answer"
    assert parked["conflict"]["conflict"] == "identity_ambiguous"
    assert [c["patient_id"] for c in parked["conflict"]["candidates"]] == ["AB_02-02-1900", "AB_02-02-1900_2"]
    assert counts()[0] == 2, "no third chart was made"
    resolved = intake().resolve_upload(
        "two-fit", key="two-fit-answer", resolution={"attachTo": "AB_02-02-1900_2"}
    )["upload"]
    assert resolved["patientId"] == "AB_02-02-1900_2"
    assert resolved["status"] == "registered"
    assert counts()[0] == 2


def test_the_same_request_is_one_council_until_the_operator_asks_again(temp_data_dir, monkeypatch):
    # 2026-09-29: one PDF, five paid runs in 82 minutes under five operation ids.
    from datetime import datetime, timedelta, timezone
    from fastapi import HTTPException
    from backend import analysis_inputs

    upload = submit(file_meta=[{"documentKind": "report"}, {}])["upload"]
    report_id = upload["items"][0]["sourceId"]
    with storage.session_scope() as s:
        patient_uuid = s.scalar(select(storage.Patient.id).where(storage.Patient.label == upload["patientId"]))
    models = dict(council_model_ids=["m1"], consolidator_model_id="m2", requested_model_ids=["m1", "m2"],
                  resolved_model_ids=["m1", "m2"], creating_instance_id="t", model_catalogue_fingerprint="f")

    def admit(operation_id, **extra):
        return analysis_inputs.admit_run(
            patient_id=patient_uuid, source_ids=[report_id], special_instructions="",
            source_session_aliases={}, operation_id=operation_id, model_fields=lambda: dict(models),
            immutable_request={"patient_id": patient_uuid, "source_ids": [report_id], **extra},
        )

    first = admit("op-one")
    with pytest.raises(HTTPException) as refused:
        admit("op-two")
    assert refused.value.detail["code"] == "ANALYSIS_ALREADY_RUNNING"
    assert refused.value.detail["run_id"] == first.id
    with storage.session_scope() as s:
        run = s.get(storage.Run, first.id)
        run.status = "complete"; run.completed_at = datetime.now(timezone.utc).replace(tzinfo=None)
        s.commit()
    with pytest.raises(HTTPException) as recent:
        admit("op-three")
    assert recent.value.detail["code"] == "ANALYSIS_RECENTLY_COMPLETED"
    third = admit("op-four", force_new=True)
    assert third.id != first.id, "force_new is the operator's yes"
    with storage.session_scope() as s:
        run = s.get(storage.Run, first.id)
        run.completed_at = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(days=2)
        s.commit()
        third_run = s.get(storage.Run, third.id)
        third_run.status = "failed"
        s.commit()
    assert admit("op-five").id not in (first.id, third.id), "an old council does not block a new one"


def test_a_duplicate_council_refusal_is_final_for_a_hub_upload(temp_data_dir):
    # The scan treated ANALYSIS_RECENTLY_COMPLETED like a transient fault and
    # retried every 30 minutes, so the re-upload's paid council would start the
    # moment the 24-hour window closed.
    import asyncio
    from fastapi import HTTPException
    from backend import clinic_analysis_intents as intents

    result = submit(
        file_meta=[{"documentKind": "report"}, {}],
        analysis_intent={
            "operationId": "op-again",
            "confirmed": True,
            "reportItemIndexes": [0],
            "specialInstructions": "",
        },
    )["upload"]
    calls = []

    class Store:
        @property
        def engine(self):
            return storage.engine

    class Runtime:
        store = Store()

        async def admission(self, fn, *args):
            calls.append(fn.__name__)
            raise HTTPException(
                409,
                {"code": "ANALYSIS_RECENTLY_COMPLETED", "message": "finished recently", "run_id": "r1"},
            )

    intents._ADMISSION_BACKOFF.clear()
    for _ in range(3):
        asyncio.run(intents.activate_confirmed_uploads(Runtime()))
    intents._ADMISSION_BACKOFF.clear()  # an engine restart
    asyncio.run(intents.activate_confirmed_uploads(Runtime()))
    assert calls == ["admit_confirmed_upload"], "refused once, never retried"
    shown = intake().get_upload(result["uploadId"])["upload"]["analysis"]
    assert shown["status"] == "blocked"
    assert shown["blockedReason"].startswith("These same reports were analysed recently")
    assert "{" not in shown["blockedReason"]


def test_hub_upload_is_answered_before_slow_filing(temp_data_dir, monkeypatch):
    # Netlify ends a synchronous function at 60 s; filing took about 23 s per
    # report on 2026-09-29, so three reports filed inline read as a failure.
    import threading

    module = intake()
    gate = threading.Event()
    real = module._file_item

    def slow(item_id, patient_uuid):
        assert gate.wait(20)
        return real(item_id, patient_uuid)

    monkeypatch.setattr(module, "_file_item", slow)
    first = submit("hub-slow", principal="thrylen-service", acknowledge_first=True)["upload"]
    assert first["status"] == "pending"
    assert first["patientId"] == "AB_02-02-1900"
    assert [i["status"] for i in first["items"]] == ["pending", "pending"]
    again = submit("hub-slow", principal="thrylen-service", acknowledge_first=True)["upload"]
    assert again["status"] == "pending", "a replay while filing answers at once"
    gate.set()
    with module._filing_lock(first["uploadId"]):
        pass
    done = module.get_upload(first["uploadId"])["upload"]
    assert done["status"] == "registered"
    assert counts() == (1, 1, 0, 2, 0), "filed once, not twice"
    # the workbench's chat staging still files inside the request
    assert submit("chat-inline", principal="workbench")["upload"]["status"] == "registered"


def test_one_damaged_upload_row_does_not_hide_the_others(temp_data_dir):
    from backend.clinic_records import ClinicUpload

    good = submit("good-row")["upload"]
    bad = submit("bad-row", identity={"firstName": "Bea", "lastName": "Carter", "birthdate": "03-03-1901"})["upload"]
    with storage.session_scope() as s:
        s.get(ClinicUpload, bad["uploadId"]).manifest_json = "{not json"
        s.commit()
    listed = {u["uploadId"]: u for u in intake().list_uploads()["uploads"]}
    assert listed[good["uploadId"]]["status"] == "registered"
    assert listed[bad["uploadId"]] == {
        "uploadId": bad["uploadId"],
        "status": "unreadable",
        "error": "This upload's saved record could not be read.",
    }


def test_replay_after_the_analysis_request_was_cleared_is_a_plain_conflict(temp_data_dir):
    from backend.clinic_records import ClinicUpload

    intent = {
        "operationId": "op-cleared",
        "confirmed": True,
        "reportItemIndexes": [0],
        "specialInstructions": "",
    }
    first = submit("cleared", file_meta=[{"documentKind": "report"}, {}], analysis_intent=intent)["upload"]
    with storage.session_scope() as s:
        s.get(ClinicUpload, first["uploadId"]).analysis_json = None
        s.commit()
    with pytest.raises(CatalogueConflict) as refused:
        submit("cleared", file_meta=[{"documentKind": "report"}, {}], analysis_intent=intent)
    assert "analysis request was cleared" in str(refused.value)
    assert "NoneType" not in str(refused.value)


def test_a_dropdown_chart_conflict_says_which_birthday_differs(temp_data_dir):
    # MF_09-05-1954's upload sat parked four days behind "The supplied identity
    # differs from this chart" while the difference was the printed birthday.
    with storage.session_scope() as s:
        storage.create_patient(
            s, label="MF_09-05-1954", first_initial="M", last_initial="F",
            birthdate="09-05-1954", first_name="Mary", last_name="Fox",
        )
    parked = submit(
        "dropdown-dob",
        identity={},
        patient_id="MF_09-05-1954",
        file_meta=[{"documentKind": "report", "reportBirthdate": "03-05-2010"}, {}],
    )["upload"]
    assert parked["status"] == "needs_operator_answer"
    assert parked["conflict"]["detail"] == (
        "The report's printed birthday 03-05-2010 does not match this chart's "
        "09-05-1954. Same person, or someone different?"
    )
