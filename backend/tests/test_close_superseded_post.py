"""Closing a blocked write-up another run already delivered; scratch DBs only.

DS_07-31-1957 and JP_09-25-1977 each have a blocked patient write-up beside a
sibling run whose PDF was delivered the same day. Regenerating would pay for a
duplicate; this closes the blocked one as superseded, with an audit record.
"""

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
from sqlalchemy import select

from backend import storage

REPO = Path(__file__).resolve().parents[2]
REASON = "the 09-29 write-up from the sibling run was delivered"


@pytest.fixture
def db(tmp_path):
    url = f"sqlite:///{tmp_path / 'app.db'}"
    storage.reset_engine(url)
    storage.init_db()
    return tmp_path


def script():
    from backend.scripts import close_superseded_post

    return close_superseded_post


def store():
    from backend.run_execution import ExecutionStore

    return ExecutionStore(storage.engine)


def seed(root, *, sibling_post="done", pdf="live", paid_state="response_saved", post_state="blocked"):
    from backend.clinic_models import ClinicArtifact, ClinicLocation

    with storage.session_scope() as s:
        patient = storage.create_patient(
            s, label="DS_07-31-1957", first_initial="D", last_initial="S", birthdate="07-31-1957"
        )
        patient_uuid = patient.id
        for run_id in ("blocked-run", "sibling-run"):
            s.add(
                storage.Run(
                    id=run_id,
                    patient_id=patient_uuid,
                    report_id="report",
                    status="complete",
                    council_model_ids_json='["original"]',
                    source_report_ids_json='["report"]',
                    analysis_input_fingerprint="input",
                )
            )
        s.commit()
    for run_id in ("blocked-run", "sibling-run"):
        store().request_run_start(run_id)
    manifest = root / "artifacts" / "blocked-run" / "post" / "patient_facing.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text("{}")
    with storage.session_scope() as s:
        s.add(
            storage.PostObligation(
                run_id="blocked-run",
                kind="patient_facing",
                manifest_path=str(manifest),
                manifest_hash="a" * 64,
                owner_token="t",
                owner_generation=1,
                state=post_state,
                blocked_reason="Patient-facing report is missing required sections: ## 4."
                if post_state == "blocked"
                else None,
            )
        )
        s.add(
            storage.PaidRequest(
                run_id="blocked-run",
                scope_key="post/patient_facing/generation",
                dispatch_ordinal=0,
                request_path="req.json",
                request_hash="c" * 64,
                route_json="{}",
                execution_manifest_hash="d" * 64,
                input_fingerprint="input",
                owner_token="t",
                owner_generation=1,
                state=paid_state,
                created_at=storage._utcnow(),
            )
        )
        if sibling_post:
            s.add(
                storage.PostObligation(
                    run_id="sibling-run",
                    kind="patient_facing",
                    manifest_path=str(root / "sibling.json"),
                    manifest_hash="e" * 64,
                    owner_token="t",
                    owner_generation=1,
                    state=sibling_post,
                    receipt_path=str(root / "sibling-receipt.json") if sibling_post == "done" else None,
                    receipt_hash="f" * 64 if sibling_post == "done" else None,
                )
            )
        if pdf:
            s.add(
                ClinicArtifact(
                    id="pdf-artifact",
                    patient_uuid=patient_uuid,
                    source_kind="patient-file",
                    source_id="pdf-source",
                    logical_family="patient-facing",
                    version=1,
                    file_key="DS_07-31-1957__patient-facing__auto-sibling.pdf",
                    original_name="DS_07-31-1957__patient-facing__auto-sibling__2026-09-29.pdf",
                    sha256="9" * 64,
                    size=1234,
                    content_type="application/pdf",
                    document_kind="patient-summary",
                    registered_at=1,
                    provenance_json=json.dumps({"runId": "sibling-run"}),
                    archived=pdf == "archived",
                )
            )
            s.add(
                ClinicLocation(
                    id="pdf-local",
                    artifact_id="pdf-artifact",
                    kind="local",
                    key="local-key",
                    patient_alias="DS_07-31-1957",
                    verified=True,
                    active=True,
                )
            )
        s.commit()
    return manifest.parent / "superseded.json"


def post_row(run_id="blocked-run"):
    with storage.session_scope() as s:
        row = s.get(storage.PostObligation, (run_id, "patient_facing"))
        return row.state, row.blocked_reason, row.receipt_path, row.receipt_hash


def test_a_blocked_write_up_with_a_delivered_sibling_closes_as_superseded(db):
    audit = seed(db)
    saved = script().close_superseded_post(store(), "blocked-run", REASON, actor="lead")
    state, reason, receipt_path, receipt_hash = post_row()
    assert (state, reason) == ("skipped", "superseded: " + REASON)
    assert receipt_path == str(audit)
    assert receipt_hash == hashlib.sha256(audit.read_bytes()).hexdigest()
    record = json.loads(audit.read_text())
    assert record == saved
    assert record["prior_state"] == "blocked"
    assert record["prior_blocked_reason"].startswith("Patient-facing report is missing")
    assert record["superseded_by"]["run_id"] == "sibling-run"
    assert record["superseded_by"]["pdf"]["sha256"] == "9" * 64
    assert record["actor"] == "lead"
    with storage.session_scope() as s:
        run = s.get(storage.Run, "blocked-run")
        assert (run.execution_state, run.blocked_reason, run.owner_token) == ("done", None, None)
        paid = s.scalars(select(storage.PaidRequest.state)).all()
    assert paid == ["response_saved"], "nothing paid is touched"
    with pytest.raises(script().Refused, match="skipped, not blocked"):
        script().close_superseded_post(store(), "blocked-run", REASON, actor="lead")


@pytest.mark.parametrize(
    ("sibling_post", "pdf"),
    [(None, "live"), ("blocked", "live"), ("done", None), ("done", "archived")],
)
def test_without_a_delivered_sibling_nothing_is_closed(db, sibling_post, pdf):
    audit = seed(db, sibling_post=sibling_post, pdf=pdf)
    before = post_row()
    with pytest.raises(script().Refused, match="delivered patient-facing PDF"):
        script().close_superseded_post(store(), "blocked-run", REASON, actor="lead")
    assert post_row() == before
    assert not audit.exists()


@pytest.mark.parametrize("paid_state", ["unknown", "prepared", "dispatched"])
def test_an_unsettled_paid_call_refuses_the_close(db, paid_state):
    audit = seed(db, paid_state=paid_state)
    before = post_row()
    with pytest.raises(script().Refused, match="reconcile it first"):
        script().close_superseded_post(store(), "blocked-run", REASON, actor="lead")
    assert post_row() == before
    assert not audit.exists()


@pytest.mark.parametrize("post_state", ["pending", "done", "skipped"])
def test_a_write_up_that_is_not_blocked_is_refused(db, post_state):
    audit = seed(db, post_state=post_state)
    before = post_row()
    with pytest.raises(script().Refused, match="not blocked"):
        script().close_superseded_post(store(), "blocked-run", REASON, actor="lead")
    assert post_row() == before
    assert not audit.exists()


def test_the_owner_close_is_a_fenced_compare_and_set(db):
    from backend.run_execution import ExecutionConflict

    seed(db)
    owner = store().claim_run_owner("blocked-run", regenerate_post=True)
    assert owner is not None
    try:
        with pytest.raises(ExecutionConflict, match="state changed"):
            owner.close_superseded_post(
                "patient_facing",
                expected_manifest_hash="0" * 64,
                reason=REASON,
                receipt_path="audit.json",
                receipt_hash="1" * 64,
            )
        with pytest.raises(ValueError):
            owner.close_superseded_post(
                "patient_facing",
                expected_manifest_hash="a" * 64,
                reason=" ",
                receipt_path="audit.json",
                receipt_hash="1" * 64,
            )
    finally:
        owner.release(state="pending")
    assert post_row()[0] == "blocked"


def test_the_command_closes_against_an_explicit_data_dir(db):
    audit = seed(db)
    storage.engine.dispose()
    env = {**os.environ, "OPENROUTER_API_KEY": "", "DATA_DIR": ""}
    done = subprocess.run(
        [sys.executable, "-m", "backend.scripts.close_superseded_post",
         "--data-dir", str(db), "blocked-run", REASON],
        cwd=REPO, env=env, capture_output=True, text=True, timeout=120,
    )
    assert done.returncode == 0, done.stderr
    assert "superseded by run sibling-run" in done.stdout
    assert post_row()[0] == "skipped"
    assert audit.exists()
    again = subprocess.run(
        [sys.executable, "-m", "backend.scripts.close_superseded_post",
         "--data-dir", str(db), "blocked-run", REASON],
        cwd=REPO, env=env, capture_output=True, text=True, timeout=120,
    )
    assert again.returncode == 1
    assert "Refused; nothing changed" in again.stderr
