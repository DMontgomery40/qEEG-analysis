"""Close a blocked patient write-up that another run of the same patient
already delivered, without paying to generate it again.

The post becomes skipped with blocked_reason "superseded: <reason>"; an
immutable superseded.json beside its manifest records why and which delivered
PDF supersedes it, and the post's receipt binds that file.
"""

import argparse
from datetime import datetime, timezone
import getpass
import hashlib
import json
import os
from pathlib import Path
import sys


class Refused(RuntimeError):
    """The run is not eligible; nothing was written."""


def _delivered_sibling(session, run):
    """The newest other run of this patient whose write-up is done and whose
    patient-facing PDF is in the catalogue, not archived, with a live copy."""
    from sqlalchemy import select
    from backend import storage
    from backend.clinic_models import ClinicArtifact, ClinicLocation

    siblings = session.scalars(
        select(storage.Run)
        .join(
            storage.PostObligation,
            storage.PostObligation.run_id == storage.Run.id,
        )
        .where(
            storage.Run.patient_id == run.patient_id,
            storage.Run.id != run.id,
            storage.PostObligation.kind == "patient_facing",
            storage.PostObligation.state == "done",
            storage.PostObligation.receipt_hash.is_not(None),
        )
        .order_by(storage.Run.created_at.desc())
    ).all()
    for sibling in siblings:
        for artifact in session.scalars(
            select(ClinicArtifact)
            .where(
                ClinicArtifact.patient_uuid == run.patient_id,
                ClinicArtifact.document_kind == "patient-summary",
                ClinicArtifact.content_type == "application/pdf",
                ClinicArtifact.archived.is_(False),
            )
            .order_by(ClinicArtifact.registered_at.desc())
        ):
            if json.loads(artifact.provenance_json or "{}").get("runId") != sibling.id:
                continue
            live = session.scalar(
                select(ClinicLocation.id).where(
                    ClinicLocation.artifact_id == artifact.id,
                    ClinicLocation.active.is_(True),
                )
            )
            if live is not None:
                return sibling, artifact
    return None, None


def _write_once(path, data):
    """Create the audit record, or accept it unchanged from an interrupted
    earlier close of the same post."""
    try:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    except FileExistsError:
        return path.read_bytes()
    with os.fdopen(fd, "wb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    directory = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)
    return data


def close_superseded_post(store, run_id, reason, *, actor):
    from sqlalchemy import select
    from backend import storage
    from backend.run_execution import UNSETTLED_PAID_STATES

    reason = (reason or "").strip()
    if not reason:
        raise Refused("Say why this write-up is superseded.")
    with storage.session_scope() as session:
        run = session.get(storage.Run, run_id)
        if run is None:
            raise Refused(f"No run {run_id}.")
        post = session.get(storage.PostObligation, (run_id, "patient_facing"))
        if post is None or post.state != "blocked":
            raise Refused(
                f"The write-up for {run_id} is "
                f"{post.state if post else 'absent'}, not blocked."
            )
        unsettled = session.scalar(
            select(storage.PaidRequest)
            .where(
                storage.PaidRequest.run_id == run_id,
                storage.PaidRequest.state.in_(UNSETTLED_PAID_STATES),
            )
            .limit(1)
        )
        if unsettled is not None:
            raise Refused(
                f"Paid call {unsettled.scope_key} is {unsettled.state}; "
                "reconcile it first."
            )
        sibling, pdf = _delivered_sibling(session, run)
        if sibling is None:
            raise Refused(
                "No other run of this patient has a delivered patient-facing PDF."
            )
        patient = session.get(storage.Patient, run.patient_id)
        record = dict(
            schema_version=1,
            kind="superseded_post",
            run_id=run_id,
            post_kind="patient_facing",
            patient_id=patient.label if patient else None,
            manifest_path=post.manifest_path,
            manifest_hash=post.manifest_hash,
            prior_state=post.state,
            prior_blocked_reason=post.blocked_reason,
            reason=reason,
            superseded_by=dict(
                run_id=sibling.id,
                pdf=dict(
                    fileId=pdf.id,
                    originalName=pdf.original_name,
                    sha256=pdf.sha256,
                    size=pdf.size,
                ),
            ),
            actor=actor,
            closed_at=datetime.now(timezone.utc).isoformat(),
        )
        manifest_hash = post.manifest_hash
        audit = Path(post.manifest_path).parent / "superseded.json"
        prior_run = (run.execution_state, run.blocked_reason)
    owner = store.claim_run_owner(run_id, regenerate_post=True)
    if owner is None:
        raise Refused(f"Run {run_id} is busy or no longer eligible.")
    try:
        with owner.file_guard():
            data = _write_once(
                audit, (json.dumps(record, indent=2, sort_keys=True) + "\n").encode()
            )
        saved = json.loads(data)
        if any(
            saved.get(k) != record[k]
            for k in ("run_id", "post_kind", "manifest_hash", "reason")
        ):
            raise Refused(f"{audit} already records a different close.")
        owner.close_superseded_post(
            "patient_facing",
            expected_manifest_hash=manifest_hash,
            reason=reason,
            receipt_path=str(audit),
            receipt_hash=hashlib.sha256(data).hexdigest(),
        )
    except BaseException:
        # Leave the run as it was found: blocked stays blocked with its reason.
        state, why = prior_run
        if state == "blocked" and why:
            owner.release(state="blocked", blocked_reason=why)
        else:
            owner.release(state="pending")
        raise
    from backend.run_execution import ExecutionConflict

    try:
        owner.release(state="done")
    except ExecutionConflict:
        # Something else is still open on this run; the ordinary consumer
        # finishes it, and the write-up stays closed.
        owner.release(state="pending")
    return saved


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        required=True,
        type=Path,
        help="Existing engine data directory containing app.db",
    )
    parser.add_argument("run_id", help="Run whose blocked write-up is superseded")
    parser.add_argument("reason", help="Why, in plain words")
    args = parser.parse_args()
    data_dir = args.data_dir.resolve()
    if not (data_dir / "app.db").is_file():
        parser.error("data directory must contain an existing app.db")
    # Select storage before importing the engine configuration. No provider
    # call, admission or new paid work is created by this command.
    os.environ["DATA_DIR"] = str(data_dir)
    from backend import storage
    from backend.run_execution import ExecutionStore

    try:
        saved = close_superseded_post(
            ExecutionStore(storage.engine), args.run_id, args.reason, actor=getpass.getuser()
        )
    except Refused as error:
        print(f"Refused; nothing changed: {error}", file=sys.stderr)
        return 1
    except Exception as error:
        print(f"Close stopped; the write-up stays blocked: {error}", file=sys.stderr)
        return 1
    print(
        f"Closed the write-up for {args.run_id} as superseded by run "
        f"{saved['superseded_by']['run_id']} ({saved['superseded_by']['pdf']['originalName']})."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
