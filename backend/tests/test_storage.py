from pathlib import Path

from backend import storage


def test_storage_init_and_basic_crud(tmp_path: Path):
    storage.reset_engine(f"sqlite:///{tmp_path / 'app.db'}")
    storage.init_db()

    with storage.session_scope() as session:
        p = storage.create_patient(session, label="Test Patient", notes="n")
        assert p.id

        p2 = storage.get_patient(session, p.id)
        assert p2 is not None
        assert p2.label == "Test Patient"

        p3 = storage.update_patient(session, p.id, label="Updated", notes="n2")
        assert p3 is not None
        assert p3.label == "Updated"

        report = storage.create_report(
            session,
            patient_id=p.id,
            filename="r.txt",
            mime_type="text/plain",
            stored_path=tmp_path / "orig.txt",
            extracted_text_path=tmp_path / "extracted.txt",
        )
        assert report.id

        run = storage.create_run(
            session,
            patient_id=p.id,
            report_id=report.id,
            council_model_ids=["m1", "m2"],
            consolidator_model_id="m1",
        )
        assert run.id

        art = storage.create_artifact(
            session,
            run_id=run.id,
            stage_num=1,
            stage_name="initial_analysis",
            model_id="m1",
            kind="analysis",
            content_path=tmp_path / "a.md",
            content_type="text/markdown",
        )
        assert art.id


def test_session_scope_does_not_expire_instances(tmp_path: Path):
    storage.reset_engine(f"sqlite:///{tmp_path / 'app.db'}")
    storage.init_db()

    with storage.session_scope() as session:
        p = storage.create_patient(session, label="P", notes="")
        r = storage.create_report(
            session,
            patient_id=p.id,
            filename="r.txt",
            mime_type="text/plain",
            stored_path=tmp_path / "orig.txt",
            extracted_text_path=tmp_path / "extracted.txt",
        )
        run = storage.create_run(
            session,
            patient_id=p.id,
            report_id=r.id,
            council_model_ids=["m1"],
            consolidator_model_id="m1",
        )

    with storage.session_scope() as session:
        report = storage.get_report(session, r.id)
        assert report is not None
        storage.update_run_status(session, run.id, status="running")

    # Accessing a previously-loaded instance after a commit+session close should not trigger refresh.
    assert isinstance(report.extracted_text_path, str)


EXECUTION_COLUMNS = (
    "start_requested_at",
    "execution_state",
    "owner_token",
    "owner_generation",
    "owner_pid",
    "owner_started_at",
    "next_check_at",
    "blocked_reason",
    "execution_manifest_path",
    "execution_manifest_hash",
)


def test_execution_old_schema_upgrade_twice_is_inactive_and_lossless(tmp_path):
    import sqlite3

    db = tmp_path / "legacy.db"
    storage.reset_engine(f"sqlite:///{db}")
    storage.init_db()
    with storage.session_scope() as session:
        for status in ["created", "running", "complete", "failed", "needs_auth"]:
            session.add(
                storage.Run(
                    id=status,
                    patient_id="p",
                    report_id="r",
                    status=status,
                    council_model_ids_json='["saved-model"]',
                    source_report_ids_json='["r"]',
                    source_manifest_json='{"original":true}',
                    operation_id="op-" + status,
                    analysis_input_fingerprint="input",
                    special_instructions="original note",
                )
            )
        session.add(
            storage.Artifact(
                id="old-artifact",
                run_id="complete",
                stage_num=6,
                stage_name="final",
                model_id="saved-model",
                kind="draft",
                content_path="/original/artifact.md",
                content_type="text/markdown",
            )
        )
        session.add(
            storage.AnalysisInputReservation(
                operation_id="reservation",
                request_fingerprint="request",
                envelope_fingerprint="envelope",
                manifest_json="{}",
                report_id="reserved-report",
                run_id="reserved-run",
            )
        )
        session.add(
            storage.PatientIdReservation(
                patient_id="ZZ_01-01-1900",
                first_initial="Z",
                last_initial="Z",
                birthdate="1900-01-01",
                ordinal=1,
            )
        )
        session.commit()
    storage.engine.dispose()
    with sqlite3.connect(db) as conn:
        for column in EXECUTION_COLUMNS:
            if column in {r[1] for r in conn.execute("PRAGMA table_info(runs)")}:
                conn.execute(f"ALTER TABLE runs DROP COLUMN {column}")
        old_columns = [r[1] for r in conn.execute("PRAGMA table_info(runs)")]
        before = conn.execute("SELECT * FROM runs ORDER BY id").fetchall()
        snapshots = {
            table: conn.execute(f"SELECT * FROM {table}").fetchall()
            for table in [
                "artifacts",
                "analysis_input_reservations",
                "patient_id_reservations",
            ]
        }
        for table in ["paid_requests", "stage_receipts", "post_obligations"]:
            conn.execute(f"DROP TABLE IF EXISTS {table}")
    storage.init_db()
    storage.init_db()
    with sqlite3.connect(db) as conn:
        present = {r[1] for r in conn.execute("PRAGMA table_info(runs)")}
        assert set(EXECUTION_COLUMNS) <= present
        assert (
            conn.execute(
                f"SELECT {','.join(old_columns)} FROM runs ORDER BY id"
            ).fetchall()
            == before
        )
        for table, rows in snapshots.items():
            assert conn.execute(f"SELECT * FROM {table}").fetchall() == rows
        assert (
            conn.execute(
                "SELECT start_requested_at, execution_state, owner_generation FROM runs"
            ).fetchall()
            == [(None, None, 0)] * 5
        )


def test_execution_receipt_composite_identities_are_unique(tmp_path):
    import pytest
    from sqlalchemy.exc import IntegrityError

    storage.reset_engine(f"sqlite:///{tmp_path / 'receipts.db'}")
    storage.init_db()
    for name in ["PaidRequest", "StageReceipt", "PostObligation"]:
        assert hasattr(storage, name), f"{name} storage absent"
    cases = [
        (
            storage.PaidRequest,
            dict(
                run_id="r",
                scope_key="s1/member/0",
                dispatch_ordinal=0,
                request_path="req",
                request_hash="a" * 64,
                route_json="{}",
                execution_manifest_hash="b" * 64,
                input_fingerprint="input",
                owner_token="owner",
                owner_generation=1,
            ),
        ),
        (
            storage.StageReceipt,
            dict(
                run_id="r",
                stage_num=1,
                receipt_path="stage",
                receipt_hash="a" * 64,
                execution_manifest_hash="b" * 64,
                input_fingerprint="input",
                policy_version="1",
                owner_token="owner",
                owner_generation=1,
            ),
        ),
        (
            storage.PostObligation,
            dict(
                run_id="r",
                kind="patient_facing",
                manifest_path="post",
                manifest_hash="a" * 64,
                owner_token="owner",
                owner_generation=1,
            ),
        ),
    ]
    for model, fields in cases:
        with storage.session_scope() as session:
            session.add(model(**fields))
            session.commit()
        with pytest.raises(IntegrityError):
            with storage.session_scope() as session:
                session.add(model(**fields))
                session.commit()
        with storage.session_scope() as session:
            session.add(model(**{**fields, "run_id": "independent"}))
            session.commit()
