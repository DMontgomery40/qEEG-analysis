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
