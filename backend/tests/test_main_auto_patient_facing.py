from __future__ import annotations

import json
import uuid
from pathlib import Path

import pytest


@pytest.mark.asyncio
async def test_auto_patient_facing_runs_for_completed_run(temp_data_dir, monkeypatch):
    from backend import main, storage

    patient_label = "HT_09-05-1954"
    report_id = str(uuid.uuid4())

    with storage.session_scope() as session:
        patient = storage.create_patient(session, label=patient_label, notes="")
        report_dir = Path(temp_data_dir) / "reports" / patient.id / report_id
        report_dir.mkdir(parents=True, exist_ok=True)
        stored_path = report_dir / "original.txt"
        extracted_path = report_dir / "extracted.txt"
        stored_path.write_text("dummy", encoding="utf-8")
        extracted_path.write_text("dummy", encoding="utf-8")
        storage.create_report(
            session,
            report_id=report_id,
            patient_id=patient.id,
            filename="original.txt",
            mime_type="text/plain",
            stored_path=stored_path,
            extracted_text_path=extracted_path,
        )
        run = storage.create_run(
            session,
            patient_id=patient.id,
            report_id=report_id,
            council_model_ids=["mock-council-a"],
            consolidator_model_id="mock-consolidator",
        )
        run_id = run.id
        storage.update_run_status(session, run_id, status="complete")
        peer_path = (
            Path(temp_data_dir)
            / "artifacts"
            / run.id
            / "stage-2"
            / "mock-council-a.json"
        )
        peer_path.parent.mkdir(parents=True, exist_ok=True)
        peer_path.write_text("{}", encoding="utf-8")
        storage.create_artifact(
            session,
            run_id=run.id,
            stage_num=2,
            stage_name="peer_review",
            model_id="mock-council-a",
            kind="peer_review",
            content_path=peer_path,
            content_type="application/json",
        )
        revision_path = (
            Path(temp_data_dir) / "artifacts" / run.id / "stage-3" / "mock-council-a.md"
        )
        revision_path.parent.mkdir(parents=True, exist_ok=True)
        revision_path.write_text("# Revision", encoding="utf-8")
        storage.create_artifact(
            session,
            run_id=run.id,
            stage_num=3,
            stage_name="revision",
            model_id="mock-council-a",
            kind="revision",
            content_path=revision_path,
            content_type="text/markdown",
        )
        final_path = (
            Path(temp_data_dir) / "artifacts" / run.id / "stage-6" / "mock-council-a.md"
        )
        final_path.parent.mkdir(parents=True, exist_ok=True)
        final_path.write_text("# Final", encoding="utf-8")
        storage.create_artifact(
            session,
            run_id=run.id,
            stage_num=6,
            stage_name="final_draft",
            model_id="mock-council-a",
            kind="final_draft",
            content_path=final_path,
            content_type="text/markdown",
        )

    class _DummyBroker:
        def __init__(self):
            self.events: list[dict] = []

        async def publish(self, _run_id: str, payload: dict) -> None:
            self.events.append(payload)

    async def forbidden_subprocess(*args, **kwargs):
        pytest.fail("Legacy helper must use original owned post admission")

    monkeypatch.setenv("QEEG_AUTO_PATIENT_FACING", "1")
    monkeypatch.setattr(main.asyncio, "create_subprocess_exec", forbidden_subprocess)
    with storage.session_scope() as session:
        consolidation_path = Path(temp_data_dir) / "consolidation.md"
        consolidation_path.write_text("# Original consolidation", encoding="utf-8")
        storage.create_artifact(
            session,
            run_id=run_id,
            stage_num=4,
            stage_name="consolidation",
            model_id="mock-council-a",
            kind="consolidation",
            content_path=consolidation_path,
            content_type="text/markdown",
        )
    broker = _DummyBroker()
    completed = await main._auto_generate_patient_facing_for_run(run_id, broker)
    assert completed is False  # Admission is pending, never fabricated completion.
    with storage.session_scope() as session:
        obligation = session.get(storage.PostObligation, (run_id, "patient_facing"))
        assert obligation is not None and obligation.state == "pending"
    assert any(e.get("status")=="pending" for e in broker.events)


@pytest.mark.asyncio
async def test_auto_patient_facing_skips_unreviewed_partial_run(
    temp_data_dir, monkeypatch
):
    from backend import main, storage
    from backend.orchestration import progress_jsonl_path

    patient_label = "HT_09-05-1954"
    report_id = str(uuid.uuid4())

    with storage.session_scope() as session:
        patient = storage.create_patient(session, label=patient_label, notes="")
        report_dir = Path(temp_data_dir) / "reports" / patient.id / report_id
        report_dir.mkdir(parents=True, exist_ok=True)
        stored_path = report_dir / "original.txt"
        extracted_path = report_dir / "extracted.txt"
        stored_path.write_text("dummy", encoding="utf-8")
        extracted_path.write_text("dummy", encoding="utf-8")
        storage.create_report(
            session,
            report_id=report_id,
            patient_id=patient.id,
            filename="original.txt",
            mime_type="text/plain",
            stored_path=stored_path,
            extracted_text_path=extracted_path,
        )
        run = storage.create_run(
            session,
            patient_id=patient.id,
            report_id=report_id,
            council_model_ids=["deepseek-v4-flash", "gpt-5.5", "claude-sonnet-4-6"],
            consolidator_model_id="gpt-5.5",
        )
        run_id = run.id
        storage.update_run_status(session, run_id, status="complete")

    progress_path = progress_jsonl_path(run_id)
    progress_path.parent.mkdir(parents=True, exist_ok=True)
    progress_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "run_id": run_id,
                        "stage_num": 2,
                        "stage_name": "peer_review",
                        "status": "complete",
                        "skipped": True,
                    }
                ),
                json.dumps(
                    {
                        "run_id": run_id,
                        "status": "complete",
                        "success_count": 2,
                        "requested_count": 3,
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    class _DummyBroker:
        def __init__(self):
            self.events: list[dict] = []

        async def publish(self, _run_id: str, payload: dict) -> None:
            self.events.append(payload)

    async def fail_if_called(*_args, **_kwargs):
        raise AssertionError("patient-facing subprocess should not start")

    monkeypatch.setattr(main.asyncio, "create_subprocess_exec", fail_if_called)

    broker = _DummyBroker()
    completed = await main._auto_generate_patient_facing_for_run(run_id, broker)

    assert completed is False
    assert any(
        e.get("stage_name") == "patient_facing" and e.get("status") == "skipped"
        for e in broker.events
    )


@pytest.mark.asyncio
async def test_auto_patient_facing_returns_false_on_subprocess_failure(
    temp_data_dir, monkeypatch
):
    from backend import main, storage

    patient_label = "HT_09-05-1954"
    report_id = str(uuid.uuid4())

    with storage.session_scope() as session:
        patient = storage.create_patient(session, label=patient_label, notes="")
        report_dir = Path(temp_data_dir) / "reports" / patient.id / report_id
        report_dir.mkdir(parents=True, exist_ok=True)
        stored_path = report_dir / "original.txt"
        extracted_path = report_dir / "extracted.txt"
        stored_path.write_text("dummy", encoding="utf-8")
        extracted_path.write_text("dummy", encoding="utf-8")
        storage.create_report(
            session,
            report_id=report_id,
            patient_id=patient.id,
            filename="original.txt",
            mime_type="text/plain",
            stored_path=stored_path,
            extracted_text_path=extracted_path,
        )
        run = storage.create_run(
            session,
            patient_id=patient.id,
            report_id=report_id,
            council_model_ids=["mock-council-a"],
            consolidator_model_id="mock-consolidator",
        )
        run_id = run.id
        storage.update_run_status(session, run_id, status="complete")
        peer_path = (
            Path(temp_data_dir)
            / "artifacts"
            / run.id
            / "stage-2"
            / "mock-council-a.json"
        )
        peer_path.parent.mkdir(parents=True, exist_ok=True)
        peer_path.write_text("{}", encoding="utf-8")
        storage.create_artifact(
            session,
            run_id=run.id,
            stage_num=2,
            stage_name="peer_review",
            model_id="mock-council-a",
            kind="peer_review",
            content_path=peer_path,
            content_type="application/json",
        )
        revision_path = (
            Path(temp_data_dir) / "artifacts" / run.id / "stage-3" / "mock-council-a.md"
        )
        revision_path.parent.mkdir(parents=True, exist_ok=True)
        revision_path.write_text("# Revision", encoding="utf-8")
        storage.create_artifact(
            session,
            run_id=run.id,
            stage_num=3,
            stage_name="revision",
            model_id="mock-council-a",
            kind="revision",
            content_path=revision_path,
            content_type="text/markdown",
        )
        final_path = (
            Path(temp_data_dir) / "artifacts" / run.id / "stage-6" / "mock-council-a.md"
        )
        final_path.parent.mkdir(parents=True, exist_ok=True)
        final_path.write_text("# Final", encoding="utf-8")
        storage.create_artifact(
            session,
            run_id=run.id,
            stage_num=6,
            stage_name="final_draft",
            model_id="mock-council-a",
            kind="final_draft",
            content_path=final_path,
            content_type="text/markdown",
        )

    class _DummyBroker:
        def __init__(self):
            self.events: list[dict] = []

        async def publish(self, _run_id: str, payload: dict) -> None:
            self.events.append(payload)

    from backend import patient_postprocessing

    def admission_unavailable(*args, **kwargs):
        raise RuntimeError("Original owned admission unavailable")

    monkeypatch.setattr(
        patient_postprocessing, "admit_patient_facing", admission_unavailable
    )

    broker = _DummyBroker()
    completed = await main._auto_generate_patient_facing_for_run(run_id, broker)

    assert completed is False
    assert any(
        e.get("stage_name") == "patient_facing" and e.get("status") == "failed"
        for e in broker.events
    )


@pytest.mark.asyncio
async def test_auto_cathode_video_prepares_handoff_and_spawns_queue(
    temp_data_dir, monkeypatch, tmp_path
):
    from backend import main, storage

    patient_label = "HT_09-05-1954"
    report_id = str(uuid.uuid4())

    with storage.session_scope() as session:
        patient = storage.create_patient(session, label=patient_label, notes="")
        report_dir = Path(temp_data_dir) / "reports" / patient.id / report_id
        report_dir.mkdir(parents=True, exist_ok=True)
        stored_path = report_dir / "original.txt"
        extracted_path = report_dir / "extracted.txt"
        stored_path.write_text("dummy", encoding="utf-8")
        extracted_path.write_text("dummy", encoding="utf-8")
        storage.create_report(
            session,
            report_id=report_id,
            patient_id=patient.id,
            filename="original.txt",
            mime_type="text/plain",
            stored_path=stored_path,
            extracted_text_path=extracted_path,
        )
        run = storage.create_run(
            session,
            patient_id=patient.id,
            report_id=report_id,
            council_model_ids=["mock-council-a"],
            consolidator_model_id="mock-consolidator",
        )
        storage.update_run_status(session, run.id, status="complete")
        peer_path = (
            Path(temp_data_dir)
            / "artifacts"
            / run.id
            / "stage-2"
            / "mock-council-a.json"
        )
        peer_path.parent.mkdir(parents=True, exist_ok=True)
        peer_path.write_text("{}", encoding="utf-8")
        storage.create_artifact(
            session,
            run_id=run.id,
            stage_num=2,
            stage_name="peer_review",
            model_id="mock-council-a",
            kind="peer_review",
            content_path=peer_path,
            content_type="application/json",
        )
        revision_path = (
            Path(temp_data_dir) / "artifacts" / run.id / "stage-3" / "mock-council-a.md"
        )
        revision_path.parent.mkdir(parents=True, exist_ok=True)
        revision_path.write_text("# Revision", encoding="utf-8")
        storage.create_artifact(
            session,
            run_id=run.id,
            stage_num=3,
            stage_name="revision",
            model_id="mock-council-a",
            kind="revision",
            content_path=revision_path,
            content_type="text/markdown",
        )
        artifact_path = (
            Path(temp_data_dir) / "artifacts" / run.id / "stage-4" / "consolidation.md"
        )
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        artifact_path.write_text(
            "# Consolidation\nusable for cathode", encoding="utf-8"
        )
        storage.create_artifact(
            session,
            run_id=run.id,
            stage_num=4,
            stage_name="consolidation",
            model_id="mock-consolidator",
            kind="consolidation",
            content_path=artifact_path,
            content_type="text/markdown",
        )
        run_id = run.id

    cathode_root = tmp_path / "cathode"
    queue_script = cathode_root / "scripts" / "qeeg_patient_video_queue.py"
    queue_script.parent.mkdir(parents=True)
    queue_script.write_text("#!/usr/bin/env python\n", encoding="utf-8")
    cathode_projects = cathode_root / "projects"
    monkeypatch.setenv("QEEG_CATHODE_PROJECTS_DIR", str(cathode_projects))

    class _DummyBroker:
        def __init__(self):
            self.events: list[dict] = []

        async def publish(self, _run_id: str, payload: dict) -> None:
            self.events.append(payload)

    class _Proc:
        pid = 12345

    called: dict[str, object] = {}

    def fake_popen(cmd, **kwargs):
        called["cmd"] = cmd
        called["kwargs"] = kwargs
        return _Proc()

    monkeypatch.setattr(
        main, "_repo_root", lambda: cathode_root.parent / "qEEG-analysis"
    )
    monkeypatch.setattr(main.subprocess, "Popen", fake_popen)

    broker = _DummyBroker()
    await main._auto_generate_cathode_video_for_run(run_id, broker)

    project_dir = cathode_projects / patient_label
    payload = json.loads(
        (project_dir / "qeeg_handoff_payload.json").read_text(encoding="utf-8")
    )
    assert payload["target_length_minutes"] == 6.5
    assert "usable for cathode" in (project_dir / "qeeg_council_source.md").read_text(
        encoding="utf-8"
    )
    assert called["cmd"][called["cmd"].index("--patients") + 1] == patient_label
    assert "--rebuild-storyboard" in called["cmd"]
    assert "--skip-scene-review" in called["cmd"]
    assert any(
        e.get("stage_name") == "cathode_video" and e.get("status") == "queued"
        for e in broker.events
    )


def test_owned_completion_retires_automatic_cathode_without_dispatch(
    temp_data_dir, monkeypatch
):
    from types import SimpleNamespace
    from backend import main, patient_postprocessing as post, storage
    from backend.council import completion
    from backend.tests.test_patient_postprocessing import ready as ready_fixture
    from sqlalchemy import select

    store, run_id, cfg = ready_fixture.__wrapped__(temp_data_dir, monkeypatch)
    cfg["retired_cathode_flag"] = "1"

    def forbidden(*args, **kwargs):
        raise AssertionError(
            "new owned completion launched a legacy automatic dispatcher"
        )

    monkeypatch.setattr(main, "_auto_generate_cathode_video_for_run", forbidden)
    store.request_run_start(run_id)
    owner = store.claim_run_owner(run_id)
    monkeypatch.setattr(
        completion,
        "current_execution",
        lambda: SimpleNamespace(owner=owner, manifest={"postprocessing": cfg}),
    )
    try:
        completion.project_run_status(None, run_id, status="complete")
        with owner.transaction() as session:
            cathode = session.get(storage.PostObligation, (run_id, "cathode"))
            assert cathode.state == "skipped"
            assert not list(session.scalars(select(storage.PaidRequest)))
        assert (
            post._load(cathode.manifest_path)["diagnostic"]
            == "automatic_cathode_routing_retired"
        )
    finally:
        owner.release()
