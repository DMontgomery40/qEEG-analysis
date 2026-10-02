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
