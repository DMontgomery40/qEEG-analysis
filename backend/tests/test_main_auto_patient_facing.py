from __future__ import annotations

import json
import uuid
from pathlib import Path

import pytest


@pytest.mark.asyncio
async def test_auto_patient_facing_runs_for_completed_run(temp_data_dir, monkeypatch):
    from backend import main, storage

    patient_label = "09-05-1954-0"
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

    class _Proc:
        returncode = 0

        async def communicate(self):
            return b"ok", b""

    called: dict[str, tuple] = {}

    async def fake_create_subprocess_exec(*args, **kwargs):
        called["args"] = args
        called["kwargs"] = kwargs
        return _Proc()

    monkeypatch.setenv("QEEG_AUTO_PATIENT_FACING", "1")
    monkeypatch.setenv("QEEG_PATIENT_FACING_MODEL", "claude-opus-4-6")
    monkeypatch.setattr(
        main.asyncio, "create_subprocess_exec", fake_create_subprocess_exec
    )

    broker = _DummyBroker()
    completed = await main._auto_generate_patient_facing_for_run(run_id, broker)

    assert completed is True
    assert "args" in called
    assert "--patient-label" in called["args"]
    assert patient_label in called["args"]
    max_tokens_index = called["args"].index("--max-tokens")
    assert called["args"][max_tokens_index + 1] == "12000"
    assert any(
        e.get("stage_name") == "patient_facing" and e.get("status") == "start"
        for e in broker.events
    )
    assert any(
        e.get("stage_name") == "patient_facing" and e.get("status") == "complete"
        for e in broker.events
    )


@pytest.mark.asyncio
async def test_auto_patient_facing_skips_unreviewed_partial_run(
    temp_data_dir, monkeypatch
):
    from backend import main, storage
    from backend.orchestration import progress_jsonl_path

    patient_label = "09-05-1954-0"
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

    patient_label = "09-05-1954-0"
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

    class _Proc:
        returncode = 7

        async def communicate(self):
            return b"", b"model unavailable"

    async def fake_create_subprocess_exec(*_args, **_kwargs):
        return _Proc()

    monkeypatch.setattr(
        main.asyncio, "create_subprocess_exec", fake_create_subprocess_exec
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

    patient_label = "09-05-1954-0"
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
