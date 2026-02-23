from __future__ import annotations

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
    monkeypatch.setattr(main.asyncio, "create_subprocess_exec", fake_create_subprocess_exec)

    broker = _DummyBroker()
    await main._auto_generate_patient_facing_for_run(run_id, broker)

    assert "args" in called
    assert "--patient-label" in called["args"]
    assert patient_label in called["args"]
    assert any(e.get("stage_name") == "patient_facing" and e.get("status") == "start" for e in broker.events)
    assert any(e.get("stage_name") == "patient_facing" and e.get("status") == "complete" for e in broker.events)
