"""One misfiled report moves to its own chart: planned first, applied as one
unit, put back whole when the database step fails, refused when analyses
already rest on it. The live case: a woman's report (printing "Female,
9/5/1954") filed on DM_09-23-1982 belongs on MF_09-05-1954."""

from __future__ import annotations

import hashlib
import importlib
import json
from pathlib import Path

import pytest
from sqlalchemy import text

from backend import chart_consistency as cc
from backend import storage
from backend.scripts import reassign_report as tool
from backend.tests.test_chart_consistency import wavi

NAME = "MF_MCI_20 tx_final.txt"


def _file(key, first, last, birthdate, body):
    intake = importlib.import_module("backend.clinic_intake")
    upload = intake.submit_upload(
        key=key, identity={"firstName": first, "lastName": last, "birthdate": birthdate},
        files=[(NAME, body.encode(), "text/plain")], file_meta=[{"documentKind": "report"}], actor="Staff",
    )["upload"]
    return upload["items"][0]["sourceId"]


@pytest.fixture
def misfiled(temp_data_dir):
    _file("dm", "Dan", "Moore", "09-23-1982", wavi("Male, 9/23/1982", (("8/11/2025", 42),)))
    report_id = _file("mf-on-dm", "Dan", "Moore", "09-23-1982",
                      wavi("Female, 9/5/1954", (("12/15/2025", 71),), rt=301))
    # Her own chart already holds a report in the same catalogue family.
    _file("mf", "Mary", "Fox", "09-05-1954", wavi("Female, 9/5/1954", (("1/19/2026", 71),), rt=305))
    return report_id


def _row(report_id):
    with storage.session_scope() as s:
        return s.execute(text(
            "SELECT p.label, r.stored_path FROM reports r JOIN patients p ON p.id = r.patient_id WHERE r.id = :r"
        ), {"r": report_id}).one()


def _violations(temp_data_dir):
    found = cc.check(Path(storage.engine.url.database), root=temp_data_dir, data_dir=temp_data_dir, cache_path=None)
    return [(v["kind"], v["patients"]) for v in found["violations"]]


def test_the_plan_moves_nothing_and_names_everything(misfiled, temp_data_dir):
    before = _row(misfiled)
    plan = tool.plan_reassign(misfiled[:8], "MF_09-05-1954")
    assert plan["report"] == {"id": misfiled, "file": NAME, "from": "DM_09-23-1982", "to": "MF_09-05-1954"}
    assert any(f["path"] == "original.txt" or f["path"].startswith("original") for f in plan["folder"]["files"])
    assert plan["artifacts"][0]["version_to"] == 2, "her chart already has version 1 of this family"
    assert len(plan["locations"]) == 1
    assert _row(misfiled) == before
    assert Path(plan["folder"]["from"]).is_dir() and not Path(plan["folder"]["to"]).exists()


def test_apply_moves_the_report_whole_and_the_chart_agrees_again(misfiled, temp_data_dir, tmp_path):
    assert ("birthday", ["DM_09-23-1982"]) in _violations(temp_data_dir)
    plan = tool.plan_reassign(misfiled, "MF_09-05-1954")
    audit = tool.apply_reassign(plan, tmp_path / "audit.json")
    label, stored = _row(misfiled)
    assert label == "MF_09-05-1954"
    folder = Path(plan["folder"]["to"])
    for entry in plan["folder"]["files"]:
        assert hashlib.sha256((folder / entry["path"]).read_bytes()).hexdigest() == entry["sha256"]
    assert Path(stored).parent == folder and not Path(plan["folder"]["from"]).exists()
    with storage.session_scope() as s:
        key, alias = s.execute(text("SELECT key, patient_alias FROM clinic_locations WHERE id = :id"),
                               {"id": plan["locations"][0]["id"]}).one()
        version = s.execute(text("SELECT version FROM clinic_artifacts WHERE id = :id"),
                            {"id": plan["artifacts"][0]["id"]}).scalar()
    assert Path(key).parent == folder and alias == "MF_09-05-1954" and version == 2
    assert json.loads((tmp_path / "audit.json").read_text())["finished_at"] == audit["finished_at"]
    assert not [v for v in _violations(temp_data_dir) if v[0] in ("birthday", "sex")]


def test_a_failed_database_step_puts_the_folder_back(misfiled, temp_data_dir, tmp_path, monkeypatch):
    before = _row(misfiled)
    plan = tool.plan_reassign(misfiled, "MF_09-05-1954")

    def broken(plan):
        raise RuntimeError("database locked")

    monkeypatch.setattr(tool, "_commit_rows", broken)
    with pytest.raises(RuntimeError):
        tool.apply_reassign(plan, tmp_path / "audit.json")
    assert _row(misfiled) == before
    source = Path(plan["folder"]["from"])
    for entry in plan["folder"]["files"]:
        assert hashlib.sha256((source / entry["path"]).read_bytes()).hexdigest() == entry["sha256"]
    assert not Path(plan["folder"]["to"]).exists()
    rollback = json.loads((tmp_path / "audit.json").read_text())["rollback"]
    assert rollback["folder"] == "returned" and rollback["changed"] == []


def _run(report_id, status):
    with storage.session_scope() as s:
        patient = s.execute(text("SELECT patient_id FROM reports WHERE id = :r"), {"r": report_id}).scalar()
        run = storage.create_run(s, patient_id=patient, report_id=report_id,
                                 council_model_ids=["m"], consolidator_model_id="m")
        run_id = run.id
    with storage.session_scope() as s:
        s.execute(text("UPDATE runs SET status = :s WHERE id = :id"), {"s": status, "id": run_id})
        s.commit()
    return run_id


def test_a_report_a_completed_run_rests_on_stays_on_its_chart(misfiled, temp_data_dir):
    _run(misfiled, "complete")
    with pytest.raises(tool.ReassignRefused, match="completed"):
        tool.plan_reassign(misfiled, "MF_09-05-1954")


def test_a_failed_run_on_only_this_report_moves_with_it(misfiled, temp_data_dir, tmp_path):
    run_id = _run(misfiled, "failed")
    plan = tool.plan_reassign(misfiled, "MF_09-05-1954")
    assert plan["runs"] == [run_id]
    tool.apply_reassign(plan, tmp_path / "audit.json")
    with storage.session_scope() as s:
        assert s.execute(text("SELECT p.label FROM runs r JOIN patients p ON p.id = r.patient_id "
                              "WHERE r.id = :id"), {"id": run_id}).scalar() == "MF_09-05-1954"


def test_apply_without_the_exact_confirmation_moves_nothing(misfiled, temp_data_dir, capsys):
    before = _row(misfiled)
    assert tool.main(["--report", misfiled, "--to", "MF_09-05-1954", "--apply",
                      "--yes-reassign", f"{misfiled}:CV_02-12-2012"]) == 2
    assert _row(misfiled) == before
    assert tool.main(["--report", misfiled, "--to", "DM_09-23-1982"]) == 2, "already there"
