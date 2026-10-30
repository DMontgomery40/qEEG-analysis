"""A chart relabelled in the catalogue must not leave its files under the old folder name.

2026-10-19: BT_12-11-1963, XX_07-05-1990 and JG_06-28-1977 were relabelled in the
catalogue (so the hub showed the right IDs) while their files stayed in folders
named CL_12-11-1963, XX_01-01-1991 and XX_06-28-1977.
"""
from __future__ import annotations

import json

import pytest
from sqlalchemy import text

from backend import clinic_catalogue, storage
from backend.portal_sync import portal_patients_dir
from backend.scripts import realign_patient_folder as realign


def _chart(label):
    with storage.session_scope() as s:
        return storage.create_patient(s, label=label, first_name="B", last_name="T", first_initial="B",
                                      last_initial="T", birthdate=label[3:]).id


def _file(uuid, folder, rel, data):
    path = portal_patients_dir() / folder / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return clinic_catalogue.register_artifact(
        patient_uuid=uuid, source_kind="renderer", source_id=f"{folder}/{rel}",
        logical_family="video", original_name=path.name, local_path=path)


def _keys(uuid):
    with storage.session_scope() as s:
        return sorted(s.execute(text(
            "SELECT l.key FROM clinic_locations l JOIN clinic_artifacts a ON a.id = l.artifact_id "
            "WHERE a.patient_uuid = :u AND l.kind = 'local' AND l.active = 1 AND l.key LIKE :p"),
            {"u": uuid, "p": str(portal_patients_dir().resolve()) + "/%"}).scalars())


def test_a_relabelled_charts_files_move_to_its_own_folder_with_their_locations(temp_data_dir, tmp_path):
    bt = _chart("BT_12-11-1963")
    _file(bt, "CL_12-11-1963", "CL_12-11-1963__explainer.mp4", b"video")
    _file(bt, "CL_12-11-1963", "council/run/final.md", b"report")
    (portal_patients_dir() / "CL_12-11-1963" / "$meta.json").write_text("{}")
    plan = realign.plan_realign("BT_12-11-1963")
    assert plan["folders"] == ["CL_12-11-1963"] and len(plan["moves"]) == 2
    audit = realign.apply_realign(plan, tmp_path / "audit.json")
    portal = portal_patients_dir().resolve()
    assert (portal / "BT_12-11-1963" / "CL_12-11-1963__explainer.mp4").read_bytes() == b"video"
    assert (portal / "BT_12-11-1963" / "council/run/final.md").read_bytes() == b"report"
    assert not (portal / "CL_12-11-1963").exists()
    assert all("/BT_12-11-1963/" in key for key in _keys(bt))
    assert json.loads((tmp_path / "audit.json").read_text())["chart"] == "BT_12-11-1963"
    assert audit["bookkeeping"] and realign.plan_realign("BT_12-11-1963")["nothing_to_do"]


def test_a_folder_holding_another_charts_file_is_refused_untouched(temp_data_dir):
    bt = _chart("BT_12-11-1963")
    other = _chart("CL_12-11-1964")
    _file(bt, "CL_12-11-1963", "a.mp4", b"bt")
    _file(other, "CL_12-11-1963", "b.mp4", b"someone else")
    with pytest.raises(realign.RealignRefused):
        realign.plan_realign("BT_12-11-1963")
    assert (portal_patients_dir() / "CL_12-11-1963" / "a.mp4").read_bytes() == b"bt"


def test_an_unlisted_file_in_the_old_folder_is_refused(temp_data_dir):
    bt = _chart("BT_12-11-1963")
    _file(bt, "CL_12-11-1963", "a.mp4", b"bt")
    (portal_patients_dir() / "CL_12-11-1963" / "loose.pdf").write_bytes(b"?")
    with pytest.raises(realign.RealignRefused):
        realign.plan_realign("BT_12-11-1963")


def test_a_failed_database_step_puts_every_file_back(temp_data_dir, tmp_path, monkeypatch):
    bt = _chart("BT_12-11-1963")
    _file(bt, "CL_12-11-1963", "a.mp4", b"bt")
    plan = realign.plan_realign("BT_12-11-1963")
    monkeypatch.setattr("backend.clinic_catalogue._bump", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    with pytest.raises(RuntimeError):
        realign.apply_realign(plan, tmp_path / "audit.json")
    assert (portal_patients_dir() / "CL_12-11-1963" / "a.mp4").read_bytes() == b"bt"
    assert all("/CL_12-11-1963/" in key for key in _keys(bt))
