"""The per-patient rekey: folder, filenames, conversations, sync state.

Each test here encodes something that actually went wrong or would go wrong:
a relabel that left the portal folder behind, a resumed run that renamed a file
twice, a conversation whose message history got rewritten on the way past.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from backend import patient_rekey
from backend.patient_rekey import PatientRekeyError


OLD = "12-11-1963-0"
NEW = "BT_12-11-1963"


def _portal_patient(root: Path, patient_id: str) -> Path:
    folder = root / patient_id
    (folder / "council").mkdir(parents=True)
    (folder / "project_mp4s").mkdir(parents=True)
    (folder / f"{patient_id}__patient-facing__v1__2026-06-22.pdf").write_bytes(
        b"%PDF-1.4 patient facing"
    )
    # The remote-key shape sync echoes back: the ID appears twice.
    (folder / f"{patient_id}__{patient_id}__v63__2026-08-03.pdf").write_bytes(
        b"%PDF-1.4 synced echo"
    )
    # An original source PDF never carried the ID and must not acquire one.
    (folder / "L_Connor_final_qeeg_Redacted.pdf").write_bytes(b"%PDF-1.4 source")
    (folder / "council" / f"{patient_id}__council-export-meta.json").write_text("{}")
    (folder / "project_mp4s" / f"{patient_id}_v1.mp4").write_bytes(b"\x00mp4 bytes")
    (folder / "$meta.json").write_text(
        json.dumps(
            {
                "patientId": patient_id,
                "birthdate": "12-11-1963",
                "index": 0,
                "identity": {
                    "schemaVersion": 2,
                    "firstInitial": "B",
                    "lastInitial": "T",
                },
            }
        )
    )
    return folder


def _conversation(directory: Path, name: str, patient_label: str) -> Path:
    path = directory / name
    path.write_text(
        json.dumps(
            {
                "id": "conv_abc123",
                "patient_label": patient_label,
                "patient_display_label": "Initials needed · 12-11-1963",
                "title": "L, Connor final qeeg.pdf",
                "messages": [
                    {"role": "user", "content": f"the {patient_label} patient"},
                    {"role": "assistant", "content": "Done — 2 files."},
                ],
                "artifacts": [
                    {
                        "kind": "pdf",
                        "path": f"portal_patients/{patient_label}/"
                        f"{patient_label}__patient-facing__v1__2026-06-22.pdf",
                        "label": "Patient-facing PDF",
                        "ts": "2026-06-22T12:00:00Z",
                    }
                ],
                "staged_reports": [
                    {
                        "patient_id": "db91e9c8-ecd1-44b9-856f-fe7a8bf205c1",
                        "patient_label": patient_label,
                        "report_id": "r-1",
                        "sha256": "abc",
                    }
                ],
            }
        )
    )
    return path


def _sync_state(path: Path, patient_id: str) -> Path:
    path.write_text(
        json.dumps(
            {
                "patients": {
                    patient_id: {"createdAt": 1, "createdBy": "local-sync"},
                    "09-05-1954-0": {"createdAt": 2, "createdBy": "local-sync"},
                },
                "files": {
                    f"{patient_id}/{patient_id}__patient-facing__v1__2026-06-22.pdf": {
                        "size": 23,
                        "version": 1,
                        "uploadedAt": 1700000000,
                        "logicalName": f"{patient_id}__patient-facing__v1__2026-06-22.pdf",
                        "remoteFileKey": f"{patient_id}__patient-facing__v1__2026-06-22.pdf",
                    },
                    "09-05-1954-0/09-05-1954-0.pdf": {
                        "size": 5,
                        "remoteFileKey": "09-05-1954-0__09-05-1954-0__v1__2026-01-01.pdf",
                    },
                },
            }
        )
    )
    return path


# --- filename token rewriting -------------------------------------------------


def test_rename_in_name_moves_every_token_and_leaves_neighbours_alone():
    assert (
        patient_rekey.rename_in_name(f"{OLD}__{OLD}__v63__2026-08-03.pdf", OLD, NEW)
        == f"{NEW}__{NEW}__v63__2026-08-03.pdf"
    )
    assert patient_rekey.rename_in_name("MF_MCI_30 TX_Redacted.pdf", OLD, NEW) == (
        "MF_MCI_30 TX_Redacted.pdf"
    )
    # A longer ID that merely starts with this one must not be partially rewritten.
    assert patient_rekey.rename_in_name(f"{OLD}1__x.pdf", OLD, NEW) == f"{OLD}1__x.pdf"


def test_rename_in_name_handles_a_high_collision_ordinal():
    old, new = "08-10-1989-0", "DK_08-10-1989_10"
    assert (
        patient_rekey.rename_in_name(f"{old}__guide__v2__2026-03-17.pdf", old, new)
        == f"{new}__guide__v2__2026-03-17.pdf"
    )


# --- the folder a relabel used to leave behind ---------------------------------


def test_rekey_moves_the_portal_folder_and_every_prefixed_file(tmp_path: Path):
    portal = tmp_path / "portal_patients"
    _portal_patient(portal, OLD)

    plan = patient_rekey.plan_patient_rekey(OLD, NEW, portal_root=portal)
    result = patient_rekey.apply_patient_rekey(plan)

    assert result.folder_moved is True
    assert not (portal / OLD).exists()
    moved = portal / NEW
    assert (moved / f"{NEW}__patient-facing__v1__2026-06-22.pdf").is_file()
    assert (moved / f"{NEW}__{NEW}__v63__2026-08-03.pdf").is_file()
    # Nested files are filed under the patient too.
    assert (moved / "council" / f"{NEW}__council-export-meta.json").is_file()
    assert (moved / "project_mp4s" / f"{NEW}_v1.mp4").is_file()
    # A source PDF that never carried the ID keeps its own name.
    assert (moved / "L_Connor_final_qeeg_Redacted.pdf").is_file()


def test_rekey_rewrites_folder_meta_to_the_one_based_canonical_ordinal(tmp_path: Path):
    portal = tmp_path / "portal_patients"
    _portal_patient(portal, "08-10-1989-1")

    plan = patient_rekey.plan_patient_rekey(
        "08-10-1989-1", "DK_08-10-1989_2", portal_root=portal
    )
    patient_rekey.apply_patient_rekey(plan)

    meta = json.loads((portal / "DK_08-10-1989_2" / "$meta.json").read_text())
    assert meta["patientId"] == "DK_08-10-1989_2"
    # Legacy index was 0-based; the canonical ordinal is 1-based.
    assert meta["index"] == 2


def _deliverable_bytes(folder: Path) -> dict[str, bytes]:
    """Historical content — the PDFs and videos that must never be rewritten."""
    return {
        path.name: path.read_bytes()
        for path in sorted(folder.rglob("*"))
        if path.is_file() and path.suffix in {".pdf", ".mp4", ".md"}
    }


def test_rekey_preserves_every_deliverable_byte_it_renames(tmp_path: Path):
    portal = tmp_path / "portal_patients"
    folder = _portal_patient(portal, OLD)
    before = _deliverable_bytes(folder)

    plan = patient_rekey.plan_patient_rekey(OLD, NEW, portal_root=portal)
    patient_rekey.apply_patient_rekey(plan)

    after = _deliverable_bytes(portal / NEW)
    assert len(after) == len(before)
    # Same bytes under new names — content by content, not name by name.
    assert sorted(before.values()) == sorted(after.values())
    # And the routing metadata that is *allowed* to change did change.
    assert json.loads((portal / NEW / "$meta.json").read_text())["patientId"] == NEW


def test_rekey_refuses_to_merge_onto_an_existing_destination(tmp_path: Path):
    portal = tmp_path / "portal_patients"
    _portal_patient(portal, OLD)
    (portal / NEW).mkdir()

    with pytest.raises(PatientRekeyError, match="already exists"):
        patient_rekey.plan_patient_rekey(OLD, NEW, portal_root=portal)


# --- resume: the run that must not duplicate work ------------------------------


def test_a_second_run_over_a_finished_patient_changes_nothing(tmp_path: Path):
    portal = tmp_path / "portal_patients"
    _portal_patient(portal, OLD)
    convs = tmp_path / "conversations"
    convs.mkdir()
    _conversation(convs, "conv_abc123.json", OLD)
    state = _sync_state(tmp_path / "sync.json", OLD)

    first = patient_rekey.apply_patient_rekey(
        patient_rekey.plan_patient_rekey(
            OLD, NEW, portal_root=portal, conversations_dir=convs,
            sync_state_paths=[state],
        )
    )
    assert first.files_renamed == 4

    listing_after_first = sorted(
        str(p.relative_to(portal)) for p in portal.rglob("*")
    )
    messages_after_first = json.loads(
        (convs / "conv_abc123.json").read_text()
    )["messages"]

    second = patient_rekey.apply_patient_rekey(
        patient_rekey.plan_patient_rekey(
            OLD, NEW, portal_root=portal, conversations_dir=convs,
            sync_state_paths=[state],
        )
    )

    assert second.files_renamed == 0
    assert second.folder_moved is False
    assert sorted(str(p.relative_to(portal)) for p in portal.rglob("*")) == (
        listing_after_first
    )
    assert json.loads((convs / "conv_abc123.json").read_text())["messages"] == (
        messages_after_first
    )


def test_a_run_resumed_after_the_folder_moved_finishes_the_file_renames(
    tmp_path: Path,
):
    portal = tmp_path / "portal_patients"
    _portal_patient(portal, OLD)
    plan = patient_rekey.plan_patient_rekey(OLD, NEW, portal_root=portal)

    # Simulate a crash right after the folder move: the folder is at the new ID
    # but every file inside still carries the old one.
    (portal / OLD).rename(portal / NEW)

    result = patient_rekey.apply_patient_rekey(plan)

    assert result.folder_moved is False
    assert result.files_renamed == 4
    assert (portal / NEW / f"{NEW}__patient-facing__v1__2026-06-22.pdf").is_file()


# --- conversations: filing changes, history does not ---------------------------


def test_rekey_repoints_conversation_filing_without_touching_messages(tmp_path: Path):
    convs = tmp_path / "conversations"
    convs.mkdir()
    path = _conversation(convs, "conv_abc123.json", OLD)
    original_messages = json.loads(path.read_text())["messages"]

    plan = patient_rekey.plan_patient_rekey(OLD, NEW, conversations_dir=convs)
    result = patient_rekey.apply_patient_rekey(plan)

    record = json.loads(path.read_text())
    assert result.conversations_repointed == 1
    assert record["patient_id"] == NEW
    assert "patient_label" not in record
    assert "patient_display_label" not in record
    # Byte-identical history, including the old ID quoted inside a message.
    assert record["messages"] == original_messages
    assert record["messages"][0]["content"] == f"the {OLD} patient"


def test_rekey_renames_the_nested_engine_uuid_field(tmp_path: Path):
    convs = tmp_path / "conversations"
    convs.mkdir()
    path = _conversation(convs, "conv_abc123.json", OLD)

    patient_rekey.apply_patient_rekey(
        patient_rekey.plan_patient_rekey(OLD, NEW, conversations_dir=convs)
    )

    staged = json.loads(path.read_text())["staged_reports"][0]
    # The engine UUID keeps its value under a name that cannot be mistaken for
    # a clinic ID if this report is dropped again.
    assert staged["patient_uuid"] == "db91e9c8-ecd1-44b9-856f-fe7a8bf205c1"
    assert staged["patient_id"] == NEW
    assert "patient_label" not in staged


def test_rekey_repoints_artifact_paths(tmp_path: Path):
    convs = tmp_path / "conversations"
    convs.mkdir()
    path = _conversation(convs, "conv_abc123.json", OLD)

    patient_rekey.apply_patient_rekey(
        patient_rekey.plan_patient_rekey(OLD, NEW, conversations_dir=convs)
    )

    artifact = json.loads(path.read_text())["artifacts"][0]
    assert artifact["path"] == (
        f"portal_patients/{NEW}/{NEW}__patient-facing__v1__2026-06-22.pdf"
    )


def test_a_rewritten_message_array_stops_the_rekey(tmp_path: Path, monkeypatch):
    convs = tmp_path / "conversations"
    convs.mkdir()
    path = _conversation(convs, "conv_abc123.json", OLD)
    plan = patient_rekey.plan_patient_rekey(OLD, NEW, conversations_dir=convs)

    real = patient_rekey.sha256_messages
    calls = {"n": 0}

    def drifting(messages):
        calls["n"] += 1
        return real(messages) if calls["n"] == 1 else "a-different-hash"

    monkeypatch.setattr(patient_rekey, "sha256_messages", drifting)

    with pytest.raises(PatientRekeyError, match="messages array changed"):
        patient_rekey.apply_patient_rekey(plan)

    # Nothing was written, so the conversation is still filed where it was.
    assert json.loads(path.read_text())["patient_label"] == OLD


# --- sync state and the remote worklist ----------------------------------------


def test_rekey_repoints_sync_state_patients_files_and_remote_keys(tmp_path: Path):
    state = _sync_state(tmp_path / "sync.json", OLD)

    result = patient_rekey.apply_patient_rekey(
        patient_rekey.plan_patient_rekey(OLD, NEW, sync_state_paths=[state])
    )

    payload = json.loads(state.read_text())
    assert result.sync_entries_repointed == 2
    assert NEW in payload["patients"] and OLD not in payload["patients"]
    assert f"{NEW}/{NEW}__patient-facing__v1__2026-06-22.pdf" in payload["files"]
    entry = payload["files"][f"{NEW}/{NEW}__patient-facing__v1__2026-06-22.pdf"]
    assert entry["remoteFileKey"] == f"{NEW}__patient-facing__v1__2026-06-22.pdf"
    # Another patient's entries are untouched.
    assert "09-05-1954-0" in payload["patients"]
    assert "09-05-1954-0/09-05-1954-0.pdf" in payload["files"]


def test_remote_worklist_preserves_version_upload_date_and_size(tmp_path: Path):
    state = json.loads(_sync_state(tmp_path / "sync.json", OLD).read_text())

    work = patient_rekey.remote_rekey_worklist(state, OLD, NEW)

    assert len(work) == 1
    item = work[0]
    assert item["oldFileKey"] == f"{OLD}__patient-facing__v1__2026-06-22.pdf"
    assert item["newFileKey"] == f"{NEW}__patient-facing__v1__2026-06-22.pdf"
    assert item["version"] == 1
    assert item["uploadedAt"] == 1700000000
    assert item["size"] == 23
