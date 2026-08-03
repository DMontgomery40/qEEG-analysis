"""The cutover migrator: classification, identity, apply, and the journal.

The fixtures here are the shapes the real clinic data actually has — a duplicate
label, a folder with no initials on file, a placeholder pair contradicted by the
report's own filename, a folder holding somebody else's reports, rows written by
a test run against the live database, and an upload prefix whose job marker
never landed.
"""

from __future__ import annotations

import json
import sqlite3
from types import SimpleNamespace
from pathlib import Path

import pytest

from scripts import migrate_canonical_patient_ids as migrator


# --------------------------------------------------------------------------- #
# Fixture world
# --------------------------------------------------------------------------- #


def _make_db(path: Path, patients: list[dict]) -> None:
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        CREATE TABLE patients (id VARCHAR NOT NULL PRIMARY KEY, label VARCHAR NOT NULL,
            notes TEXT NOT NULL DEFAULT '', created_at DATETIME NOT NULL,
            updated_at DATETIME NOT NULL DEFAULT '');
        CREATE TABLE reports (id VARCHAR NOT NULL PRIMARY KEY, patient_id VARCHAR NOT NULL,
            filename VARCHAR NOT NULL DEFAULT '', mime_type VARCHAR NOT NULL DEFAULT '',
            stored_path VARCHAR NOT NULL DEFAULT '',
            extracted_text_path VARCHAR NOT NULL DEFAULT '',
            created_at DATETIME NOT NULL DEFAULT '');
        CREATE TABLE runs (id VARCHAR NOT NULL PRIMARY KEY, patient_id VARCHAR NOT NULL,
            report_id VARCHAR NOT NULL DEFAULT '', status VARCHAR NOT NULL DEFAULT '',
            council_model_ids_json TEXT NOT NULL DEFAULT '',
            consolidator_model_id VARCHAR NOT NULL DEFAULT '',
            label_map_json TEXT NOT NULL DEFAULT '',
            error_message TEXT NOT NULL DEFAULT '', created_at DATETIME NOT NULL DEFAULT '');
        CREATE TABLE patient_files (id VARCHAR NOT NULL PRIMARY KEY,
            patient_id VARCHAR NOT NULL, filename VARCHAR NOT NULL DEFAULT '',
            mime_type VARCHAR NOT NULL DEFAULT '', size_bytes INTEGER NOT NULL DEFAULT 0,
            stored_path VARCHAR NOT NULL DEFAULT '', created_at DATETIME NOT NULL DEFAULT '');
        """
    )
    for entry in patients:
        conn.execute(
            "INSERT INTO patients (id, label, notes, created_at, updated_at) "
            "VALUES (?, ?, '', ?, ?)",
            (entry["uuid"], entry["label"], entry.get("created_at", "2026-01-01"),
             entry.get("created_at", "2026-01-01")),
        )
        for index, stored in enumerate(entry.get("reports", [])):
            conn.execute(
                "INSERT INTO reports (id, patient_id, stored_path) VALUES (?, ?, ?)",
                (f"{entry['uuid']}-r{index}", entry["uuid"], stored),
            )
        for index in range(entry.get("runs", 0)):
            conn.execute(
                "INSERT INTO runs (id, patient_id) VALUES (?, ?)",
                (f"{entry['uuid']}-u{index}", entry["uuid"]),
            )
        for index in range(entry.get("files", 0)):
            conn.execute(
                "INSERT INTO patient_files (id, patient_id) VALUES (?, ?)",
                (f"{entry['uuid']}-f{index}", entry["uuid"]),
            )
    conn.commit()
    conn.close()


def _folder(portal: Path, patient_id: str, *, initials=None, files=()) -> Path:
    folder = portal / patient_id
    folder.mkdir(parents=True, exist_ok=True)
    if initials:
        (folder / "$meta.json").write_text(
            json.dumps(
                {
                    "patientId": patient_id,
                    "index": 0,
                    "identity": {
                        "schemaVersion": 2,
                        "firstInitial": initials[0],
                        "lastInitial": initials[1],
                    },
                }
            )
        )
    for name in files:
        (folder / name).write_bytes(b"%PDF-1.4 " + name.encode())
    return folder


@pytest.fixture
def world(tmp_path: Path):
    """One small clinic containing every awkward shape the real one has."""
    portal = tmp_path / "portal_patients"
    portal.mkdir()
    db = tmp_path / "app.db"

    _folder(portal, "01-19-1966-0", initials=("P", "G"),
            files=["PG_femal_TBI_Redacted.pdf",
                   "01-19-1966-0__patient-facing__v1__2026-01-16.pdf"])
    _folder(portal, "06-28-1977-0", files=["06-28-1977-0__patient-facing__v1__2026-07-01.pdf"])
    _folder(portal, "12-11-1963-0", initials=("B", "T"),
            files=["12-11-1963-0__patient-facing__v1__2026-06-22.pdf"])
    _folder(portal, "08-10-1989-0", initials=("D", "K"),
            files=["DK_20Tx_toxic-brain-injury_Redacted.pdf"])
    _folder(portal, "03-05-2010-0", initials=("R", "W"),
            files=["03-05-2010-0__DK_20Tx_toxic-brain-injury__v1__2026-05-09.pdf",
                   "03-05-2010-0__patient-facing__v1__2026-05-09.pdf"])

    _make_db(
        db,
        [
            {"uuid": "aaaaaaaa-0000-0000-0000-000000000001",
             "label": "01-19-1966-0", "created_at": "2026-01-13",
             "reports": ["data/reports/a/original.pdf"], "runs": 37, "files": 1},
            {"uuid": "bbbbbbbb-0000-0000-0000-000000000002",
             "label": "01-19-1966-0", "created_at": "2026-01-14",
             "reports": ["data/reports/b/original.pdf"], "runs": 1, "files": 3},
            {"uuid": "cccccccc-0000-0000-0000-000000000003",
             "label": "06-28-1977-0", "reports": ["data/reports/c/original.pdf"]},
            {"uuid": "dddddddd-0000-0000-0000-000000000004",
             "label": "12-11-1963-0", "reports": ["data/reports/d/original.pdf"]},
            {"uuid": "eeeeeeee-0000-0000-0000-000000000005",
             "label": "08-10-1989-0", "reports": ["data/reports/e/original.pdf"]},
            {"uuid": "ffffffff-0000-0000-0000-000000000006",
             "label": "03-05-2010-0", "reports": ["data/reports/f/original.pdf"]},
            {"uuid": "99999999-0000-0000-0000-000000000007",
             "label": "02-29-1984-0", "reports": ["data/reports/g/original.pdf"]},
            {"uuid": "88888888-0000-0000-0000-000000000008",
             "label": "08-10-1989-1",
             "reports": ["/private/var/pytest-of-davidmontgomery/pytest-15/t/report.pdf"]},
            {"uuid": "77777777-0000-0000-0000-000000000009",
             "label": "diag-175b76e4", "reports": ["data/reports/h/original.pdf"]},
        ],
    )

    convs = tmp_path / "conversations"
    convs.mkdir()
    (convs / "conv_1.json").write_text(
        json.dumps(
            {
                "id": "conv_1",
                "patient_label": "12-11-1963-0",
                "title": "L, Connor final qeeg.pdf",
                "messages": [{"role": "user", "content": "hello"}],
                "artifacts": [],
            }
        )
    )

    (portal / ".qeeg_portal_sync_state.json").write_text(
        json.dumps(
            {
                "patients": {
                    "01-19-1966-0": {
                        "identity": {"schemaVersion": 2, "firstInitial": "P",
                                     "lastInitial": "G"}
                    },
                },
                "files": {
                    "01-19-1966-0/01-19-1966-0__patient-facing__v1__2026-01-16.pdf": {
                        "size": 1024,
                        "version": 1,
                        "uploadedAt": 1700000000,
                        "logicalName": "01-19-1966-0__patient-facing__v1__2026-01-16.pdf",
                        "remoteFileKey":
                            "01-19-1966-0__patient-facing__v1__2026-01-16.pdf",
                    }
                },
            }
        )
    )

    return SimpleNamespace(
        db=str(db),
        portal_root=str(portal),
        conversations_dir=str(convs),
        cathode_root="",
        explainer_root="",
        manifest_out="",
        journal=str(tmp_path / "journal.jsonl"),
        apply=False,
        dry_run=True,
        window_confirmed=True,
    )


# --------------------------------------------------------------------------- #
# Classification: every row in exactly one bucket
# --------------------------------------------------------------------------- #


def test_every_row_lands_in_exactly_one_bucket(world):
    report = migrator.build_report(world)

    rows = migrator.read_patient_rows(Path(world.db))
    assert sum(report["buckets"].values()) == len(rows)
    assert len({e["key"] for e in report["classified"]}) == len(rows)


def test_a_row_written_by_a_test_run_is_never_migrated(world):
    report = migrator.build_report(world)

    polluted = [
        e for e in report["classified"]
        if e["bucket"] == migrator.BUCKET_TEST_POLLUTION
    ]
    assert [e["label"] for e in polluted] == ["08-10-1989-1"]
    assert "08-10-1989-1" not in report["mapping"]


def test_the_qa_fixture_is_carved_out_of_the_production_mapping(world):
    report = migrator.build_report(world)

    fixture = [
        e for e in report["classified"] if e["bucket"] == migrator.BUCKET_QA_FIXTURE
    ]
    assert [e["label"] for e in fixture] == ["02-29-1984-0"]
    # Bundled for rollback, but never given a canonical ID or a reservation.
    assert "02-29-1984-0" not in report["mapping"]
    assert "DM_02-29-1984" not in report["mapping"].values()


def test_a_scratch_row_is_not_mistaken_for_a_patient(world):
    report = migrator.build_report(world)

    scratch = [
        e for e in report["classified"] if e["bucket"] == migrator.BUCKET_NON_PATIENT
    ]
    assert [e["label"] for e in scratch] == ["diag-175b76e4"]


def test_the_duplicate_label_names_the_survivor_that_holds_the_work(world):
    report = migrator.build_report(world)

    dupes = [
        e for e in report["classified"] if e["bucket"] == migrator.BUCKET_DUPLICATE
    ]
    assert len(dupes) == 1
    # The row with 37 runs survives; the one-run row folds into it.
    assert dupes[0]["uuid"].startswith("bbbbbbbb")
    assert dupes[0]["survivor_uuid"].startswith("aaaaaaaa")
    # And the label still migrates exactly once.
    assert report["mapping"]["01-19-1966-0"] == "PG_01-19-1966"


# --------------------------------------------------------------------------- #
# Identity: resolved, or a precise question
# --------------------------------------------------------------------------- #


def test_a_folder_with_no_initials_blocks_the_run_by_name(world):
    report = migrator.build_report(world)

    assert any(
        b.startswith("06-28-1977-0:") and "no initials on file" in b
        for b in report["blockers"]
    )
    assert "06-28-1977-0" not in report["mapping"]


def test_placeholder_initials_are_caught_by_the_name_on_the_report(world):
    report = migrator.build_report(world)

    blocker = next(b for b in report["blockers"] if b.startswith("12-11-1963-0:"))
    assert "BT" in blocker and "L, Connor final qeeg.pdf" in blocker
    assert "placeholder" in blocker
    assert "12-11-1963-0" not in report["mapping"]


def test_initials_written_into_the_filename_are_not_a_false_alarm(world):
    report = migrator.build_report(world)

    # DK_20Tx… agrees with the stored DK by construction.
    assert not any(b.startswith("08-10-1989-0:") for b in report["blockers"])
    assert report["mapping"]["08-10-1989-0"] == "DK_08-10-1989"


def test_a_folder_holding_another_patients_reports_stops_the_run(world):
    report = migrator.build_report(world)

    blocker = next(b for b in report["blockers"] if b.startswith("03-05-2010-0:"))
    assert "08-10-1989-0" in blocker and "DK" in blocker
    assert "03-05-2010-0" not in report["mapping"]
    assert report["mixed_patient_folders"][0]["patient_id"] == "03-05-2010-0"


def test_a_clean_patient_gets_an_unsuffixed_canonical_id(world):
    report = migrator.build_report(world)

    assert report["mapping"]["01-19-1966-0"] == "PG_01-19-1966"


def test_the_legacy_ordinal_is_never_carried_across():
    """Legacy N counts birthdate collisions; canonical N counts initials+DOB."""
    entries = [
        migrator.Classified(
            key="a", bucket=migrator.BUCKET_MIGRATE, reason="", label="01-01-1990-0",
            birthdate="01-01-1990", initials=("A", "B"), evidence={"legacy_ordinal": 0},
        ),
        # A different person, same birthday, different initials: legacy gave them
        # `-1`, but canonically they collide with nobody and stay unsuffixed.
        migrator.Classified(
            key="b", bucket=migrator.BUCKET_MIGRATE, reason="", label="01-01-1990-1",
            birthdate="01-01-1990", initials=("C", "D"), evidence={"legacy_ordinal": 1},
        ),
    ]

    mapping, problems = migrator.allocate_new_ids(entries)

    assert problems == []
    assert mapping == {"01-01-1990-0": "AB_01-01-1990", "01-01-1990-1": "CD_01-01-1990"}


def test_two_people_sharing_initials_and_a_birthday_get_an_ordinal():
    entries = [
        migrator.Classified(
            key="a", bucket=migrator.BUCKET_MIGRATE, reason="", label="01-01-1990-0",
            birthdate="01-01-1990", initials=("A", "B"), evidence={"legacy_ordinal": 0},
        ),
        migrator.Classified(
            key="b", bucket=migrator.BUCKET_MIGRATE, reason="", label="01-01-1990-1",
            birthdate="01-01-1990", initials=("A", "B"), evidence={"legacy_ordinal": 1},
        ),
    ]

    mapping, problems = migrator.allocate_new_ids(entries)

    assert problems == []
    # `_1` never exists: the first is unsuffixed and the second is `_2`.
    assert mapping == {"01-01-1990-0": "AB_01-01-1990", "01-01-1990-1": "AB_01-01-1990_2"}


# --------------------------------------------------------------------------- #
# The remote manifest and the estimate
# --------------------------------------------------------------------------- #


def test_the_remote_manifest_carries_version_and_upload_date_across(world):
    report = migrator.build_report(world)

    item = next(
        i for i in report["remote_manifest"] if i["patientIdOld"] == "01-19-1966-0"
    )
    assert item["newFileKey"] == "PG_01-19-1966__patient-facing__v1__2026-01-16.pdf"
    assert item["version"] == 1
    assert item["uploadedAt"] == 1700000000


def test_the_estimate_separates_metadata_renames_from_bytes_over_the_wire():
    estimate = migrator.estimate_window(
        local_bytes=5 * 1024**3, local_files=3000,
        remote_items=[{"size": 1024**3}],
    )

    # Local renames are metadata; 5 GB of them must not read as hours of work.
    assert estimate["local_rename_minutes"] < 1
    # The wire is what costs: 1 GB up and back at the assumed rate.
    assert estimate["remote_copy_verify_minutes"] > estimate["hash_verify_minutes"]
    assert estimate["estimated_minutes"] == pytest.approx(
        estimate["subtotal_minutes"] * migrator.VERIFICATION_ALLOWANCE, rel=0.01
    )


def test_orphan_upload_prefixes_are_listed_for_cleanup(tmp_path: Path):
    pending = tmp_path / "pending"
    (pending / "upload-live").mkdir(parents=True)
    (pending / "upload-abandoned").mkdir(parents=True)

    orphans = migrator.find_orphan_pending_prefixes(pending, ["upload-live"])

    assert orphans == ["upload-abandoned"]


# --------------------------------------------------------------------------- #
# Applying, and resuming
# --------------------------------------------------------------------------- #


def _unblock(world):
    """Supply the identities the dry run asks for, so apply has something to do."""
    portal = Path(world.portal_root)
    for patient_id, initials in (
        ("06-28-1977-0", ("J", "M")),
        ("12-11-1963-0", ("C", "L")),
    ):
        meta = json.loads((portal / patient_id / "$meta.json").read_text()) if (
            portal / patient_id / "$meta.json"
        ).is_file() else {"patientId": patient_id, "index": 0}
        meta["identity"] = {
            "schemaVersion": 2,
            "firstInitial": initials[0],
            "lastInitial": initials[1],
        }
        (portal / patient_id / "$meta.json").write_text(json.dumps(meta))
    # The mixed-patient folder is a decision, not something apply can invent.
    import shutil

    shutil.rmtree(portal / "03-05-2010-0")
    conn = sqlite3.connect(world.db)
    conn.execute("DELETE FROM reports WHERE patient_id LIKE 'ffffffff%'")
    conn.execute("DELETE FROM patients WHERE id LIKE 'ffffffff%'")
    conn.commit()
    conn.close()


def test_apply_refuses_while_anything_is_unresolved(world, capsys):
    report = migrator.build_report(world)

    assert migrator.run_apply(world, report) == 1


def test_apply_renames_the_world_and_the_journal_records_each_patient(world):
    _unblock(world)
    report = migrator.build_report(world)
    assert report["blockers"] == [], report["blockers"]

    assert migrator.run_apply(world, report) == 0

    portal = Path(world.portal_root)
    assert (portal / "PG_01-19-1966").is_dir()
    assert not (portal / "01-19-1966-0").exists()
    assert (
        portal / "PG_01-19-1966" / "PG_01-19-1966__patient-facing__v1__2026-01-16.pdf"
    ).is_file()
    # The source PDF that never carried the ID keeps its own name.
    assert (portal / "PG_01-19-1966" / "PG_femal_TBI_Redacted.pdf").is_file()

    conn = sqlite3.connect(world.db)
    labels = {r[0] for r in conn.execute("SELECT label FROM patients")}
    # The duplicate row is gone and its work moved to the survivor.
    assert "PG_01-19-1966" in labels
    survivor_runs = conn.execute(
        "SELECT COUNT(*) FROM runs WHERE patient_id = ?",
        ("aaaaaaaa-0000-0000-0000-000000000001",),
    ).fetchone()[0]
    assert survivor_runs == 38
    conn.close()

    journal = Journal_lines(Path(world.journal))
    assert any(j.get("new_id") == "PG_01-19-1966" for j in journal)


def Journal_lines(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def test_a_resumed_apply_replays_nothing_it_already_finished(world):
    _unblock(world)
    report = migrator.build_report(world)
    assert migrator.run_apply(world, report) == 0

    portal = Path(world.portal_root)
    listing = sorted(str(p.relative_to(portal)) for p in portal.rglob("*"))
    journal_length = len(Journal_lines(Path(world.journal)))

    # The second run reads the same world — now already migrated — and the
    # journal tells it there is nothing left to do.
    again = migrator.build_report(world)
    assert migrator.run_apply(world, again) == 0

    assert sorted(str(p.relative_to(portal)) for p in portal.rglob("*")) == listing
    assert len(Journal_lines(Path(world.journal))) == journal_length


def test_apply_refuses_to_touch_the_live_clinic_data(world, monkeypatch, capsys):
    world.apply = True
    world.dry_run = False
    world.db = "/Users/davidmontgomery/qEEG-analysis/data/app.db"

    exit_code = migrator.main(
        [
            "--apply",
            "--window-confirmed",
            "--db",
            "/Users/davidmontgomery/qEEG-analysis/data/app.db",
            "--portal-root",
            "/Users/davidmontgomery/qEEG-analysis/data/portal_patients",
        ]
    )

    assert exit_code == 2
    assert "Refusing to --apply against the live database" in capsys.readouterr().err


def test_apply_requires_the_window_to_be_confirmed(capsys, tmp_path):
    exit_code = migrator.main(
        ["--apply", "--db", str(tmp_path / "x.db"), "--portal-root", str(tmp_path)]
    )

    assert exit_code == 2
    assert "--window-confirmed" in capsys.readouterr().err


# --------------------------------------------------------------------------- #
# The renderer boundary
# --------------------------------------------------------------------------- #


def test_the_scanner_finds_a_validator_that_would_reject_a_canonical_id(tmp_path: Path):
    """A renderer that cannot read the clinic's ID skips every real patient."""
    repo = tmp_path / "renderer"
    repo.mkdir()
    (repo / "publish.py").write_text(
        'import re\n'
        'PATIENT_ID_RE = re.compile(r"^\\d{2}-\\d{2}-\\d{4}-\\d+$")\n'
    )

    findings = migrator.scan_renderer_repo(repo)

    assert len(findings) == 1
    assert findings[0]["line"] == 2
    assert "date-of-birth-only" in findings[0]["issue"]


def test_named_groups_do_not_hide_a_date_only_validator(tmp_path: Path):
    """The real one was written with named groups and slipped a naive scan."""
    repo = tmp_path / "renderer"
    repo.mkdir()
    (repo / "publish.py").write_text(
        'import re\n'
        '_PATIENT_ID_RE = re.compile('
        'r"^(?P<mm>\\d{2})-(?P<dd>\\d{2})-(?P<yyyy>\\d{4})-(?P<n>\\d+)$")\n'
    )

    assert len(migrator.scan_renderer_repo(repo)) == 1


def test_the_canonical_contract_is_not_reported_as_a_finding(tmp_path: Path):
    """The canonical ID contains a birthdate too — the initials are what differ."""
    repo = tmp_path / "renderer"
    repo.mkdir()
    (repo / "publish.py").write_text(
        'import re\n'
        'PATIENT_ID_RE = re.compile('
        'r"^[A-Z]{2}_\\d{2}-\\d{2}-\\d{4}(?:_(?:[2-9]|[1-9]\\d+))?$")\n'
    )

    assert migrator.scan_renderer_repo(repo) == []


def test_an_unresolved_renderer_finding_fails_the_dry_run(world, tmp_path: Path):
    repo = tmp_path / "renderer"
    repo.mkdir()
    (repo / "publish.py").write_text(
        'import re\n'
        'PATIENT_ID_RE = re.compile(r"^\\d{2}-\\d{2}-\\d{4}-\\d+$")\n'
    )
    world.explainer_root = str(repo)

    report = migrator.build_report(world)

    assert any("local-explainer-video:" in b for b in report["blockers"])
