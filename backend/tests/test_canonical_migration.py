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


def _make_db(
    path: Path, patients: list[dict], *, upgraded: bool = True, data_root: Path | None = None
) -> None:
    """A database in the shape the new engine leaves behind.

    ``upgraded=False`` gives the pre-cutover five-column shape, which is what
    the live database still looks like until the new engine runs against it.
    """
    conn = sqlite3.connect(path)
    if upgraded:
        conn.executescript(
            """
            CREATE TABLE patient_id_reservations (
                patient_id VARCHAR NOT NULL PRIMARY KEY,
                first_initial VARCHAR NOT NULL DEFAULT '',
                last_initial VARCHAR NOT NULL DEFAULT '',
                birthdate VARCHAR NOT NULL DEFAULT '',
                ordinal INTEGER NOT NULL DEFAULT 1,
                created_at DATETIME NOT NULL DEFAULT '');
            """
        )
    identity_columns = (
        ", birthdate VARCHAR, first_name VARCHAR, last_name VARCHAR,"
        " first_initial VARCHAR, last_initial VARCHAR"
        if upgraded
        else ""
    )
    conn.executescript(
        f"""
        CREATE TABLE patients (id VARCHAR NOT NULL PRIMARY KEY, label VARCHAR NOT NULL,
            notes TEXT NOT NULL DEFAULT '', created_at DATETIME NOT NULL,
            updated_at DATETIME NOT NULL DEFAULT ''{identity_columns});
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
                "INSERT INTO reports (id, patient_id, filename, stored_path) "
                "VALUES (?, ?, ?, ?)",
                (
                    f"{entry['uuid']}-r{index}",
                    entry["uuid"],
                    entry.get("report_names", [None] * (index + 1))[index]
                    or Path(stored).name,
                    stored,
                ),
            )
            # Real bytes behind the row: the migrator matches misfiled rows back
            # to their patient by digest, so a fixture without files tests nothing.
            if not stored.startswith("/") and data_root is not None:
                target = data_root / stored
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(
                    (entry.get("report_bytes", [None] * (index + 1))[index]
                     or f"%PDF-1.4 {entry['uuid']}-{index}").encode()
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
    """One small clinic containing every awkward shape the real one has.

    Laid out like the real installation — ``<repo>/data/portal_patients`` with
    report files under ``<repo>/data/reports`` — because the migrator resolves
    stored paths relative to the repository root.
    """
    portal = tmp_path / "data" / "portal_patients"
    portal.mkdir(parents=True)
    db = tmp_path / "data" / "app.db"

    _folder(portal, "07-07-1907-0", initials=("P", "G"),
            files=["PG_femal_TBI_Redacted.pdf",
                   "07-07-1907-0__patient-facing__v1__2026-01-16.pdf"])
    _folder(portal, "10-10-1910-0", files=["10-10-1910-0__patient-facing__v1__2026-07-01.pdf"])
    _folder(portal, "08-08-1908-0", initials=("B", "T"),
            files=["08-08-1908-0__patient-facing__v1__2026-06-22.pdf"])
    _folder(portal, "09-09-1909-0", initials=("D", "K"),
            files=["DK_20Tx_toxic-brain-injury_Redacted.pdf"])
    _folder(portal, "11-11-1911-0", initials=("R", "W"),
            files=["11-11-1911-0__DK_20Tx_toxic-brain-injury__v1__2026-05-09.pdf",
                   "11-11-1911-0__patient-facing__v1__2026-05-09.pdf"])

    _make_db(
        db,
        [
            {"uuid": "aaaaaaaa-0000-0000-0000-000000000001",
             "label": "07-07-1907-0", "created_at": "2026-01-13",
             "reports": ["data/reports/a/original.pdf"], "runs": 37, "files": 1},
            {"uuid": "bbbbbbbb-0000-0000-0000-000000000002",
             "label": "07-07-1907-0", "created_at": "2026-01-14",
             "reports": ["data/reports/b/original.pdf"], "runs": 1, "files": 3},
            {"uuid": "cccccccc-0000-0000-0000-000000000003",
             "label": "10-10-1910-0", "reports": ["data/reports/c/original.pdf"]},
            {"uuid": "dddddddd-0000-0000-0000-000000000004",
             "label": "08-08-1908-0", "reports": ["data/reports/d/original.pdf"]},
            {"uuid": "eeeeeeee-0000-0000-0000-000000000005",
             "label": "09-09-1909-0", "reports": ["data/reports/e/original.pdf"]},
            {"uuid": "ffffffff-0000-0000-0000-000000000006",
             "label": "11-11-1911-0", "reports": ["data/reports/f/original.pdf"]},
            {"uuid": "99999999-0000-0000-0000-000000000007",
             "label": "12-12-1912-0", "reports": ["data/reports/g/original.pdf"]},
            {"uuid": "88888888-0000-0000-0000-000000000008",
             "label": "09-09-1909-1",
             "reports": ["/private/var/pytest-of-davidmontgomery/pytest-15/t/report.pdf"]},
            {"uuid": "77777777-0000-0000-0000-000000000009",
             "label": "diag-175b76e4", "reports": ["data/reports/h/original.pdf"]},
            # An unpadded twin holding a byte-identical copy of PG's report:
            # answerable from the bytes, so it should reduce to a confirmation.
            {"uuid": "66666666-0000-0000-0000-000000000010",
             "label": "7-7-1907",
             "reports": ["data/reports/i/original.pdf"],
             "report_names": ["PG_femal_TBI_Redacted.pdf"],
             "report_bytes": ["%PDF-1.4 aaaaaaaa-0000-0000-0000-000000000001-0"]},
        ],
        data_root=tmp_path,
    )

    jobs = tmp_path / "pipeline_jobs"
    jobs.mkdir()
    (jobs / "07-07-1907-0.json").write_text(
        json.dumps({"patient_id": "07-07-1907-0", "status": "complete"})
    )

    convs = tmp_path / "conversations"
    convs.mkdir()
    (convs / "conv_1.json").write_text(
        json.dumps(
            {
                "id": "conv_1",
                "patient_label": "08-08-1908-0",
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
                    "07-07-1907-0": {
                        "identity": {"schemaVersion": 2, "firstInitial": "P",
                                     "lastInitial": "G"}
                    },
                },
                "files": {
                    "07-07-1907-0/07-07-1907-0__patient-facing__v1__2026-01-16.pdf": {
                        "size": 1024,
                        "version": 1,
                        "uploadedAt": 1700000000,
                        "logicalName": "07-07-1907-0__patient-facing__v1__2026-01-16.pdf",
                        "remoteFileKey":
                            "07-07-1907-0__patient-facing__v1__2026-01-16.pdf",
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
        pipeline_jobs_dir=str(jobs),
        pending_uploads_dir="",
        this_is_the_scheduled_cutover=False,
        qa_candidates_confirmed=False,
        answers="",
        # Synthetic label sets: a fixture world must never wear a real
        # patient's ID, or an --apply pointed at a live directory would
        # match one.
        qa_fixture_labels=["12-12-1912-0"],
        qa_candidate_labels=["03-03-1903-0", "13-13-1913-0"],
        rollback_bundle=str(tmp_path / "rollback-bundle"),
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
    assert [e["label"] for e in polluted] == ["09-09-1909-1"]
    assert "09-09-1909-1" not in report["mapping"]


def test_the_qa_fixture_is_carved_out_of_the_production_mapping(world):
    report = migrator.build_report(world)

    fixture = [
        e for e in report["classified"] if e["bucket"] == migrator.BUCKET_QA_FIXTURE
    ]
    assert [e["label"] for e in fixture] == ["12-12-1912-0"]
    # Bundled for rollback, but never given a canonical ID or a reservation.
    assert "12-12-1912-0" not in report["mapping"]
    assert "DM_12-12-1912" not in report["mapping"].values()


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
    assert report["mapping"]["07-07-1907-0"] == "PG_07-07-1907"


# --------------------------------------------------------------------------- #
# Identity: resolved, or a precise question
# --------------------------------------------------------------------------- #


def test_a_folder_with_no_initials_blocks_the_run_by_name(world):
    report = migrator.build_report(world)

    assert any(
        b.startswith("10-10-1910-0:") and "no initials on file" in b
        for b in report["blockers"]
    )
    assert "10-10-1910-0" not in report["mapping"]


def test_placeholder_initials_are_caught_by_the_name_on_the_report(world):
    report = migrator.build_report(world)

    blocker = next(b for b in report["blockers"] if b.startswith("08-08-1908-0:"))
    assert "BT" in blocker and "L, Connor final qeeg.pdf" in blocker
    assert "placeholder" in blocker
    assert "08-08-1908-0" not in report["mapping"]


def test_initials_written_into_the_filename_are_not_a_false_alarm(world):
    report = migrator.build_report(world)

    # DK_20Tx… agrees with the stored DK by construction.
    assert not any(b.startswith("09-09-1909-0:") for b in report["blockers"])
    assert report["mapping"]["09-09-1909-0"] == "DK_09-09-1909"


def test_a_folder_holding_another_patients_reports_stops_the_run(world):
    report = migrator.build_report(world)

    blocker = next(b for b in report["blockers"] if b.startswith("11-11-1911-0:"))
    assert "09-09-1909-0" in blocker and "DK" in blocker
    assert "11-11-1911-0" not in report["mapping"]
    assert report["mixed_patient_folders"][0]["patient_id"] == "11-11-1911-0"


def test_a_clean_patient_gets_an_unsuffixed_canonical_id(world):
    report = migrator.build_report(world)

    assert report["mapping"]["07-07-1907-0"] == "PG_07-07-1907"


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
        i for i in report["remote_manifest"] if i["patientIdOld"] == "07-07-1907-0"
    )
    assert item["newFileKey"] == "PG_07-07-1907__patient-facing__v1__2026-01-16.pdf"
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
        ("10-10-1910-0", ("J", "M")),
        ("08-08-1908-0", ("C", "L")),
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

    shutil.rmtree(portal / "11-11-1911-0")
    conn = sqlite3.connect(world.db)
    conn.execute("DELETE FROM reports WHERE patient_id LIKE 'ffffffff%'")
    conn.execute("DELETE FROM patients WHERE id LIKE 'ffffffff%'")
    # The operator confirmed the unpadded twin is a duplicate of the padded row
    # and retired it, which is what the dry run asked for.
    conn.execute("DELETE FROM reports WHERE patient_id LIKE '66666666%'")
    conn.execute("DELETE FROM patients WHERE id LIKE '66666666%'")
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
    assert (portal / "PG_07-07-1907").is_dir()
    assert not (portal / "07-07-1907-0").exists()
    assert (
        portal / "PG_07-07-1907" / "PG_07-07-1907__patient-facing__v1__2026-01-16.pdf"
    ).is_file()
    # The source PDF that never carried the ID keeps its own name.
    assert (portal / "PG_07-07-1907" / "PG_femal_TBI_Redacted.pdf").is_file()

    conn = sqlite3.connect(world.db)
    labels = {r[0] for r in conn.execute("SELECT label FROM patients")}
    # The duplicate row is gone and its work moved to the survivor.
    assert "PG_07-07-1907" in labels
    survivor_runs = conn.execute(
        "SELECT COUNT(*) FROM runs WHERE patient_id = ?",
        ("aaaaaaaa-0000-0000-0000-000000000001",),
    ).fetchone()[0]
    assert survivor_runs == 38
    conn.close()

    journal = Journal_lines(Path(world.journal))
    assert any(j.get("new_id") == "PG_07-07-1907" for j in journal)


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


def test_apply_refuses_a_database_the_new_engine_has_not_upgraded(tmp_path: Path):
    """Writing canonical labels into the old shape leaves every patient with no
    reservation and no identity — the exact state the cutover exists to end."""
    db = tmp_path / "old.db"
    _make_db(db, [{"uuid": "a" * 8, "label": "07-07-1907-0"}], upgraded=False)

    with pytest.raises(migrator.MigrationStop, match="patient_id_reservations"):
        migrator.assert_schema_ready(db)


def test_migrated_patients_get_a_reservation_and_their_identity_columns(world):
    _unblock(world)
    report = migrator.build_report(world)
    assert migrator.run_apply(world, report) == 0

    conn = sqlite3.connect(world.db)
    reserved = {r[0] for r in conn.execute("SELECT patient_id FROM patient_id_reservations")}
    # Without this the ID becomes reissuable the moment the row is deleted.
    assert "PG_07-07-1907" in reserved
    assert "DK_09-09-1909" in reserved

    row = conn.execute(
        "SELECT birthdate, first_initial, last_initial FROM patients WHERE label = ?",
        ("PG_07-07-1907",),
    ).fetchone()
    # A migrated patient with NULL identity sends the next intake through name
    # matching with nothing to match.
    assert row == ("07-07-1907", "P", "G")
    conn.close()


def test_the_qa_fixture_is_bundled_before_it_is_removed(world):
    report = migrator.build_report(world)

    bundle = report["qa_fixture_bundle"]
    assert [p["label"] for p in bundle["patients"]] == ["12-12-1912-0"]
    assert bundle["reports"], "its rows have to be recoverable from the bundle"

    _unblock(world)
    assert migrator.run_apply(world, migrator.build_report(world)) == 0

    conn = sqlite3.connect(world.db)
    remaining = conn.execute(
        "SELECT COUNT(*) FROM patients WHERE label = ?", ("12-12-1912-0",)
    ).fetchone()[0]
    conn.close()
    assert remaining == 0


def test_the_pipeline_job_file_moves_with_the_patient(world):
    _unblock(world)
    assert migrator.run_apply(world, migrator.build_report(world)) == 0

    jobs = Path(world.pipeline_jobs_dir)
    assert not (jobs / "07-07-1907-0.json").exists()
    moved = json.loads((jobs / "PG_07-07-1907.json").read_text())
    # The worker keys its status on this; left behind it reports on a dead ID.
    assert moved["patient_id"] == "PG_07-07-1907"
    assert moved["status"] == "complete"


# --------------------------------------------------------------------------- #
# Evidence instead of a shrug
# --------------------------------------------------------------------------- #


def test_a_mistyped_label_is_matched_back_by_the_bytes_it_holds(tmp_path: Path):
    """`10-7-1963` sits next to `10-07-1963-0` holding the same report file.
    That is answerable from the bytes, so it should reduce to a confirmation."""
    evidence = {
        "10-7-1963": [{"filename": "BB_male.pdf", "sha256": "aaa"}],
        "10-07-1963-0": [
            {"filename": "BB_male.pdf", "sha256": "aaa"},
            {"filename": "10-07-1963-0__BB_male__v1.pdf", "sha256": "aaa"},
        ],
    }

    match = migrator.match_row_by_report_bytes("10-7-1963", evidence, candidates={"10-07-1963-0"})

    assert match["conclusive"] is True
    assert match["matched_patients"] == ["10-07-1963-0"]


def test_a_row_holding_two_patients_reports_is_never_called_conclusive(tmp_path: Path):
    """`4-8-1997` holds one report belonging to BB and one belonging to LJ.
    Folding it into either would move the other patient's file with it."""
    evidence = {
        "4-8-1997": [
            {"filename": "BB_male.pdf", "sha256": "aaa"},
            {"filename": "LJ_TBI.pdf", "sha256": "bbb"},
        ],
        "10-07-1963-0": [{"filename": "BB_male.pdf", "sha256": "aaa"}],
        "04-08-1997-0": [{"filename": "LJ_TBI.pdf", "sha256": "bbb"}],
    }

    match = migrator.match_row_by_report_bytes(
        "4-8-1997", evidence, candidates={"10-07-1963-0", "04-08-1997-0"}
    )

    assert match["conclusive"] is False
    assert match["matched_patients"] == ["04-08-1997-0", "10-07-1963-0"]


def test_placeholder_initials_come_with_a_correction_to_confirm(world):
    """A bare failure makes the operator do the reading. The report's own
    filename already says who this is — propose it and ask."""
    report = migrator.build_report(world)

    finding = next(
        f for f in report["identity_findings"] if f["patient_id"] == "08-08-1908-0"
    )
    assert [p["initials"] for p in finding["proposed"]] == ["CL", "LC"]
    blocker = next(b for b in report["blockers"] if b.startswith("08-08-1908-0:"))
    assert "CL" in blocker and "confirm" in blocker


def test_initials_are_proposed_surname_first_as_the_clinic_names_files():
    assert migrator.propose_initials_from_title("Stubner Helga, initial qeeg.pdf")[0] == {
        "initials": "HS",
        "reading": "Helga Stubner (surname first, as the clinic names files)",
    }
    # One name is not two initials — say what is known, do not invent a pair.
    only_one = migrator.propose_initials_from_title("Snyder mid qeeg.pdf")
    assert only_one[0]["initials"] == ""
    assert "the last initial is S" in only_one[0]["reading"]
    assert "?" not in only_one[0]["reading"]


def test_a_candidate_qa_row_is_not_removed_without_the_owner_saying_so(world):
    """Guessing wrong here deletes a real patient, so the default is to ask."""
    conn = sqlite3.connect(world.db)
    conn.execute(
        "INSERT INTO patients (id, label, notes, created_at, updated_at) "
        "VALUES ('qa-cand', '03-03-1903-0', '', '2026-01-01', '2026-01-01')"
    )
    conn.commit()
    conn.close()

    report = migrator.build_report(world)
    blocker = next(b for b in report["blockers"] if b.startswith("03-03-1903-0:"))
    assert "is this your test data?" in blocker
    assert "03-03-1903-0" not in report["mapping"]

    world.qa_candidates_confirmed = True
    confirmed = migrator.build_report(world)
    assert not any(b.startswith("03-03-1903-0:") for b in confirmed["blockers"])
    assert "03-03-1903-0" not in confirmed["mapping"]


def test_the_share_folder_readme_stops_teaching_the_retired_format(tmp_path: Path):
    from backend import patient_rekey

    portal = tmp_path / "portal_patients"
    portal.mkdir()
    readme = portal / "_README.txt"
    readme.write_text(
        "Create one folder per patient using the standardized ID:\n  MM-DD-YYYY-N\n"
    )

    assert patient_rekey.rewrite_portal_readme(portal) is True

    text = readme.read_text()
    assert "MM-DD-YYYY-N" not in text
    assert "XX_MM-DD-YYYY[_N]" in text
    # Running it twice is not a change.
    assert patient_rekey.rewrite_portal_readme(portal) is False


def test_an_unpadded_twin_reduces_to_a_confirmation_when_the_bytes_agree(world):
    """Do not hand the operator a question the data already answers."""
    report = migrator.build_report(world)

    blocker = next(b for b in report["blockers"] if b.startswith("7-7-1907:"))
    assert "byte-identical to one already filed under 07-07-1907-0" in blocker
    assert "confirmation only" in blocker
    entry = next(
        e for e in report["classified"] if e["label"] == "7-7-1907"
    )
    assert entry["evidence"]["report_byte_match"]["conclusive"] is True


def test_two_patients_sharing_initials_are_both_named_not_one_of_them(world):
    """This clinic has two SF patients born in 1970. Keyed on the token alone,
    one would silently replace the other and a misfiled report would be
    attributed to whichever happened to be classified last — on the one
    decision where files get moved between patients."""
    portal = Path(world.portal_root)
    # A second patient with the same initials as 08-08-1908-0's neighbour.
    _folder(portal, "01-02-1970-0", initials=("S", "F"), files=["SF_one.pdf"])
    _folder(portal, "03-04-1970-0", initials=("S", "F"), files=["SF_two.pdf"])
    _folder(portal, "05-06-1990-0", initials=("Q", "Z"),
            files=["05-06-1990-0__SF_Long_COVID_30_TX__v1__2026-01-01.pdf"])
    conn = sqlite3.connect(world.db)
    for uuid, label in (
        ("s1111111-0000-0000-0000-000000000011", "01-02-1970-0"),
        ("s2222222-0000-0000-0000-000000000012", "03-04-1970-0"),
        ("q3333333-0000-0000-0000-000000000013", "05-06-1990-0"),
    ):
        conn.execute(
            "INSERT INTO patients (id, label, notes, created_at, updated_at) "
            "VALUES (?, ?, '', '2026-01-01', '2026-01-01')",
            (uuid, label),
        )
    conn.commit()
    conn.close()

    report = migrator.build_report(world)

    blocker = next(b for b in report["blockers"] if b.startswith("05-06-1990-0:"))
    assert "either 01-02-1970-0 or 03-04-1970-0" in blocker
    mixed = next(
        m for m in report["mixed_patient_folders"] if m["patient_id"] == "05-06-1990-0"
    )
    assert mixed["foreign_initials"]["SF"]["belongs_to"] == [
        "01-02-1970-0",
        "03-04-1970-0",
    ]


def test_a_confirmed_qa_candidate_is_bundled_and_actually_removed(world):
    """Excluded from the mapping but never bundled and never deleted would leave
    the row sitting in the live database wearing a legacy label — neither
    migrated nor recoverable."""
    conn = sqlite3.connect(world.db)
    conn.execute(
        "INSERT INTO patients (id, label, notes, created_at, updated_at) "
        "VALUES ('qa-cand', '03-03-1903-0', '', '2026-01-01', '2026-01-01')"
    )
    conn.execute(
        "INSERT INTO reports (id, patient_id, filename, stored_path) "
        "VALUES ('qa-cand-r0', 'qa-cand', 'synthetic.pdf', 'data/reports/qa/x.pdf')"
    )
    conn.commit()
    conn.close()
    world.qa_candidates_confirmed = True
    _unblock(world)

    report = migrator.build_report(world)
    assert set(report["qa_fixture_labels"]) == {"12-12-1912-0", "03-03-1903-0"}
    bundled = {p["label"] for p in report["qa_fixture_bundle"]["patients"]}
    assert bundled == {"12-12-1912-0", "03-03-1903-0"}
    assert any(
        r["filename"] == "synthetic.pdf"
        for r in report["qa_fixture_bundle"]["reports"]
    )

    assert migrator.run_apply(world, report) == 0

    conn = sqlite3.connect(world.db)
    left = conn.execute(
        "SELECT COUNT(*) FROM patients WHERE label IN ('03-03-1903-0','12-12-1912-0')"
    ).fetchone()[0]
    orphan_reports = conn.execute(
        "SELECT COUNT(*) FROM reports WHERE patient_id = 'qa-cand'"
    ).fetchone()[0]
    conn.close()
    assert left == 0
    assert orphan_reports == 0


# --------------------------------------------------------------------------- #
# Crashing, resuming, and the guard on live data
# --------------------------------------------------------------------------- #


def test_a_crash_between_the_relabel_and_the_folder_move_is_finished_on_resume(
    world, monkeypatch
):
    """The database label moves before the folder does. A crash in between
    leaves a row that already looks migrated: on the next run it classifies
    already_canonical, drops out of the mapping, and is never revisited — its
    deliverables keep their old names inside the new folder and the run reports
    success. Only the journal knows the work was left half-done."""
    _unblock(world)
    report = migrator.build_report(world)

    real_apply = migrator.patient_rekey.apply_patient_rekey
    crashed = {"done": False}

    def crash_on_the_first_patient(plan, **kwargs):
        if not crashed["done"] and plan.old_id == "07-07-1907-0":
            crashed["done"] = True
            # Move the folder, then die before the journal line is written.
            if plan.folder_move:
                plan.folder_move[0].rename(plan.folder_move[1])
            raise RuntimeError("power cut")
        return real_apply(plan, **kwargs)

    monkeypatch.setattr(
        migrator.patient_rekey, "apply_patient_rekey", crash_on_the_first_patient
    )
    assert migrator.run_apply(world, report) == 1

    portal = Path(world.portal_root)
    stranded = sorted(
        p.name for p in (portal / "PG_07-07-1907").iterdir()
        if p.is_file() and p.name.startswith("07-07-1907-0")
    )
    assert stranded, "the crash should leave legacy-named files behind"

    # Resume: build_report now sees a canonical label and no longer offers it.
    monkeypatch.setattr(migrator.patient_rekey, "apply_patient_rekey", real_apply)
    resumed = migrator.build_report(world)
    assert "07-07-1907-0" not in resumed["mapping"]

    assert migrator.run_apply(world, resumed) == 0

    # The patient is actually finished, not merely reported as finished.
    left = sorted(
        p.name for p in (portal / "PG_07-07-1907").iterdir()
        if p.is_file() and p.name.startswith("07-07-1907-0")
    )
    assert left == []
    assert json.loads((portal / "PG_07-07-1907" / "$meta.json").read_text())[
        "patientId"
    ] == "PG_07-07-1907"


def test_a_fixtures_apply_cannot_reach_the_live_conversations_directory(
    world, capsys
):
    """--conversations-dir defaults to a production root, so an apply naming
    only a fixture --db and --portal-root used to pass the guard and then
    rewrite the clinic's live conversation files."""
    exit_code = migrator.main(
        [
            "--apply",
            "--window-confirmed",
            "--db", world.db,
            "--portal-root", world.portal_root,
            # not passed: --conversations-dir, so it takes the production default
        ]
    )

    assert exit_code == 2
    assert "conversations directory" in capsys.readouterr().err


def test_an_id_already_worn_or_reserved_is_never_handed_out_again(world):
    """Once Tasks 1-4 deploy, a patient can arrive canonical before the window.
    Allocating only within the migrate set would hand their ID to someone else."""
    conn = sqlite3.connect(world.db)
    conn.execute(
        "INSERT INTO patients (id, label, notes, created_at, updated_at) "
        "VALUES ('early', 'PG_07-07-1907', '', '2026-01-01', '2026-01-01')"
    )
    conn.execute(
        "INSERT INTO patient_id_reservations (patient_id) VALUES ('PG_07-07-1907_2')"
    )
    conn.commit()
    conn.close()

    report = migrator.build_report(world)

    # The migrating patient must not be given the ID somebody already wears,
    # nor the one already retired behind it.
    assert report["mapping"]["07-07-1907-0"] == "PG_07-07-1907_3"


def test_the_qa_records_files_are_kept_before_the_folder_is_removed(world):
    """A list of digests is not a rollback bundle."""
    portal = Path(world.portal_root)
    _folder(portal, "12-12-1912-0", initials=("D", "M"),
            files=["12-12-1912-0__patient-facing__v1__2026-08-02.pdf"])
    _unblock(world)

    report = migrator.build_report(world)
    assert migrator.run_apply(world, report) == 0

    assert not (portal / "12-12-1912-0").exists()
    kept = Path(world.rollback_bundle) / "qa-fixture-artifacts" / "12-12-1912-0"
    survivor = kept / "12-12-1912-0__patient-facing__v1__2026-08-02.pdf"
    # The bytes survive the removal, not just a record that they existed.
    assert survivor.is_file()
    assert survivor.read_bytes().startswith(b"%PDF-1.4")


def test_apply_refuses_to_remove_qa_records_with_nowhere_to_keep_them(world, capsys):
    _unblock(world)
    world.rollback_bundle = ""

    assert migrator.run_apply(world, migrator.build_report(world)) == 1
    assert "--rollback-bundle" in capsys.readouterr().err


def test_a_patient_finished_on_resume_still_reaches_the_hub(world, monkeypatch):
    """C3's fix moved the failure one layer out: a resumed patient is already
    canonical, so the recomputed mapping omits them — and so did the remote
    worklist. Their hub blobs would have stayed under the retired prefix while
    the run reported success. The journal is what remembers."""
    _unblock(world)
    report = migrator.build_report(world)

    real_apply = migrator.patient_rekey.apply_patient_rekey
    crashed = {"done": False}

    def crash_on_the_first_patient(plan, **kwargs):
        if not crashed["done"] and plan.old_id == "07-07-1907-0":
            crashed["done"] = True
            if plan.folder_move:
                plan.folder_move[0].rename(plan.folder_move[1])
            raise RuntimeError("power cut")
        return real_apply(plan, **kwargs)

    monkeypatch.setattr(
        migrator.patient_rekey, "apply_patient_rekey", crash_on_the_first_patient
    )
    migrator.run_apply(world, report)

    monkeypatch.setattr(migrator.patient_rekey, "apply_patient_rekey", real_apply)
    resumed = migrator.build_report(world)
    assert "07-07-1907-0" not in resumed["mapping"]
    assert migrator.run_apply(world, resumed) == 0

    worklist_dir = Path(world.journal).parent
    mapping_file = json.loads(
        (worklist_dir / "remote-rekey-mapping.json").read_text()
    )
    worklist = json.loads((worklist_dir / "remote-rekey-worklist.json").read_text())

    # The resumed patient is in both artifacts the hub rekey consumes.
    assert mapping_file["07-07-1907-0"] == "PG_07-07-1907"
    assert any(item["patientIdOld"] == "07-07-1907-0" for item in worklist)


def test_the_mapping_file_is_the_shape_the_hub_rekey_consumes(world):
    """The manifest is the whole report and the worklist is an array; neither
    is a mapping, and feeding either to the rekey exited 0 having done nothing."""
    _unblock(world)
    report = migrator.build_report(world)
    assert migrator.run_apply(world, report) == 0

    mapping_file = json.loads(
        (Path(world.journal).parent / "remote-rekey-mapping.json").read_text()
    )

    assert isinstance(mapping_file, dict) and mapping_file
    for old_id, new_id in mapping_file.items():
        assert isinstance(new_id, str) and old_id != new_id
        assert migrator.parse_canonical_patient_id(new_id) is not None
        assert migrator.parse_legacy_id(old_id) is not None


def test_apply_without_qa_records_does_not_need_a_rollback_bundle(world):
    """Failing here loses the worklist, which is written afterwards — and there
    was nothing to remove in the first place."""
    _unblock(world)
    world.rollback_bundle = ""
    conn = sqlite3.connect(world.db)
    conn.execute("DELETE FROM patients WHERE label = '12-12-1912-0'")
    conn.commit()
    conn.close()

    report = migrator.build_report(world)
    assert report["qa_fixture_labels"] == []
    assert migrator.run_apply(world, report) == 0
    assert (Path(world.journal).parent / "remote-rekey-worklist.json").is_file()


def test_a_patient_whose_rekey_failed_never_reaches_the_hub_artifacts(
    world, monkeypatch
):
    """The intent line is written before the work is attempted, so it exists
    even for a patient that failed immediately — nothing local touched, no
    reservation taken. Driving the hub off intents would hand the operator a
    command that moves those blobs, stranding the patient the other way round:
    hub canonical, local still legacy."""
    _unblock(world)
    report = migrator.build_report(world)

    real_plan = migrator.patient_rekey.plan_patient_rekey

    def fail_one_patient_before_anything_moves(old_id, new_id, **kwargs):
        if old_id == "07-07-1907-0":
            raise RuntimeError("disk went away")
        return real_plan(old_id, new_id, **kwargs)

    monkeypatch.setattr(
        migrator.patient_rekey,
        "plan_patient_rekey",
        fail_one_patient_before_anything_moves,
    )

    assert migrator.run_apply(world, report) == 1

    worklist_dir = Path(world.journal).parent
    mapping_file = json.loads((worklist_dir / "remote-rekey-mapping.json").read_text())
    worklist = json.loads((worklist_dir / "remote-rekey-worklist.json").read_text())

    # The failed patient is absent from both artifacts the operator would paste.
    assert "07-07-1907-0" not in mapping_file
    assert all(item["patientIdOld"] != "07-07-1907-0" for item in worklist)
    # Its portal folder is genuinely untouched, which is why the hub must be too.
    assert (Path(world.portal_root) / "07-07-1907-0").is_dir()

    # A sibling that succeeded in the same run is still there.
    assert mapping_file["09-09-1909-0"] == "DK_09-09-1909"


# --------------------------------------------------------------------------- #
# The clinic's answers
# --------------------------------------------------------------------------- #


def _answers(world, payload: dict) -> None:
    path = Path(world.portal_root).parent / "answers.json"
    path.write_text(json.dumps(payload))
    world.answers = str(path)


def test_an_unknown_initial_is_X_and_never_a_blocker(world):
    """The clinic corrects an X whenever they like through a normal relabel.
    Not knowing a letter is not a reason to stop a migration."""
    _answers(world, {"initials": {
        "10-10-1910-0": {"first": "X", "last": "X"},
        "11-11-1911-0": {"first": "X", "last": "X"},
    }})

    report = migrator.build_report(world)

    assert report["mapping"]["10-10-1910-0"] == "XX_10-10-1910"
    assert not any(b.startswith("10-10-1910-0:") for b in report["blockers"])
    # And it is a real canonical id, not a special case downstream.
    assert migrator.parse_canonical_patient_id("XX_10-10-1910") is not None


def test_a_clinic_answer_outranks_the_name_on_the_document(world):
    """The clinic is the authority on who their patients are. A stored pair the
    dry run flagged as a placeholder is simply replaced by what they say."""
    _answers(world, {"initials": {"08-08-1908-0": {"first": "C", "last": "L"}}})

    report = migrator.build_report(world)

    # The folder stores BT and the report is named 'L, Connor final qeeg.pdf';
    # neither gets a vote once the clinic has answered.
    assert report["mapping"]["08-08-1908-0"] == "CL_08-08-1908"
    assert not any(b.startswith("08-08-1908-0:") for b in report["blockers"])


def test_a_merged_row_moves_its_work_to_the_survivor(world):
    _answers(world, {
        "initials": {"10-10-1910-0": {"first": "J", "last": "M"},
                     "08-08-1908-0": {"first": "C", "last": "L"},
                     "11-11-1911-0": {"first": "X", "last": "X"}},
        "dissolve": {"7-7-1907": {"note": "a copy filed elsewhere"}},
        "merge_into": {"10-10-1910-0": {"survivor": "07-07-1907-0"}},
    })

    report = migrator.build_report(world)

    merged = next(
        e for e in report["classified"] if e["label"] == "10-10-1910-0"
    )
    assert merged["bucket"] == migrator.BUCKET_DUPLICATE
    assert merged["survivor_uuid"].startswith("aaaaaaaa")
    assert "10-10-1910-0" not in report["mapping"]


def test_a_row_that_was_never_a_chart_is_retired_not_migrated(world):
    _answers(world, {
        "initials": {"10-10-1910-0": {"first": "J", "last": "M"},
                     "08-08-1908-0": {"first": "C", "last": "L"},
                     "11-11-1911-0": {"first": "X", "last": "X"}},
        "dissolve": {"7-7-1907": {"note": "a copy filed elsewhere"}},
    })

    report = migrator.build_report(world)

    dissolved = next(e for e in report["classified"] if e["label"] == "7-7-1907")
    assert dissolved["bucket"] == migrator.BUCKET_DISSOLVED
    assert "7-7-1907" not in report["mapping"]
    assert not any(b.startswith("7-7-1907:") for b in report["blockers"])


def test_a_split_folder_moves_each_report_to_the_patient_it_belongs_to(world):
    """The residual owner keeps the folder; everybody else's reports go home."""
    _answers(world, {
        "initials": {"10-10-1910-0": {"first": "J", "last": "M"},
                     "08-08-1908-0": {"first": "C", "last": "L"},
                     "11-11-1911-0": {"first": "X", "last": "X"}},
        "dissolve": {"7-7-1907": {"note": "a copy filed elsewhere"}},
        "split_misfiled": {"11-11-1911-0": {"note": "holds DK's reports"}},
    })

    report = migrator.build_report(world)
    assert not any(b.startswith("11-11-1911-0:") for b in report["blockers"])
    assert report["mapping"]["11-11-1911-0"] == "XX_11-11-1911"

    assert migrator.run_apply(world, report) == 0

    portal = Path(world.portal_root)
    # DK's misfiled report is now under DK's canonical folder...
    assert (
        portal / "DK_09-09-1909"
        / "11-11-1911-0__DK_20Tx_toxic-brain-injury__v1__2026-05-09.pdf"
    ).is_file()
    # ...and the residual owner kept the folder, with their own file in it.
    assert (portal / "XX_11-11-1911" / "XX_11-11-1911__patient-facing__v1__2026-05-09.pdf").is_file()


def test_a_patient_with_no_share_folder_is_given_one(world):
    """Nothing of theirs could ever be published otherwise."""
    _answers(world, {"initials": {
        "10-10-1910-0": {"first": "X", "last": "X"},
        "11-11-1911-0": {"first": "X", "last": "X"},
    }})
    conn = sqlite3.connect(world.db)
    conn.execute(
        "INSERT INTO patients (id, label, notes, created_at, updated_at) "
        "VALUES ('nofolder', '01-01-1913-0', '', '2026-01-01', '2026-01-01')"
    )
    conn.commit()
    conn.close()
    _answers(world, {"initials": {
        "10-10-1910-0": {"first": "X", "last": "X"},
        "11-11-1911-0": {"first": "X", "last": "X"},
        "01-01-1913-0": {"first": "X", "last": "X"},
    }})
    _unblock(world)

    report = migrator.build_report(world)
    assert report["mapping"]["01-01-1913-0"] == "XX_01-01-1913"
    assert migrator.run_apply(world, report) == 0

    assert (Path(world.portal_root) / "XX_01-01-1913").is_dir()


# --------------------------------------------------------------------------- #
# The line between a question the file answers and one it does not
# --------------------------------------------------------------------------- #


def _patient_with_a_named_report(world, label, initials, title):
    """A patient whose chart stores initials and whose report carries a name."""
    _folder(Path(world.portal_root), label, initials=initials)
    conn = sqlite3.connect(world.db)
    conn.execute(
        "INSERT INTO patients (id, label, notes, created_at, updated_at) "
        "VALUES (?, ?, '', '2026-01-01', '2026-01-01')",
        (f"uuid-{label}", label),
    )
    conn.commit()
    conn.close()
    (Path(world.conversations_dir) / f"conv-{label}.json").write_text(
        json.dumps(
            {
                "id": f"conv-{label}",
                "patient_label": label,
                "title": title,
                "messages": [],
                "artifacts": [],
            }
        )
    )


def test_a_surname_that_matches_the_stored_last_initial_is_not_a_question(world):
    """`Knowles intial qeeg.pdf` says nothing about a first name, and stored
    `LK` is exactly what `L. Knowles` looks like. Asking about it asks the
    operator something the file has already answered."""
    _patient_with_a_named_report(
        world, "02-02-1902-0", ("L", "K"), "Knowles intial qeeg.pdf"
    )

    report = migrator.build_report(world)

    assert not any(b.startswith("02-02-1902-0:") for b in report["blockers"])
    assert report["mapping"]["02-02-1902-0"] == "LK_02-02-1902"


def test_two_known_names_in_the_wrong_order_is_still_a_question(world):
    """`Stubner Helga` names both, and they read as HS. A chart storing SN is
    not consistent with that, and no amount of quieting the surname-only case
    may swallow it."""
    _patient_with_a_named_report(
        world, "04-04-1904-0", ("S", "N"), "Stubner Helga, initial qeeg.pdf"
    )

    report = migrator.build_report(world)

    blocker = next(b for b in report["blockers"] if b.startswith("04-04-1904-0:"))
    assert "SN" in blocker and "Stubner Helga" in blocker
    assert "HS" in blocker
    assert "04-04-1904-0" not in report["mapping"]


# --------------------------------------------------------------------------- #
# A patient who owns a legacy folder AND received split-in files
# --------------------------------------------------------------------------- #


def _split_recipient_sorting_after_the_mixed_folder(world):
    """A patient whose legacy label sorts AFTER the mixed folder's.

    Order is the whole defect: the mixed folder is processed first, its split
    creates the recipient's canonical directory, and only then does the
    recipient's own rekey try to rename their legacy folder onto it.
    """
    portal = Path(world.portal_root)
    _folder(portal, "12-12-1922-0", initials=("Z", "W"),
            files=["ZW_own_report.pdf"])
    (portal / "11-11-1911-0"
     / "11-11-1911-0__ZW_misfiled__v1__2026-01-01.pdf").write_bytes(b"%PDF-1.4 zw")
    conn = sqlite3.connect(world.db)
    conn.execute(
        "INSERT INTO patients (id, label, notes, created_at, updated_at) "
        "VALUES ('zw-uuid', '12-12-1922-0', '', '2026-01-01', '2026-01-01')"
    )
    conn.commit()
    conn.close()


def test_a_patient_who_received_split_files_still_gets_their_own_folder(world):
    """The shape that stopped the real window.

    Un-misfiling created the recipient's canonical directory to receive a report
    split out of the mixed folder. Their own rekey then tried to rename their
    legacy folder onto that directory and refused, because renaming would have
    had to destroy one side. Their folder joins the destination instead, file by
    file, and nothing is written over.
    """
    _split_recipient_sorting_after_the_mixed_folder(world)
    _answers(world, {
        "initials": {"10-10-1910-0": {"first": "J", "last": "M"},
                     "08-08-1908-0": {"first": "C", "last": "L"},
                     "11-11-1911-0": {"first": "X", "last": "X"},
                     "12-12-1922-0": {"first": "Z", "last": "W"}},
        "dissolve": {"7-7-1907": {"note": "a copy filed elsewhere"}},
        "split_misfiled": {"11-11-1911-0": {"note": "holds ZW's report"}},
    })

    report = migrator.build_report(world)
    assert migrator.run_apply(world, report) == 0

    portal = Path(world.portal_root)
    landed = sorted(p.name for p in (portal / "ZW_12-12-1922").iterdir() if p.is_file())

    # Both sides are there: the split-in report and their own file.
    assert "11-11-1911-0__ZW_misfiled__v1__2026-01-01.pdf" in landed
    assert "ZW_own_report.pdf" in landed
    # And the legacy folder is gone rather than left behind.
    assert not (portal / "12-12-1922-0").exists()


def test_a_destination_holding_anything_this_run_did_not_put_there_is_refused(world):
    """Merging is only ever safe for what this migration put there itself.

    A destination carrying anything else is a real collision, and joining the
    two folders would silently mix two patients' work. It fails that patient by
    name and touches neither side.
    """
    _split_recipient_sorting_after_the_mixed_folder(world)
    _answers(world, {
        "initials": {"10-10-1910-0": {"first": "J", "last": "M"},
                     "08-08-1908-0": {"first": "C", "last": "L"},
                     "11-11-1911-0": {"first": "X", "last": "X"},
                     "12-12-1922-0": {"first": "Z", "last": "W"}},
        "dissolve": {"7-7-1907": {"note": "a copy filed elsewhere"}},
        "split_misfiled": {"11-11-1911-0": {"note": "holds ZW's report"}},
    })
    portal = Path(world.portal_root)
    # Something this run did not write is already standing at the destination.
    (portal / "ZW_12-12-1922").mkdir()
    (portal / "ZW_12-12-1922" / "someone-elses-work.pdf").write_bytes(b"%PDF-1.4 x")

    report = migrator.build_report(world)

    # The dry run stops first: a folder standing on a canonical destination with
    # no patient behind it is named before anything is allowed to move.
    assert any("ZW_12-12-1922" in b for b in report["blockers"]), report["blockers"]
    assert migrator.run_apply(world, report) == 1

    # Neither side was touched: their legacy folder stands, and the stranger's
    # file is exactly where it was.
    assert (portal / "12-12-1922-0" / "ZW_own_report.pdf").is_file()
    assert (portal / "ZW_12-12-1922" / "someone-elses-work.pdf").is_file()


def test_the_merge_gate_itself_refuses_anything_it_did_not_write(tmp_path: Path):
    """The inner gate, on its own: merging is permitted only when every file at
    the destination is one this run's split put there."""
    portal = tmp_path / "portal_patients"
    (portal / "ZW_12-12-1922").mkdir(parents=True)
    (portal / "ZW_12-12-1922" / "split-in.pdf").write_bytes(b"%PDF-1.4 a")

    ours = {"split-in.pdf"}
    assert migrator.target_is_this_runs_own_work(portal, "ZW_12-12-1922", ours) == (
        True,
        [],
    )

    # One file nobody can account for is enough to refuse the whole merge.
    (portal / "ZW_12-12-1922" / "stranger.pdf").write_bytes(b"%PDF-1.4 b")
    allowed, unaccounted = migrator.target_is_this_runs_own_work(
        portal, "ZW_12-12-1922", ours
    )
    assert allowed is False
    assert unaccounted == ["stranger.pdf"]

    # And a destination nothing was split into is never mergeable, even if it
    # happens to be empty of surprises.
    assert migrator.target_is_this_runs_own_work(portal, "ZW_12-12-1922", set()) == (
        False,
        ["split-in.pdf", "stranger.pdf"],
    )


def test_the_split_is_journalled_so_a_later_run_can_verify_it(world):
    """A resumed run must be able to tell a folder this migration created from
    one that was already there — which is not something to infer from names."""
    _answers(world, {
        "initials": {"10-10-1910-0": {"first": "J", "last": "M"},
                     "08-08-1908-0": {"first": "C", "last": "L"},
                     "11-11-1911-0": {"first": "X", "last": "X"}},
        "dissolve": {"7-7-1907": {"note": "a copy filed elsewhere"}},
        "split_misfiled": {"11-11-1911-0": {"note": "holds DK's report"}},
    })
    assert migrator.run_apply(world, migrator.build_report(world)) == 0

    splits = [
        r for r in Journal_lines(Path(world.journal))
        if r.get("kind") == "split_moved"
    ]
    assert splits, "the split must be on the record"
    assert splits[0]["owner"] == "09-09-1909-0"
    assert "11-11-1911-0__DK_20Tx_toxic-brain-injury__v1__2026-05-09.pdf" in (
        splits[0]["files"]
    )
