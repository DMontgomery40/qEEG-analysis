"""Canonical clinic patient ID contract: XX_MM-DD-YYYY[_N].

The engine keeps its SQLite UUID as an invisible relational primary key. The
canonical clinic ID is what folders, filenames, sync keys, and clinic staff see,
and it lives in the patient label column.
"""

from __future__ import annotations

import sqlite3
import threading
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select
from sqlalchemy.orm import Session

from backend import patient_identity as pid
from backend.patient_identity import (
    PatientIdentityError,
    allocate_canonical_patient_id,
    canonical_patient_id,
    derive_initial,
    parse_canonical_patient_id,
)


# ---------------------------------------------------------------- parser


def test_parse_accepts_base_canonical_id():
    parsed = parse_canonical_patient_id("BT_12-11-1963")

    assert parsed is not None
    assert parsed.value == "BT_12-11-1963"
    assert parsed.first_initial == "B"
    assert parsed.last_initial == "T"
    assert parsed.birthdate == "12-11-1963"
    assert parsed.ordinal == 1


@pytest.mark.parametrize(
    "value,ordinal",
    [
        ("BT_12-11-1963_2", 2),
        ("BT_12-11-1963_9", 9),
        ("BT_12-11-1963_10", 10),
        ("BT_12-11-1963_11", 11),
        ("BT_12-11-1963_100", 100),
    ],
)
def test_parse_accepts_collision_suffixes(value, ordinal):
    parsed = parse_canonical_patient_id(value)

    assert parsed is not None
    assert parsed.ordinal == ordinal
    assert parsed.value == value


def test_parse_rejects_ordinal_one_suffix():
    """`_1` never exists — ordinal 1 is the unsuffixed base form."""
    assert parse_canonical_patient_id("BT_12-11-1963_1") is None


def test_parse_accepts_leap_day_birthdate():
    parsed = parse_canonical_patient_id("DM_02-29-1984")

    assert parsed is not None
    assert parsed.birthdate == "02-29-1984"


@pytest.mark.parametrize(
    "value",
    [
        "BT_02-30-1984",  # February never has 30 days
        "BT_02-29-1983",  # 1983 is not a leap year
        "BT_13-01-1990",  # month 13
        "BT_00-10-1990",  # month 0
        "BT_12-32-1990",  # day 32
        "BT_12-00-1990",  # day 0
        "BT_04-31-1990",  # April has 30 days
    ],
)
def test_parse_rejects_impossible_calendar_dates(value):
    assert parse_canonical_patient_id(value) is None


@pytest.mark.parametrize("value", ["BT_12-11-1900", "BT_12-11-2100"])
def test_parse_accepts_years_inside_the_plausible_range(value):
    assert parse_canonical_patient_id(value) is not None


@pytest.mark.parametrize(
    "value",
    [
        "BT_01-01-0001",  # a mistyped year must not earn a permanent reservation
        "BT_12-11-1899",
        "BT_12-11-2101",
        "BT_12-11-9999",
    ],
)
def test_parse_rejects_years_outside_the_plausible_range(value):
    """Matches the 1900-2100 bound the legacy portal normalizer already uses."""
    assert parse_canonical_patient_id(value) is None


@pytest.mark.parametrize(
    "value",
    [
        "12-11-1963-0",
        "09-05-1954-1",
        "05-13-1947-0",
        "02-29-1984-0",
    ],
)
def test_parse_rejects_legacy_dob_keys(value):
    """The legacy MM-DD-YYYY-N storage key is not a canonical clinic ID."""
    assert parse_canonical_patient_id(value) is None


@pytest.mark.parametrize(
    "value",
    [
        "",
        "   ",
        "bt_12-11-1963",  # initials must be uppercase A-Z
        "B_12-11-1963",  # one initial
        "BTS_12-11-1963",  # three initials
        "B7_12-11-1963",  # digit as an initial
        "BT 12-11-1963",  # space instead of underscore
        "BT_12.11.1963",  # periods instead of dashes
        "BT-12-11-1963",  # dash instead of underscore
        "BT_1-5-1963",  # month/day not zero padded
        "BT_12-11-63",  # two digit year
        "BT_12-11-1963_0",
        "BT_12-11-1963_02",  # leading zero in the ordinal
        "BT_12-11-1963_",
        " BT_12-11-1963",  # no surrounding whitespace tolerated
        "BT_12-11-1963 ",
    ],
)
def test_parse_rejects_malformed_values(value):
    assert parse_canonical_patient_id(value) is None


def test_parse_rejects_none():
    assert parse_canonical_patient_id(None) is None


# ---------------------------------------------------------------- builder


def test_canonical_patient_id_ordinal_one_is_unsuffixed():
    assert canonical_patient_id("B", "T", "12-11-1963") == "BT_12-11-1963"
    assert canonical_patient_id("B", "T", "12-11-1963", ordinal=1) == "BT_12-11-1963"


@pytest.mark.parametrize(
    "ordinal,expected",
    [
        (2, "BT_12-11-1963_2"),
        (9, "BT_12-11-1963_9"),
        (10, "BT_12-11-1963_10"),
        (11, "BT_12-11-1963_11"),
    ],
)
def test_canonical_patient_id_appends_collision_suffix(ordinal, expected):
    assert canonical_patient_id("B", "T", "12-11-1963", ordinal=ordinal) == expected


def test_canonical_patient_id_normalizes_case_and_zero_pads_the_date():
    assert canonical_patient_id("b", "t", "1-5-1990") == "BT_01-05-1990"


@pytest.mark.parametrize("ordinal", [0, -1, -10])
def test_canonical_patient_id_rejects_nonpositive_ordinals(ordinal):
    with pytest.raises(PatientIdentityError):
        canonical_patient_id("B", "T", "12-11-1963", ordinal=ordinal)


@pytest.mark.parametrize(
    "first,last,birthdate",
    [
        ("B", "T", "02-30-1984"),
        ("B", "T", "13-01-1990"),
        ("B", "T", "not-a-date"),
        ("B", "T", ""),
        ("B", "T", "01-01-0001"),
        ("B", "T", "12-11-1899"),
        ("B", "T", "12-11-2101"),
        ("", "T", "12-11-1963"),
        ("B", "", "12-11-1963"),
        ("BT", "T", "12-11-1963"),
        ("7", "T", "12-11-1963"),
        ("Ñ", "T", "12-11-1963"),
    ],
)
def test_canonical_patient_id_rejects_invalid_identity(first, last, birthdate):
    with pytest.raises(PatientIdentityError):
        canonical_patient_id(first, last, birthdate)


@pytest.mark.parametrize("ordinal", [1, 2, 9, 10, 47])
def test_built_ids_round_trip_through_the_parser(ordinal):
    value = canonical_patient_id("D", "M", "02-29-1984", ordinal=ordinal)
    parsed = parse_canonical_patient_id(value)

    assert parsed is not None
    assert parsed.ordinal == ordinal
    assert parsed.first_initial == "D"
    assert parsed.last_initial == "M"
    assert parsed.birthdate == "02-29-1984"


# ---------------------------------------------------------------- initials


@pytest.mark.parametrize(
    "name,expected",
    [
        ("Peña", "P"),
        ("Ñunez", "N"),
        ("Élodie", "E"),
        ("Ångström", "A"),
        ("Çelik", "C"),
        ("O'Brien", "O"),
        ("  ana  ", "A"),
        ("de la Cruz", "D"),
    ],
)
def test_derive_initial_normalizes_unicode_to_an_ascii_letter(name, expected):
    assert derive_initial(name) == expected


@pytest.mark.parametrize(
    "name",
    [
        "Ørsted",  # NFKD does not decompose the stroke
        "Łukasz",  # NFKD does not decompose the stroke
        "李",  # no A-Z letter at all
        "Ω",
        "9lives",
        "",
        "   ",
        None,
    ],
)
def test_derive_initial_asks_instead_of_guessing(name):
    """When no unambiguous A-Z letter results, the caller must supply the initial."""
    with pytest.raises(PatientIdentityError) as excinfo:
        derive_initial(name)

    assert "initial" in str(excinfo.value).lower()


def test_derive_initial_only_reads_the_first_character():
    """`Ørsted` must not silently become `R` by scanning past the first letter."""
    with pytest.raises(PatientIdentityError):
        derive_initial("Ørsted")


# ---------------------------------------------------------------- allocation


@pytest.fixture
def db(tmp_path: Path):
    from backend import storage

    storage.reset_engine(f"sqlite:///{tmp_path / 'app.db'}")
    storage.init_db()
    return storage


def _allocate(storage, **kwargs) -> str:
    with storage.session_scope() as session:
        return allocate_canonical_patient_id(session, **kwargs)


def test_allocate_issues_the_unsuffixed_id_first(db):
    assert (
        _allocate(db, first_initial="B", last_initial="T", birthdate="12-11-1963")
        == "BT_12-11-1963"
    )


def test_allocate_records_a_durable_reservation(db):
    value = _allocate(db, first_initial="B", last_initial="T", birthdate="12-11-1963")

    with db.session_scope() as session:
        row = session.get(db.PatientIdReservation, value)

    assert row is not None
    assert row.ordinal == 1
    assert row.birthdate == "12-11-1963"


def test_allocate_skips_an_id_already_worn_by_a_patient(db):
    with db.session_scope() as session:
        db.create_patient(session, label="BT_12-11-1963", notes="")

    assert (
        _allocate(db, first_initial="B", last_initial="T", birthdate="12-11-1963")
        == "BT_12-11-1963_2"
    )


def test_allocate_skips_a_case_differing_patient_label(db):
    with db.session_scope() as session:
        db.create_patient(session, label="bt_12-11-1963", notes="")

    assert (
        _allocate(db, first_initial="B", last_initial="T", birthdate="12-11-1963")
        == "BT_12-11-1963_2"
    )


def test_allocate_never_reuses_a_reservation_after_the_patient_is_gone(db):
    first = _allocate(db, first_initial="B", last_initial="T", birthdate="12-11-1963")
    assert first == "BT_12-11-1963"

    # No patient row was ever created against that reservation, so only the
    # reservation table remembers it. It must still never be handed out again.
    second = _allocate(db, first_initial="B", last_initial="T", birthdate="12-11-1963")

    assert second == "BT_12-11-1963_2"


def test_allocate_never_reuses_an_id_freed_by_a_relabel(db):
    value = _allocate(db, first_initial="B", last_initial="T", birthdate="12-11-1963")
    with db.session_scope() as session:
        patient = db.create_patient(session, label=value, notes="")
        db.update_patient(session, patient.id, label="ZZ_01-01-1900", notes="")

    assert (
        _allocate(db, first_initial="B", last_initial="T", birthdate="12-11-1963")
        == "BT_12-11-1963_2"
    )


def test_allocate_is_a_no_op_when_a_patient_keeps_its_own_identity(db):
    value = _allocate(db, first_initial="B", last_initial="T", birthdate="12-11-1963")
    with db.session_scope() as session:
        patient = db.create_patient(session, label=value, notes="")

    again = _allocate(
        db,
        first_initial="B",
        last_initial="T",
        birthdate="12-11-1963",
        exclude_patient_uuid=patient.id,
    )

    assert again == value


def test_allocate_walks_past_nine_into_double_digits(db):
    issued = [
        _allocate(db, first_initial="B", last_initial="T", birthdate="12-11-1963")
        for _ in range(10)
    ]

    assert issued[0] == "BT_12-11-1963"
    assert issued[1] == "BT_12-11-1963_2"
    assert issued[8] == "BT_12-11-1963_9"
    assert issued[9] == "BT_12-11-1963_10"
    assert "BT_12-11-1963_1" not in issued


def test_allocate_separates_different_birthdates_and_initials(db):
    a = _allocate(db, first_initial="B", last_initial="T", birthdate="12-11-1963")
    b = _allocate(db, first_initial="B", last_initial="T", birthdate="12-12-1963")
    c = _allocate(db, first_initial="B", last_initial="R", birthdate="12-11-1963")

    assert (a, b, c) == ("BT_12-11-1963", "BT_12-12-1963", "BR_12-11-1963")


def test_allocate_rejects_identity_it_cannot_validate(db):
    with pytest.raises(PatientIdentityError):
        _allocate(db, first_initial="B", last_initial="T", birthdate="02-30-1984")


def test_allocate_survives_losing_the_race_to_another_writer(db, monkeypatch):
    """The reservation primary key, not the pre-scan, is what prevents reissue.

    Simulates a competing writer that claims the base ID in the window between
    this allocator's scan and its own insert: the scan reports the ID free, the
    insert loses, and allocation must fall through to the next ordinal rather
    than raising or handing out a duplicate.
    """
    with db.session_scope() as session:
        session.add(
            db.PatientIdReservation(
                patient_id="BT_12-11-1963",
                first_initial="B",
                last_initial="T",
                birthdate="12-11-1963",
                ordinal=1,
            )
        )
        session.commit()

    monkeypatch.setattr(pid, "_canonical_id_is_taken", lambda *a, **k: False)

    assert (
        _allocate(db, first_initial="B", last_initial="T", birthdate="12-11-1963")
        == "BT_12-11-1963_2"
    )


def test_confirming_an_unreserved_id_a_patient_already_wears_reserves_it(db):
    """A canonical-looking label can reach the database without a reservation.

    Bulk upload mints a label from the filename stem, so `BT_12-11-1963.pdf`
    produces a patient wearing that ID with nothing reserved behind it.
    Confirming that patient's identity has to close the gap.
    """
    with db.session_scope() as session:
        patient = db.create_patient(session, label="BT_12-11-1963", notes="")
        assert session.get(db.PatientIdReservation, "BT_12-11-1963") is None

    confirmed = _allocate(
        db,
        first_initial="B",
        last_initial="T",
        birthdate="12-11-1963",
        exclude_patient_uuid=patient.id,
    )

    assert confirmed == "BT_12-11-1963"
    with db.session_scope() as session:
        assert session.get(db.PatientIdReservation, "BT_12-11-1963") is not None


def test_an_id_vacated_after_confirmation_never_reaches_another_person(db):
    """The whole chain: unreserved label, confirmed, relabelled away, reissued."""
    with db.session_scope() as session:
        patient = db.create_patient(session, label="BT_12-11-1963", notes="")

    _allocate(
        db,
        first_initial="B",
        last_initial="T",
        birthdate="12-11-1963",
        exclude_patient_uuid=patient.id,
    )

    with db.session_scope() as session:
        db.update_patient(session, patient.id, label="ZZ_01-01-1900", notes="")

    assert (
        _allocate(db, first_initial="B", last_initial="T", birthdate="12-11-1963")
        == "BT_12-11-1963_2"
    )


def test_losing_the_race_keeps_the_callers_pending_work(db, monkeypatch):
    """A lost race must not discard writes the caller had pending.

    Task 5's migrator holds a pending patient mutation while it allocates, so a
    rollback here would silently throw away work already done.
    """
    with db.session_scope() as session:
        session.add(
            db.PatientIdReservation(
                patient_id="BT_12-11-1963",
                first_initial="B",
                last_initial="T",
                birthdate="12-11-1963",
                ordinal=1,
            )
        )
        session.commit()

    monkeypatch.setattr(pid, "_canonical_id_is_taken", lambda *a, **k: False)

    with Session(db.engine, expire_on_commit=False) as session:
        session.add(db.Patient(id="pending-uuid", label="work in progress", notes=""))
        session.flush()

        value = allocate_canonical_patient_id(
            session, first_initial="B", last_initial="T", birthdate="12-11-1963"
        )

        assert value == "BT_12-11-1963_2"
        assert session.get(db.Patient, "pending-uuid") is not None
        session.commit()

    with db.session_scope() as session:
        assert session.get(db.Patient, "pending-uuid") is not None


def test_concurrent_allocators_never_issue_the_same_id(db):
    issued: list[str] = []
    errors: list[BaseException] = []
    lock = threading.Lock()
    barrier = threading.Barrier(6)

    def worker() -> None:
        try:
            barrier.wait(timeout=10)
            with Session(db.engine, expire_on_commit=False) as session:
                value = allocate_canonical_patient_id(
                    session,
                    first_initial="B",
                    last_initial="T",
                    birthdate="12-11-1963",
                )
            with lock:
                issued.append(value)
        except BaseException as exc:  # noqa: BLE001 - surfaced below
            with lock:
                errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)

    assert errors == []
    assert len(issued) == 6
    assert len(set(issued)) == 6
    assert set(issued) == {
        "BT_12-11-1963",
        "BT_12-11-1963_2",
        "BT_12-11-1963_3",
        "BT_12-11-1963_4",
        "BT_12-11-1963_5",
        "BT_12-11-1963_6",
    }


# ---------------------------------------------------------------- schema upgrade


def test_init_db_adds_identity_columns_to_an_existing_patients_table(tmp_path: Path):
    """The production database predates the identity columns and must survive."""
    from backend import storage

    db_path = tmp_path / "legacy.db"
    conn = sqlite3.connect(db_path)
    conn.execute(
        """
        CREATE TABLE patients (
            id VARCHAR NOT NULL PRIMARY KEY,
            label VARCHAR NOT NULL,
            notes TEXT NOT NULL,
            created_at DATETIME,
            updated_at DATETIME
        )
        """
    )
    conn.execute(
        "INSERT INTO patients (id, label, notes, created_at, updated_at)"
        " VALUES ('uuid-1', '09-05-1954-0', 'keep me', '2026-01-01', '2026-01-01')"
    )
    conn.commit()
    conn.close()

    storage.reset_engine(f"sqlite:///{db_path}")
    storage.init_db()

    conn = sqlite3.connect(db_path)
    columns = {row[1] for row in conn.execute("PRAGMA table_info(patients)")}
    row = conn.execute(
        "SELECT label, notes FROM patients WHERE id = 'uuid-1'"
    ).fetchone()
    tables = {
        r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
    }
    conn.close()

    assert {
        "birthdate",
        "first_name",
        "last_name",
        "first_initial",
        "last_initial",
    } <= columns
    assert row == ("09-05-1954-0", "keep me")
    assert "patient_id_reservations" in tables


def test_init_db_is_idempotent_on_an_already_upgraded_database(tmp_path: Path):
    from backend import storage

    storage.reset_engine(f"sqlite:///{tmp_path / 'app.db'}")
    storage.init_db()
    storage.init_db()

    with storage.session_scope() as session:
        patient = storage.create_patient(session, label="BT_12-11-1963", notes="")

    assert patient.id


# ---------------------------------------------------------------- HTTP API


def _test_app(temp_data_dir, monkeypatch):
    monkeypatch.setenv("QEEG_MOCK_LLM", "1")
    monkeypatch.setenv("QEEG_PORTAL_RAW_SYNC_WATCHER", "0")
    from backend import main

    monkeypatch.setattr(
        main,
        "_ensure_project_clipr_config",
        lambda: Path(temp_data_dir) / "cliproxyapi.conf",
    )
    monkeypatch.setattr(main, "_sync_home_auth_to_project", lambda: 0)
    return main.app, main


def test_create_patient_from_structured_identity_returns_the_canonical_id(
    temp_data_dir, monkeypatch
):
    app, _main = _test_app(temp_data_dir, monkeypatch)

    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.post(
            "/api/patients",
            json={
                "first_name": "Bob",
                "last_name": "Tester",
                "birthdate": "12-11-1963",
                "notes": "referred by Dr. Henderson",
            },
        )

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["patient_id"] == "BT_12-11-1963"
    assert body["label"] == "BT_12-11-1963"
    assert body["first_name"] == "Bob"
    assert body["last_name"] == "Tester"
    assert body["birthdate"] == "12-11-1963"


def test_patient_response_keeps_the_internal_uuid_separate_from_the_clinic_id(
    temp_data_dir, monkeypatch
):
    app, _main = _test_app(temp_data_dir, monkeypatch)

    with TestClient(app, raise_server_exceptions=False) as client:
        created = client.post(
            "/api/patients",
            json={
                "first_name": "Bob",
                "last_name": "Tester",
                "birthdate": "12-11-1963",
            },
        ).json()
        fetched = client.get(f"/api/patients/{created['id']}").json()

    assert created["id"] != created["patient_id"]
    assert len(created["id"]) == 36  # still a UUID
    assert fetched["id"] == created["id"]
    assert fetched["patient_id"] == "BT_12-11-1963"


def test_second_patient_with_the_same_initials_and_dob_gets_a_suffix(
    temp_data_dir, monkeypatch
):
    app, _main = _test_app(temp_data_dir, monkeypatch)

    with TestClient(app, raise_server_exceptions=False) as client:
        first = client.post(
            "/api/patients",
            json={
                "first_name": "Bob",
                "last_name": "Tester",
                "birthdate": "12-11-1963",
            },
        )
        second = client.post(
            "/api/patients",
            json={
                "first_name": "Brenda",
                "last_name": "Tolliver",
                "birthdate": "12-11-1963",
            },
        )

    assert first.json()["patient_id"] == "BT_12-11-1963"
    assert second.status_code == 200, second.text
    assert second.json()["patient_id"] == "BT_12-11-1963_2"


def test_create_patient_rejects_an_impossible_birthdate(temp_data_dir, monkeypatch):
    app, _main = _test_app(temp_data_dir, monkeypatch)

    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.post(
            "/api/patients",
            json={
                "first_name": "Bob",
                "last_name": "Tester",
                "birthdate": "02-30-1984",
            },
        )

    assert response.status_code == 400
    assert "02-30-1984" in response.json()["detail"]


def test_create_patient_asks_for_the_initial_it_cannot_derive(
    temp_data_dir, monkeypatch
):
    app, _main = _test_app(temp_data_dir, monkeypatch)

    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.post(
            "/api/patients",
            json={
                "first_name": "Ørsted",
                "last_name": "Tester",
                "birthdate": "12-11-1963",
            },
        )

    assert response.status_code == 400
    assert "initial" in response.json()["detail"].lower()


def test_create_patient_accepts_an_operator_supplied_initial(
    temp_data_dir, monkeypatch
):
    app, _main = _test_app(temp_data_dir, monkeypatch)

    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.post(
            "/api/patients",
            json={
                "first_name": "Ørsted",
                "last_name": "Tester",
                "birthdate": "12-11-1963",
                "first_initial": "O",
            },
        )

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["patient_id"] == "OT_12-11-1963"
    assert body["first_name"] == "Ørsted"


def test_create_patient_stores_a_unicode_name_under_a_derived_ascii_id(
    temp_data_dir, monkeypatch
):
    app, _main = _test_app(temp_data_dir, monkeypatch)

    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.post(
            "/api/patients",
            json={
                "first_name": "Sofía",
                "last_name": "Peña",
                "birthdate": "03-04-1991",
            },
        )

    body = response.json()
    assert body["patient_id"] == "SP_03-04-1991"
    assert body["first_name"] == "Sofía"
    assert body["last_name"] == "Peña"


def test_reconfirming_the_same_identity_does_not_move_the_patient_to_a_suffix(
    temp_data_dir, monkeypatch
):
    app, _main = _test_app(temp_data_dir, monkeypatch)

    with TestClient(app, raise_server_exceptions=False) as client:
        created = client.post(
            "/api/patients",
            json={
                "first_name": "Bob",
                "last_name": "Tester",
                "birthdate": "12-11-1963",
            },
        ).json()
        updated = client.put(
            f"/api/patients/{created['id']}",
            json={
                "first_name": "Bob",
                "last_name": "Tester",
                "birthdate": "12-11-1963",
                "notes": "confirmed at intake",
            },
        )

    assert updated.status_code == 200, updated.text
    body = updated.json()
    assert body["patient_id"] == "BT_12-11-1963"
    assert body["id"] == created["id"]
    assert body["notes"] == "confirmed at intake"


def test_correcting_a_name_to_a_new_initial_issues_a_new_id_and_retires_the_old(
    temp_data_dir, monkeypatch
):
    """A corrected name is a different identity, so it earns a different ID.

    The ID it vacates stays reserved and must never reach a different person.
    """
    app, _main = _test_app(temp_data_dir, monkeypatch)
    from backend import storage

    with TestClient(app, raise_server_exceptions=False) as client:
        created = client.post(
            "/api/patients",
            json={
                "first_name": "Bob",
                "last_name": "Tester",
                "birthdate": "12-11-1963",
            },
        ).json()
        updated = client.put(
            f"/api/patients/{created['id']}",
            json={
                "first_name": "Robert",
                "last_name": "Tester",
                "birthdate": "12-11-1963",
                "notes": "corrected first name",
            },
        ).json()

    assert updated["patient_id"] == "RT_12-11-1963"
    assert updated["first_name"] == "Robert"
    assert updated["id"] == created["id"]  # the internal UUID never moves

    with storage.session_scope() as session:
        assert (
            allocate_canonical_patient_id(
                session, first_initial="B", last_initial="T", birthdate="12-11-1963"
            )
            == "BT_12-11-1963_2"
        )


def test_update_patient_to_a_taken_identity_allocates_the_next_ordinal(
    temp_data_dir, monkeypatch
):
    app, _main = _test_app(temp_data_dir, monkeypatch)

    with TestClient(app, raise_server_exceptions=False) as client:
        client.post(
            "/api/patients",
            json={
                "first_name": "Bob",
                "last_name": "Tester",
                "birthdate": "12-11-1963",
            },
        )
        other = client.post(
            "/api/patients",
            json={
                "first_name": "Ann",
                "last_name": "Rowe",
                "birthdate": "01-02-1970",
            },
        ).json()
        updated = client.put(
            f"/api/patients/{other['id']}",
            json={
                "first_name": "Bill",
                "last_name": "Turner",
                "birthdate": "12-11-1963",
            },
        )

    assert updated.status_code == 200, updated.text
    assert updated.json()["patient_id"] == "BT_12-11-1963_2"


def test_creating_a_patient_from_a_legacy_label_is_refused(
    temp_data_dir, monkeypatch
):
    """A label that routes nowhere is worse than no patient at all.

    Legacy DOB-keyed patients still in the database predate the cutover and Task
    5 migrates them, but nothing may mint a new one: every portal path now
    rejects that key, so the patient would be created, appear in the roster, and
    silently have no folder, no publishing, no sync, and no batch work.
    """
    app, _main = _test_app(temp_data_dir, monkeypatch)
    from backend import storage

    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.post(
            "/api/patients", json={"label": "09-05-1954-0", "notes": ""}
        )

    assert response.status_code == 400, response.text
    assert "09-05-1954-0" in response.json()["detail"]

    with storage.session_scope() as session:
        assert storage.list_patients(session) == []


def test_creating_a_patient_from_a_free_text_label_is_refused(
    temp_data_dir, monkeypatch
):
    """The only label a patient can be created with is a clinic id."""
    app, _main = _test_app(temp_data_dir, monkeypatch)

    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.post("/api/patients", json={"label": "Barto T", "notes": ""})

    assert response.status_code == 400, response.text


def test_create_patient_requires_identity_or_a_label(temp_data_dir, monkeypatch):
    app, _main = _test_app(temp_data_dir, monkeypatch)

    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.post("/api/patients", json={"notes": "nothing to file"})

    assert response.status_code == 400


def test_a_canonical_label_typed_directly_is_reserved_against_reissue(
    temp_data_dir, monkeypatch
):
    app, _main = _test_app(temp_data_dir, monkeypatch)
    from backend import storage

    with TestClient(app, raise_server_exceptions=False) as client:
        created = client.post(
            "/api/patients", json={"label": "BT_12-11-1963", "notes": ""}
        )
        assert created.json()["patient_id"] == "BT_12-11-1963"

    with storage.session_scope() as session:
        assert (
            session.get(storage.PatientIdReservation, "BT_12-11-1963") is not None
        )
        assert (
            allocate_canonical_patient_id(
                session, first_initial="B", last_initial="T", birthdate="12-11-1963"
            )
            == "BT_12-11-1963_2"
        )


# ---------------------------------------------------------------- neutral preview


def test_report_preview_extracts_text_without_creating_anything(
    temp_data_dir, monkeypatch
):
    app, _main = _test_app(temp_data_dir, monkeypatch)
    from backend import storage

    # Create the folders up front so "still empty" is a real assertion rather
    # than one that passes because nothing ever made them.
    portal_root = Path(temp_data_dir) / "portal_patients"
    portal_root.mkdir(parents=True, exist_ok=True)

    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.post(
            "/api/reports/preview",
            files={
                "file": (
                    "intake.txt",
                    b"Patient: Bob Tester\nDOB: 12/11/1963\n",
                    "text/plain",
                )
            },
        )

    assert response.status_code == 200, response.text
    body = response.json()
    assert "Bob Tester" in body["text"]
    assert "Bob Tester" in body["preview"]
    assert body["filename"] == "intake.txt"

    with storage.session_scope() as session:
        assert storage.list_patients(session) == []
        assert session.scalars(select(storage.Report)).all() == []

    assert portal_root.exists()
    assert list(portal_root.iterdir()) == []


def test_report_preview_leaves_no_report_files_behind(temp_data_dir, monkeypatch):
    app, _main = _test_app(temp_data_dir, monkeypatch)

    reports_root = Path(temp_data_dir) / "reports"
    reports_root.mkdir(parents=True, exist_ok=True)

    with TestClient(app, raise_server_exceptions=False) as client:
        client.post(
            "/api/reports/preview",
            files={"file": ("intake.txt", b"Patient: Bob Tester\n", "text/plain")},
        )

    assert reports_root.exists()
    assert list(reports_root.rglob("*")) == []


def test_report_preview_reuses_the_real_pdf_extraction_path(
    temp_data_dir, monkeypatch, example_pdf_bytes
):
    app, _main = _test_app(temp_data_dir, monkeypatch)

    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.post(
            "/api/reports/preview",
            files={"file": ("report.pdf", example_pdf_bytes, "application/pdf")},
        )

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["mime_type"] == "application/pdf"
    assert len(body["text"]) > 100
    assert body["page_count"] >= 1
