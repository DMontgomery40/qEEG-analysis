"""Read-only identity preview through actual HTTP and original SQLite authority."""

import pytest
from sqlalchemy import event
from backend import storage
from backend.tests.test_clinic_api import live_api  # noqa: F401
from backend.tests.clinic_test_helpers import forbid_clinic_paid  # noqa: F401


def test_selected_chart_compares_every_supplied_field_without_writes(live_api):
    client, chart, _ = live_api
    statements = []

    def capture(conn, cursor, statement, parameters, context, executemany):
        statements.append(statement)

    before = client.get("/patients").json()
    event.listen(storage.engine, "before_cursor_execute", capture)
    try:
        response = client.post(
            "/identity-preview",
            json={
                "records": [
                    {
                        "patientId": chart.label,
                        "firstName": " Zoë ",
                        "lastName": "Zéphyr",
                        "birthdate": "2-3-1901",
                    },
                    {
                        "patientId": chart.label,
                        "firstInitial": "z",
                        "lastInitial": "z",
                        "birthdate": "1-1-1900",
                    },
                    {"patientId": chart.label, "firstInitial": "a"},
                    {"patientId": chart.label, "birthdate": "02-30-1900"},
                ]
            },
        )
    finally:
        event.remove(storage.engine, "before_cursor_execute", capture)
    assert response.status_code == 200, response.text
    rows = response.json()["rows"]
    assert [r["status"] for r in rows] == ["change", "unchanged", "change", "invalid"]
    assert rows[0]["proposed"] == {
        "firstName": "Zoë",
        "lastName": "Zéphyr",
        "birthdate": "02-03-1901",
    }
    assert rows[0]["patientId"] == chart.label
    assert [r["row"] for r in rows] == [1, 2, 3, 4]
    assert all("patientId" not in r["proposed"] for r in rows)
    assert not any(
        s.lstrip()
        .upper()
        .startswith(("INSERT", "UPDATE", "DELETE", "REPLACE", "CREATE"))
        for s in statements
    )
    assert client.get("/patients").json() == before


def test_identity_matching_missing_names_conflicts_and_multiple_matches(live_api):
    client, chart, _ = live_api
    with storage.session_scope() as s:
        storage.create_patient(
            s,
            label="AB_02-02-1900",
            first_name="Ada",
            last_name="Baker",
            first_initial="A",
            last_initial="B",
            birthdate="02-02-1900",
        )
        storage.create_patient(
            s,
            label="AB_02-02-1900_2",
            first_name="Anne",
            last_name="Baker",
            first_initial="A",
            last_initial="B",
            birthdate="02-02-1900",
        )
        storage.create_patient(
            s,
            label="CD_03-03-1900",
            first_initial="C",
            last_initial="D",
            birthdate="03-03-1900",
        )
    records = [
        {"firstName": "Ada", "lastName": "Baker", "birthdate": "2-2-1900"},
        {"firstName": "Alice", "lastName": "Baker", "birthdate": "2-2-1900"},
        {"firstName": "Carl", "lastName": "Dane", "birthdate": "3-3-1900"},
        {"firstName": "Else", "lastName": "Free", "birthdate": "4-4-1900"},
    ]
    rows = client.post("/identity-preview", json={"records": records}).json()["rows"]
    assert [r["status"] for r in rows] == [
        "unchanged",
        "needs_operator_answer",
        "change",
        "not_found",
    ]
    assert {p["patientId"] for p in rows[1]["candidates"]} == {
        "AB_02-02-1900",
        "AB_02-02-1900_2",
    }
    with storage.session_scope() as s:
        storage.create_patient(
            s,
            label="CD_03-03-1900_2",
            first_initial="C",
            last_initial="D",
            birthdate="03-03-1900",
        )
    row = client.post("/identity-preview", json={"records": [records[2]]}).json()[
        "rows"
    ][0]
    assert row["status"] == "needs_operator_answer"
    assert len(row["candidates"]) == 2


@pytest.mark.parametrize(
    "body",
    [
        {},
        {"records": []},
        {"records": [{}], "extra": True},
        {"records": [{"unknown": "x"}]},
        {"records": [{}] * 101},
    ],
)
def test_bad_preview_shape_is_not_silently_accepted(live_api, body):
    client, _, _ = live_api
    assert client.post("/identity-preview", json=body).status_code == 400


@pytest.mark.parametrize("relabels", [0, 1, 2])
def test_historical_collision_never_selects_first_chart(live_api, relabels):
    client, _, _ = live_api
    with storage.session_scope() as s:
        a = storage.create_patient(s, label="AB_01-01-1900")
        b = storage.create_patient(s, label="AB_01-01-1900")
        if relabels:
            storage.update_patient(s, a.id, label="AC_01-01-1900")
        if relabels == 2:
            storage.update_patient(s, b.id, label="AD_01-01-1900")
    row = client.post(
        "/identity-preview",
        json={"records": [{"patientId": "AB_01-01-1900", "firstName": "Amy"}]},
    ).json()["rows"][0]
    assert row["status"] == "needs_operator_answer"
    assert row["patientId"] is None and row["current"] is None


@pytest.mark.parametrize(
    "record",
    [
        {"patientId": None},
        {"firstInitial": "AB"},
        {"birthdate": "not-date"},
        {"firstName": 123},
        {"firstName": "李"},
        {"firstName": ""},
    ],
)
def test_invalid_values_are_ordered_row_errors(live_api, record):
    client, chart, _ = live_api
    response = client.post(
        "/identity-preview", json={"records": [record, {"patientId": chart.label}]}
    )
    assert response.status_code == 200
    assert [r["status"] for r in response.json()["rows"]] == ["invalid", "unchanged"]


def test_preview_selected_alias_and_normalized_material_applies_exactly(live_api):
    client, chart, _ = live_api
    from backend.clinic_catalogue import register_patient_alias

    register_patient_alias(chart.id, "historical-chart")
    record = {
        "patientId": "historical-chart",
        "firstName": "李",
        "firstInitial": "l",
        "lastName": " Éclair ",
        "lastInitial": "e",
        "birthdate": "2-3-1900",
    }
    response = client.post("/identity-preview", json={"records": [record]})
    row = response.json()["rows"][0]
    assert row["status"] == "change" and row["patientId"] == chart.label
    applied = client.patch(
        "/patients/" + row["patientId"],
        json=row["proposed"],
        headers={"Idempotency-Key": "apply-preview"},
    )
    assert applied.status_code == 200, applied.text
    current = applied.json()["patient"]
    assert current["identity"] == {
        "firstName": "李",
        "lastName": "Éclair",
        "firstInitial": "L",
        "lastInitial": "E",
    }
    assert current["birthdate"] == "02-03-1900"
