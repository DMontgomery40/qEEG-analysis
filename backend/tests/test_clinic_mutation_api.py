from backend.tests import test_clinic_api as api_fixtures
from backend.tests.clinic_test_helpers import forbid_clinic_paid  # noqa: F401
import pytest


live_api = api_fixtures.live_api


def upload(client, key, principal, **fields):
    return client.post(
        "/uploads",
        headers={
            "Idempotency-Key": key,
            "X-Clinic-Actor": "Staff",
            "X-Clinic-Principal": principal,
        },
        data={
            "firstName": "Ada",
            "lastName": "Baker",
            "birthdate": "02-02-1900",
            **fields,
        },
        files=[
            ("files", ("same.txt", b"one", "text/plain")),
            ("fileMeta", (None, "{}")),
            ("files", ("same.txt", b"two", "text/plain")),
            ("fileMeta", (None, "{}")),
        ],
    )


def test_actual_multipart_two_adapter_principals_share_one_database(live_api):
    client, chart, root = live_api
    a = upload(client, "same-key", "workbench")
    assert a.status_code == 200, a.text
    b = upload(client, "same-key", "thrylen-service")
    assert b.status_code == 200, b.text
    assert a.json()["upload"] == b.json()["upload"]
    u = a.json()["upload"]
    assert client.get("/uploads/" + u["uploadId"]).json()["upload"] == u
    assert len(client.get("/uploads").json()["uploads"]) == 1
    result = client.post(
        "/feedback",
        headers={"Idempotency-Key": "feedback"},
        json={
            "patientId": u["patientId"],
            "fileId": u["items"][0]["fileId"],
            "version": 1,
            "action": "approve",
        },
    )
    assert result.status_code == 200, result.text
    assert (
        next(
            f
            for f in client.get(
                "/patient-files", params={"patientId": u["patientId"]}
            ).json()["files"]
            if f["fileId"] == u["items"][0]["fileId"]
        )["feedback"]["action"]
        == "approve"
    )
    patch = client.patch(
        "/patients/" + u["patientId"],
        headers={"Idempotency-Key": "notes"},
        json={"notes": "Likes music"},
    )
    assert patch.status_code == 200, patch.text
    assert patch.json()["patient"]["notes"] == "Likes music"
    assert (
        client.get("/jobs", params={"patientId": u["patientId"]}).json()["jobs"] == []
    )


@pytest.mark.parametrize(
    "extra",
    [
        {"unknown": "x"},
        {"submissionId": "different"},
        {"resolution": '{"forceNew":"true"}'},
        {"analysisIntent": '{"confirmed":true}'},
        {"resolution": '{"attachTo":"AB_02-02-1900","forceNew":true}'},
    ],
)
def test_malformed_multipart_rejected_before_patient_allocation(live_api, extra):
    client, chart, root = live_api
    response = upload(client, "bad", "workbench", **extra)
    assert response.status_code == 400, response.text
    assert len(client.get("/patients").json()["patients"]) == 1


def test_admission_preserves_first_trusted_principal_without_breaking_cross_adapter_replay(
    live_api,
):
    from backend import storage
    from backend.clinic_records import ClinicUpload
    from sqlalchemy import select

    client, _, _ = live_api
    assert upload(client, "attribution", "workbench").status_code == 200
    assert upload(client, "attribution", "thrylen-service").status_code == 200
    with storage.session_scope() as s:
        u = s.scalar(select(ClinicUpload).where(ClinicUpload.id == "attribution"))
        assert u.uploaded_principal == "workbench"
        assert u.uploaded_by == "Staff"


def test_http_analysis_confirmation_binds_shown_policy_before_filing(
    live_api, monkeypatch
):
    import hashlib
    import json
    from backend import clinic_analysis_intents

    client, _, root = live_api
    shown = client.get("/policy").json()["analysisPolicyFingerprint"]
    intent = dict(
        operationId="http-policy",
        confirmed=True,
        reportItemIndexes=[0],
        specialInstructions="",
        expectedPolicyFingerprint=shown,
    )
    fields = dict(
        firstName="Ada",
        lastName="Baker",
        birthdate="02-02-1900",
        analysisIntent=json.dumps(intent),
    )
    files = [
        ("files", ("source.txt", b"original report", "text/plain")),
        ("fileMeta", (None, '{"documentKind":"report"}')),
    ]

    def post(key):
        return client.post(
            "/uploads",
            headers={
                "Idempotency-Key": key,
                "X-Clinic-Actor": "Staff",
                "X-Clinic-Principal": "thrylen-service",
            },
            data=fields,
            files=files,
        )

    first = post("confirmed-http")
    assert first.status_code == 200, first.text
    upload_id = first.json()["upload"]["uploadId"]
    assert (
        clinic_analysis_intents.confirmed_analysis_binding(upload_id)["policyHash"]
        == shown
    )
    monkeypatch.setenv("QEEG_STAGE1_MAX_TOKENS", "87654")
    assert post("confirmed-http").json()["upload"]["uploadId"] == upload_id
    response = post("stale-http")
    assert response.status_code == 409 and "policy changed" in response.text
    assert not (
        root
        / "clinic_intake"
        / "submissions"
        / hashlib.sha256(b"stale-http").hexdigest()
    ).exists()
