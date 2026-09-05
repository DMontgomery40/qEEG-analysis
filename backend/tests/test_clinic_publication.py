"""Database-issued publication bindings and actual local helper byte transport."""

import hashlib
import sys
import pytest
from backend import storage, clinic_catalogue as catalogue
from backend.tests.clinic_test_helpers import forbid_clinic_paid  # noqa: F401


def seed(root):
    with storage.session_scope() as s:
        patient = storage.create_patient(s, label="ZZ_01-01-1900")
    path = root / "one.bin"
    path.write_bytes(b"original")
    artifact = catalogue.register_artifact(
        patient_uuid=patient.id,
        source_kind="renderer",
        source_id="op:one",
        logical_family="video",
        original_name="one.bin",
        local_path=path,
    )
    return patient, artifact


def test_prepare_and_replay_relabel_and_readback(temp_data_dir, monkeypatch):
    from backend import clinic_publication as publisher

    patient, artifact = seed(temp_data_dir)
    initial = publisher.prepare_publication(artifact["fileId"])
    key = initial["item"]["remoteKey"]
    revision = initial["catalogRevision"]
    assert (
        publisher.prepare_publication(artifact["fileId"])["catalogRevision"] == revision
    )
    with storage.session_scope() as s:
        storage.update_patient(s, patient.id, label="AZ_01-01-1900")
    assert publisher.prepare_publication(artifact["fileId"])["item"]["remoteKey"] == key
    monkeypatch.setattr(
        publisher, "strong_readback", lambda key, size: iter([b"orig", b"inal"])
    )
    result = publisher.verify_publication(artifact["fileId"], key)
    assert result["item"]["verified"]
    monkeypatch.setattr(
        publisher, "strong_readback", lambda key, size: iter([b"changed!"])
    )
    with pytest.raises(catalogue.CatalogueConflict):
        publisher.verify_publication(artifact["fileId"], key)
    assert not publisher.prepare_publication(artifact["fileId"])["item"]["verified"]


@pytest.mark.parametrize(
    "script,outcome",
    [
        ("import sys;sys.stdout.buffer.write(b'original')", "ok"),
        ("import sys;sys.stdout.buffer.write(b'original');sys.exit(7)", "error"),
        ("import sys;sys.stdout.buffer.write(b'originalx')", "error"),
        ("import time;time.sleep(30)", "timeout"),
    ],
)
def test_actual_helper_completion_caps_and_timeout(temp_data_dir, script, outcome):
    from backend.clinic_publication import _helper_bytes

    path = temp_data_dir / "helper.py"
    path.write_text(script)
    chunks = _helper_bytes(
        [sys.executable, str(path)],
        cwd=temp_data_dir,
        key="patients/ZZ_01-01-1900/files/f",
        size=8,
        timeout=0.2,
    )
    if outcome == "ok":
        assert b"".join(chunks) == b"original"
    else:
        with pytest.raises(catalogue.CatalogueUnavailable):
            b"".join(chunks)


def test_publication_pages_bind_revision_and_do_not_write(temp_data_dir):
    from backend import clinic_publication as publisher

    patient, artifact = seed(temp_data_dir)
    first = publisher.publication_items(patient.label, limit=1)
    assert first["items"][0]["fileId"] == artifact["fileId"]
    assert first["items"][0]["remoteKey"] is None
    assert first["items"][0]["sha256"] == hashlib.sha256(b"original").hexdigest()
    assert publisher.publication_items(patient.label, limit=1) == first


def test_publication_target_survives_later_import_of_same_filename(temp_data_dir):
    from backend import clinic_publication as publisher

    patient, artifact = seed(temp_data_dir)
    original = publisher.prepare_publication(artifact["fileId"])["item"]["remoteKey"]
    catalogue.register_patient_alias(patient.id, "AA_01-01-1900")
    catalogue.add_remote_location(
        artifact["fileId"], "patients/AA_01-01-1900/files/" + artifact["fileKey"]
    )
    assert (
        publisher.prepare_publication(artifact["fileId"])["item"]["remoteKey"]
        == original
    )


def test_publication_internal_boundary_and_exact_source_binding(live_api, monkeypatch):
    client, chart, root = live_api
    path = root / "portal_patients" / chart.label / "out.mp4"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"output")
    monkeypatch.setenv("QEEG_PORTAL_PATIENTS_DIR", str(root / "portal_patients"))
    data = {
        "operationId": "op",
        "patientId": chart.label,
        "producer": "renderer",
        "kind": "video",
        "original": {"receiptId": "old"},
    }
    for headers in (
        {"Origin": "http://localhost"},
        {"X-Clinic-Principal": "workbench"},
    ):
        assert (
            client.post("/internal/operations", json=data, headers=headers).status_code
            == 403
        )
    assert (
        client.post(
            "/internal/operations", content="{}", headers={"Content-Type": "text/plain"}
        ).status_code
        == 415
    )
    assert client.post("/internal/operations", json=data).status_code == 200
    output = {
        "patientId": chart.label,
        "operationId": "op",
        "outputId": "mp4",
        "relativePath": "out.mp4",
        "originalName": "out.mp4",
        "logicalFamily": "video",
        "documentKind": "video",
    }
    first = client.post("/internal/artifacts", json=output)
    assert first.status_code == 200, first.text
    assert (
        client.post("/internal/artifacts", json=output).json()["artifact"]
        == first.json()["artifact"]
    )
    path.write_bytes(b"changed")
    assert client.post("/internal/artifacts", json=output).status_code == 409
    outside = root / "outside.mp4"
    outside.write_bytes(b"outside")
    (path.parent / "escape.mp4").symlink_to(outside)
    assert (
        client.post(
            "/internal/artifacts",
            json={**output, "outputId": "escape", "relativePath": "escape.mp4"},
        ).status_code
        == 400
    )


from backend.tests.test_clinic_api import live_api  # noqa: E402,F401


def test_notification_claim_is_single_use_even_when_ack_is_lost(live_api):
    client, chart, root = live_api
    p = root / "feedback.bin"
    p.write_bytes(b"feedback")
    a = catalogue.register_artifact(
        patient_uuid=chart.id,
        source_kind="manual",
        source_id="notify",
        original_name=p.name,
        logical_family="f",
        local_path=p,
    )
    headers = {
        "X-Clinic-Principal": "thrylen-service",
        "X-Clinic-Actor": "Staff",
        "Idempotency-Key": "event",
    }
    body = {
        "patientId": chart.label,
        "fileId": a["fileId"],
        "version": 1,
        "action": "approve",
    }
    assert client.post("/feedback", json=body, headers=headers).status_code == 200
    headers = {**headers, "Idempotency-Key": "one"}
    route = "/feedback/event/notification"
    assert client.post(route + "/claim", json={"claimId": "one"}).status_code == 403
    first = client.post(route + "/claim", json={"claimId": "one"}, headers=headers)
    assert first.status_code == 200, first.text
    assert (
        first.json()["acquired"] is True
        and first.json()["notification"]["status"] == "unknown"
    )
    revision = first.json()["catalogRevision"]
    for claim in ("one", "other"):
        repeated = client.post(
            route + "/claim",
            json={"claimId": claim},
            headers={**headers, "Idempotency-Key": claim},
        ).json()
        assert repeated["acquired"] is False and repeated["catalogRevision"] == revision
    assert (
        client.post(
            route,
            json={"claimId": "other", "status": "sent"},
            headers={**headers, "Idempotency-Key": "other"},
        ).status_code
        == 409
    )
    ack = {"claimId": "one", "status": "sent", "detail": "original sent"}
    first_ack = client.post(route, json=ack, headers=headers)
    assert first_ack.status_code == 200, first_ack.text
    assert client.post(route, json=ack, headers=headers).json() == first_ack.json()
    assert (
        client.post(
            route, json={**ack, "status": "failed"}, headers=headers
        ).status_code
        == 409
    )


def test_prepared_key_cannot_be_stolen_by_later_import(temp_data_dir):
    from backend import clinic_publication as publisher

    patient, artifact = seed(temp_data_dir)
    key = publisher.prepare_publication(artifact["fileId"])["item"]["remoteKey"]
    path = temp_data_dir / "other.bin"
    path.write_bytes(b"other")
    other = catalogue.register_artifact(
        patient_uuid=patient.id,
        source_kind="renderer",
        source_id="other",
        logical_family="video",
        original_name=path.name,
        local_path=path,
    )
    with pytest.raises(catalogue.CatalogueConflict):
        catalogue.add_remote_location(other["fileId"], key)


def test_concurrent_prepare_import_relabel_keeps_one_target(temp_data_dir):
    from concurrent.futures import ThreadPoolExecutor
    from backend import clinic_publication as publisher

    patient, artifact = seed(temp_data_dir)

    def prepare(_):
        return publisher.prepare_publication(artifact["fileId"])["item"]["remoteKey"]

    def import_copy(_):
        catalogue.add_remote_location(
            artifact["fileId"], f"patients/{patient.label}/files/historic.bin"
        )

    def relabel(_):
        with storage.session_scope() as session:
            storage.update_patient(session, patient.id, label="AZ_01-01-1900")

    with ThreadPoolExecutor(max_workers=6) as pool:
        tasks = [pool.submit(prepare, i) for i in range(8)] + [
            pool.submit(import_copy, 0),
            pool.submit(relabel, 0),
        ]
        results = [t.result() for t in tasks]
    assert len(set(results[:8])) == 1
    assert prepare(0) == results[0]


@pytest.mark.parametrize("boundary", ["prepare", "upload", "verify"])
def test_real_process_death_replacement_keeps_original_publication(
    temp_data_dir, monkeypatch, boundary
):
    import json
    import os
    import subprocess
    from pathlib import Path
    from sqlalchemy import select, func
    from backend import clinic_publication as publisher
    from backend.clinic_models import ClinicArtifact, ClinicPublication

    _, artifact = seed(temp_data_dir)
    code = """
import json, os, signal, sys
from pathlib import Path
from backend import storage, clinic_publication as publisher
from backend.paid_transport import PaidSyncTransport, PaidAsyncTransport
def forbidden(*a, **k): raise AssertionError('Paid transport forbidden')
PaidSyncTransport.handle_request = forbidden
PaidAsyncTransport.handle_async_request = forbidden
storage.init_db()
file_id, boundary = json.load(sys.stdin)
item = publisher.prepare_publication(file_id)['item']
if boundary != 'prepare':
    remote = Path(storage.DATA_DIR)/'synthetic-remote'
    with remote.open('wb') as output:
        output.write(b'original');output.flush();os.fsync(output.fileno())
if boundary == 'verify':
    publisher.strong_readback = lambda key, size: iter([remote.read_bytes()])
    publisher.verify_publication(file_id,item['remoteKey'])
os.kill(os.getpid(),signal.SIGKILL)
"""
    child = subprocess.run(
        [sys.executable, "-c", code],
        input=json.dumps([artifact["fileId"], boundary]),
        text=True,
        capture_output=True,
        timeout=20,
        env={
            **os.environ,
            "DATA_DIR": str(temp_data_dir),
            "QEEG_ANALYSIS_ROOT": str(temp_data_dir.parent),
        },
    )
    assert child.returncode == -9, child.stderr
    first = publisher.prepare_publication(artifact["fileId"])["item"]
    assert first["verified"] == (boundary == "verify")
    remote = Path(temp_data_dir) / "synthetic-remote"
    if not remote.exists():
        remote.write_bytes(b"original")
    monkeypatch.setattr(
        publisher, "strong_readback", lambda key, size: iter([remote.read_bytes()])
    )
    replacement = publisher.verify_publication(artifact["fileId"], first["remoteKey"])[
        "item"
    ]
    assert replacement["verified"] and replacement["remoteKey"] == first["remoteKey"]
    with storage.session_scope() as session:
        assert session.scalar(select(func.count()).select_from(ClinicArtifact)) == 1
        assert session.scalar(select(func.count()).select_from(ClinicPublication)) == 1


def test_helper_cancellation_after_pipe_eof_drains_process(temp_data_dir):
    import threading
    import time
    from backend.clinic_publication import _helper_bytes

    script = temp_data_dir / "closed-pipes.py"
    script.write_text("import os,time\nos.close(1);os.close(2);time.sleep(5)\n")
    stop = threading.Event()
    timer = threading.Timer(0.1, stop.set)
    start = time.monotonic()
    timer.start()
    try:
        with pytest.raises(catalogue.CatalogueUnavailable):
            list(
                _helper_bytes(
                    [sys.executable, str(script)],
                    cwd=temp_data_dir,
                    key="key",
                    size=8,
                    timeout=1,
                    stop_event=stop,
                )
            )
        assert time.monotonic() - start < 0.75
    finally:
        timer.cancel()


@pytest.mark.parametrize(
    "replacement", [b"replaced", b"longer replacement", b"original"]
)
def test_expected_producer_bytes_guard_first_snapshot_and_restore(
    live_api, monkeypatch, replacement
):
    client, chart, root = live_api
    path = root / "portal_patients" / chart.label / "bound.mp4"
    path.parent.mkdir(parents=True)
    path.write_bytes(replacement)
    monkeypatch.setenv("QEEG_PORTAL_PATIENTS_DIR", str(root / "portal_patients"))
    assert (
        client.post(
            "/internal/operations",
            json={
                "operationId": "expected-op",
                "patientId": chart.label,
                "producer": "renderer",
                "kind": "video",
                "original": {"receiptId": "original"},
            },
        ).status_code
        == 200
    )
    output = {
        "patientId": chart.label,
        "operationId": "expected-op",
        "outputId": "mp4",
        "relativePath": path.name,
        "originalName": path.name,
        "logicalFamily": "video",
        "expectedSha256": hashlib.sha256(b"original").hexdigest(),
        "expectedSize": 8,
    }
    first = client.post("/internal/artifacts", json=output)
    if replacement != b"original":
        assert first.status_code == 409, first.text
        assert not list((root / "clinic_producer_bytes").rglob("original"))
        with storage.session_scope() as session:
            from backend.clinic_models import ClinicArtifact

            assert (
                not session.query(ClinicArtifact)
                .filter_by(source_kind="renderer")
                .all()
            )
    else:
        assert first.status_code == 200, first.text
    path.write_bytes(b"original")
    accepted = client.post("/internal/artifacts", json=output)
    assert accepted.status_code == 200, accepted.text
    repeated = client.post("/internal/artifacts", json=output)
    assert repeated.json()["artifact"] == accepted.json()["artifact"]
    assert accepted.json()["artifact"]["version"] == 1
    assert accepted.json()["artifact"]["sha256"] == output["expectedSha256"]


@pytest.mark.parametrize(
    "fields",
    [
        {"expectedSha256": "a" * 64},
        {"expectedSize": 8},
        {"expectedSha256": None, "expectedSize": None},
        {"expectedSha256": "bad", "expectedSize": 8},
        {"expectedSha256": "a" * 64, "expectedSize": True},
        {"expectedSha256": "a" * 64, "expectedSize": -1},
    ],
)
def test_expected_producer_pair_rejects_invalid_material_without_snapshot(
    live_api, fields
):
    client, chart, root = live_api
    response = client.post(
        "/internal/artifacts",
        json={
            "patientId": chart.label,
            "operationId": "invalid-op",
            "outputId": "mp4",
            "relativePath": "unused.mp4",
            "originalName": "unused.mp4",
            "logicalFamily": "video",
            **fields,
        },
    )
    assert response.status_code == 400, response.text
    assert not (root / "clinic_producer_bytes").exists()
