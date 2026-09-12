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


def test_publication_census_skips_remote_only_history(temp_data_dir):
    """Sep 5 incident: the catalogue import registered ~182k remote-only
    history rows (hub blobs with no local bytes). The publisher listed every
    one, then failed prepare -> snapshot -> verify for each and looped forever,
    holding the engine at 25-35% CPU. Only artifacts with an active local
    location can be published, so only those may appear in the census."""
    from backend import clinic_publication as publisher

    patient, local = seed(temp_data_dir)
    remote_key = "patients/ZZ_01-01-1900/files/old-history.bin"
    remote_only = catalogue.register_artifact(
        patient_uuid=patient.id,
        source_kind="netlify-history",
        source_id=remote_key,
        logical_family="video",
        original_name="old-history.bin",
        sha256=hashlib.sha256(b"remote bytes").hexdigest(),
        size=len(b"remote bytes"),
        provenance=dict(originalRemoteKey=remote_key),
    )
    catalogue.add_remote_location(remote_only["fileId"], remote_key)

    page = publisher.publication_items(patient.label, limit=10)
    assert [item["fileId"] for item in page["items"]] == [local["fileId"]]
    assert page["nextCursor"] is None
    # A remote-only row is still a real catalogue entry; it is just not
    # publication work.
    with storage.session_scope() as s:
        from backend.clinic_models import ClinicArtifact

        assert s.get(ClinicArtifact, remote_only["fileId"]) is not None


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


@pytest.mark.parametrize("admitted_before_patch", [False, True])
@pytest.mark.parametrize("current_folder", ["absent", "empty", "different"])
def test_original_producer_alias_survives_patch_and_exact_replay(
    live_api, monkeypatch, admitted_before_patch, current_folder
):
    client, chart, root = live_api
    portal = root / "portal_patients"
    monkeypatch.setenv("QEEG_PORTAL_PATIENTS_DIR", str(portal))
    original_path = portal / chart.label / "original.mp4"
    original_path.parent.mkdir(parents=True)
    original_path.write_bytes(b"original")
    operation = {
        "operationId": "relabel-original",
        "patientId": chart.label,
        "producer": "renderer",
        "kind": "video",
        "original": {"receiptId": "retained-original"},
    }
    assert client.post("/internal/operations", json=operation).status_code == 200
    material = {
        "patientId": chart.label,
        "operationId": operation["operationId"],
        "outputId": "final-video",
        "relativePath": original_path.name,
        "originalName": original_path.name,
        "logicalFamily": "video",
        "expectedSha256": hashlib.sha256(b"original").hexdigest(),
        "expectedSize": 8,
    }
    accepted = None
    publication_key = None
    if admitted_before_patch:
        response = client.post("/internal/artifacts", json=material)
        assert response.status_code == 200, response.text
        accepted = response.json()["artifact"]
        publication_key = client.post(
            f"/internal/publication/{accepted['fileId']}/prepare", json={}
        ).json()["item"]["remoteKey"]
    patched = client.patch(
        f"/patients/{chart.label}",
        headers={"Idempotency-Key": "original-relabel"},
        json={"firstInitial": "A"},
    )
    assert patched.status_code == 200, patched.text
    current = patched.json()["patient"]["patientId"]
    if current_folder != "absent":
        (portal / current).mkdir()
    if current_folder == "different":
        (portal / current / original_path.name).write_bytes(b"different")
    response = client.post("/internal/artifacts", json=material)
    assert response.status_code == 200, response.text
    artifact = response.json()["artifact"]
    if accepted:
        assert (artifact["fileId"], artifact["fileKey"], artifact["version"]) == (
            accepted["fileId"],
            accepted["fileKey"],
            accepted["version"],
        )
        assert (
            client.post(
                f"/internal/publication/{artifact['fileId']}/prepare", json={}
            ).json()["item"]["remoteKey"]
            == publication_key
        )
    assert artifact["patientId"] == current
    assert artifact["sha256"] == material["expectedSha256"]
    assert (
        client.post("/internal/artifacts", json=material).json()["artifact"] == artifact
    )
    for label in (chart.label, current):
        download = client.get(
            "/file", params={"patientId": label, "fileId": artifact["fileId"]}
        )
        assert download.status_code == 200 and download.content == b"original"
    original_path.write_bytes(b"replaced")
    assert client.post("/internal/artifacts", json=material).status_code == 409
    original_path.write_bytes(b"original")
    assert client.post("/internal/artifacts", json=material).status_code == 200
    # A new operation targets the current chart directory independently.
    new_path = portal / current / "new.mp4"
    new_path.parent.mkdir(exist_ok=True)
    new_path.write_bytes(b"new")
    assert (
        client.post(
            "/internal/operations",
            json={**operation, "operationId": "new-output", "patientId": current},
        ).status_code
        == 200
    )
    new_material = {
        **material,
        "patientId": current,
        "operationId": "new-output",
        "relativePath": new_path.name,
        "originalName": new_path.name,
        "expectedSha256": hashlib.sha256(b"new").hexdigest(),
        "expectedSize": 3,
    }
    assert client.post("/internal/artifacts", json=new_material).status_code == 200


@pytest.mark.parametrize("alias_kind", ["dot", "parent", "outside", "other-chart"])
def test_original_producer_directory_keeps_owned_containment(
    live_api, monkeypatch, alias_kind
):
    client, chart, root = live_api
    portal = root / "portal_patients"
    portal.mkdir()
    monkeypatch.setenv("QEEG_PORTAL_PATIENTS_DIR", str(portal))
    operation = {
        "operationId": "directory-owner",
        "patientId": chart.label,
        "producer": "renderer",
        "kind": "video",
        "original": {"receiptId": "original"},
    }
    assert client.post("/internal/operations", json=operation).status_code == 200
    if alias_kind == "other-chart":
        alias = "BB_01-01-1900"
        with storage.session_scope() as session:
            storage.create_patient(session, label=alias)
        directory = portal / alias
        directory.mkdir()
    else:
        alias = {"dot": ".", "parent": "..", "outside": "old-chart"}[alias_kind]
        catalogue.register_patient_alias(chart.id, alias)
        if alias_kind == "outside":
            directory = root / "outside"
            directory.mkdir()
            (portal / alias).symlink_to(directory, target_is_directory=True)
        else:
            directory = (portal / alias).resolve()
    (directory / "unowned.mp4").write_bytes(b"original")
    response = client.post(
        "/internal/artifacts",
        json={
            "patientId": alias,
            "operationId": operation["operationId"],
            "outputId": "mp4",
            "relativePath": "unowned.mp4",
            "originalName": "unowned.mp4",
            "logicalFamily": "video",
            "expectedSha256": hashlib.sha256(b"original").hexdigest(),
            "expectedSize": 8,
        },
    )
    assert response.status_code == (409 if alias_kind == "other-chart" else 400)
    assert not list((root / "clinic_producer_bytes").rglob("original"))


@pytest.mark.parametrize("already_verified", [False, True])
def test_equal_patient_content_reuses_target_and_preserves_links(
    temp_data_dir, monkeypatch, already_verified
):
    from backend import clinic_publication as publisher
    from backend.clinic_catalogue_reads import file_binding

    patient, first = seed(temp_data_dir)
    key = publisher.prepare_publication(first["fileId"])["item"]["remoteKey"]
    monkeypatch.setattr(
        publisher, "strong_readback", lambda key, size: iter([b"original"])
    )
    if already_verified:
        publisher.verify_publication(first["fileId"], key)
    second = catalogue.register_artifact(
        patient_uuid=patient.id,
        source_kind="renderer",
        source_id="op:two",
        logical_family="video",
        original_name="second.bin",
        local_path=temp_data_dir / "one.bin",
    )
    prepared = publisher.prepare_publication(second["fileId"])["item"]
    assert prepared["remoteKey"] == key
    assert prepared["verified"] == already_verified
    publisher.verify_publication(second["fileId"], key)
    assert publisher.prepare_publication(second["fileId"])["item"]["verified"]
    for artifact in (first, second):
        bound = file_binding(patient.label, file_key=artifact["fileKey"])
        assert bound["fileId"] == artifact["fileId"]
        assert any(l["key"] == key and l["verified"] for l in bound["locations"])
    # Same bytes in another patient's chart are never used as its storage target.
    with storage.session_scope() as s:
        other = storage.create_patient(s, label="AB_01-01-1900")
    third = catalogue.register_artifact(
        patient_uuid=other.id,
        source_kind="renderer",
        source_id="op:three",
        logical_family="video",
        original_name="third.bin",
        local_path=temp_data_dir / "one.bin",
    )
    assert publisher.prepare_publication(third["fileId"])["item"]["remoteKey"] != key
    changed = temp_data_dir / "changed.bin"
    changed.write_bytes(b"changed!")
    fourth = catalogue.register_artifact(
        patient_uuid=patient.id,
        source_kind="renderer",
        source_id="op:four",
        logical_family="video",
        original_name="changed.bin",
        local_path=changed,
    )
    assert publisher.prepare_publication(fourth["fileId"])["item"]["remoteKey"] != key


def test_content_cleanup_keeps_old_keys_and_local_paths_and_replays(
    temp_data_dir, monkeypatch
):
    from backend import clinic_publication as publisher
    from backend.clinic_dedup import consolidate_remote, consolidate_local
    from backend.clinic_catalogue_reads import file_binding, open_local_file

    patient, first = seed(temp_data_dir)
    path = temp_data_dir / "duplicate.bin"
    path.write_bytes(b"original")
    second = catalogue.register_artifact(
        patient_uuid=patient.id,
        source_kind="renderer",
        source_id="cleanup:second",
        logical_family="video",
        original_name="duplicate.bin",
        local_path=path,
    )
    first_key = "patients/" + patient.label + "/files/" + first["fileKey"]
    old_key = "patients/" + patient.label + "/files/" + second["fileKey"]
    orphan_key = "patients/" + patient.label + "/files/old-unindexed-copy.bin"
    catalogue.add_remote_location(first["fileId"], first_key)
    catalogue.add_remote_location(second["fileId"], old_key)
    digest = hashlib.sha256(b"original").hexdigest()
    for _ in range(2):
        consolidate_remote(
            patient.label,
            first_key,
            [old_key, orphan_key],
            digest,
            8,
            lambda: [b"original"],
        )
        for key in (first["fileKey"], second["fileKey"], "old-unindexed-copy.bin"):
            bound = file_binding(patient.label, file_key=key)
            assert any(
                l["key"] == first_key and l["active"] and l["verified"]
                for l in bound["locations"]
            )
        assert (
            publisher.prepare_publication(second["fileId"])["item"]["remoteKey"]
            == first_key
        )
    with pytest.raises(catalogue.CatalogueConflict):
        consolidate_remote(
            patient.label, first_key, [old_key], digest, 8, lambda: [b"changed!"]
        )
    result = consolidate_local(patient.label, producers_quiescent=True)
    assert result == {"removedBodies": 1, "savedBytes": 8}
    assert path.is_symlink() and path.read_bytes() == b"original"
    assert (temp_data_dir / "one.bin").read_bytes() == b"original"
    assert (
        consolidate_local(patient.label, producers_quiescent=True)["removedBodies"] == 0
    )
    for artifact in (first, second):
        snapshot = open_local_file(artifact["fileId"])
        try:
            assert snapshot.read() == b"original"
        finally:
            snapshot.close()


def test_local_cleanup_restart_after_receipt_failure_and_changed_file(temp_data_dir):
    from backend.clinic_dedup import consolidate_local

    patient, first = seed(temp_data_dir)
    for i in range(3):
        path = temp_data_dir / f"copy-{i}.bin"
        path.write_bytes(b"original")
        catalogue.register_artifact(
            patient_uuid=patient.id,
            source_kind="renderer",
            source_id=f"copy:{i}",
            logical_family="video",
            original_name=path.name,
            local_path=path,
        )
    changed = temp_data_dir / "copy-2.bin"
    changed.write_bytes(b"new data")

    def interrupted(record):
        raise RuntimeError("interrupted after replacement")

    with pytest.raises(RuntimeError, match="interrupted"):
        consolidate_local(patient.label, interrupted, producers_quiescent=True)
    consolidate_local(patient.label, producers_quiescent=True)
    assert changed.read_bytes() == b"new data" and not changed.is_symlink()
    assert (temp_data_dir / "one.bin").read_bytes() == b"original"
    assert (temp_data_dir / "copy-0.bin").resolve() == (
        temp_data_dir / "copy-1.bin"
    ).resolve()
    assert consolidate_local(patient.label, producers_quiescent=True)["savedBytes"] == 0


def test_remote_cleanup_rejects_recorded_different_content(temp_data_dir):
    from backend.clinic_dedup import consolidate_remote

    patient, first = seed(temp_data_dir)
    path = temp_data_dir / "different.bin"
    path.write_bytes(b"different")
    other = catalogue.register_artifact(
        patient_uuid=patient.id,
        source_kind="renderer",
        source_id="other-content",
        logical_family="video",
        original_name=path.name,
        local_path=path,
    )
    first_key = f"patients/{patient.label}/files/{first['fileKey']}"
    other_key = f"patients/{patient.label}/files/{other['fileKey']}"
    catalogue.add_remote_location(first["fileId"], first_key)
    catalogue.add_remote_location(other["fileId"], other_key)
    with pytest.raises(catalogue.CatalogueConflict, match="different recorded"):
        consolidate_remote(
            patient.label,
            first_key,
            [other_key],
            hashlib.sha256(b"original").hexdigest(),
            8,
            lambda: [b"original"],
        )


@pytest.mark.parametrize("oversize", [False, True])
def test_shared_remote_corruption_revokes_all_file_receipts(
    temp_data_dir, monkeypatch, oversize
):
    from backend import clinic_publication as publisher

    patient, first = seed(temp_data_dir)
    second = catalogue.register_artifact(
        patient_uuid=patient.id,
        source_kind="renderer",
        source_id="corruption:two",
        logical_family="video",
        original_name="second.bin",
        local_path=temp_data_dir / "one.bin",
    )
    key = publisher.prepare_publication(first["fileId"])["item"]["remoteKey"]
    publisher.prepare_publication(second["fileId"])
    monkeypatch.setattr(publisher, "strong_readback", lambda key, size: [b"original"])
    publisher.verify_publication(first["fileId"], key)

    def bad_read(key, size):
        if oversize:
            raise publisher.ReadbackOversize("too large")
        return [b"changed!"]

    monkeypatch.setattr(publisher, "strong_readback", bad_read)
    with pytest.raises(catalogue.CatalogueConflict):
        publisher.verify_publication(first["fileId"], key)
    for a in (first, second):
        assert not publisher.prepare_publication(a["fileId"])["item"]["verified"]


def test_producer_replay_after_local_body_consolidation(live_api, monkeypatch):
    from backend.clinic_dedup import consolidate_local

    client, chart, root = live_api
    portal = root / "portal_patients"
    patient_folder = portal / chart.label
    patient_folder.mkdir(parents=True)
    monkeypatch.setenv("QEEG_PORTAL_PATIENTS_DIR", str(portal))
    (patient_folder / "video.mp4").write_bytes(b"original")
    operation = {
        "operationId": "cleanup-replay",
        "patientId": chart.label,
        "producer": "renderer",
        "kind": "video",
        "original": {"receiptId": "original"},
    }
    assert client.post("/internal/operations", json=operation).status_code == 200
    material = {
        "patientId": chart.label,
        "operationId": "cleanup-replay",
        "outputId": "video",
        "relativePath": "video.mp4",
        "originalName": "video.mp4",
        "logicalFamily": "video",
    }
    first = client.post("/internal/artifacts", json=material)
    assert first.status_code == 200
    consolidate_local(chart.label, producers_quiescent=True)
    replay = client.post("/internal/artifacts", json=material)
    assert replay.status_code == 200, replay.text
    assert replay.json()["artifact"]["fileId"] == first.json()["artifact"]["fileId"]


def test_reactivated_old_target_cannot_recreate_a_removed_duplicate(temp_data_dir):
    from backend import clinic_publication as publisher
    from backend.clinic_models import ClinicPublication
    from backend.clinic_dedup import consolidate_remote

    patient, first = seed(temp_data_dir)
    key = publisher.prepare_publication(first["fileId"])["item"]["remoteKey"]
    second = catalogue.register_artifact(
        patient_uuid=patient.id,
        source_kind="renderer",
        source_id="reactivate",
        logical_family="video",
        original_name="second.bin",
        local_path=temp_data_dir / "one.bin",
    )
    old = f"patients/{patient.label}/files/{second['fileKey']}"
    catalogue.add_remote_location(second["fileId"], old)
    with catalogue._write() as s:
        s.add(ClinicPublication(artifact_id=second["fileId"], remote_key=old))
    consolidate_remote(
        patient.label,
        key,
        [old],
        hashlib.sha256(b"original").hexdigest(),
        8,
        lambda: [b"original"],
    )
    catalogue.add_remote_location(second["fileId"], old)
    item = publisher.prepare_publication(second["fileId"])["item"]
    assert item["remoteKey"] == key and item["verified"]


@pytest.mark.parametrize("configured", ["inside", "outside"])
def test_cleanup_configured_root_and_noop_revision(
    temp_data_dir, monkeypatch, tmp_path, configured
):
    from backend.clinic_dedup import consolidate_local
    from backend import clinic_catalogue_reads as reads

    portal = (
        temp_data_dir / "custom-portal"
        if configured == "inside"
        else tmp_path.parent / (tmp_path.name + "-external")
    )
    monkeypatch.setenv("QEEG_PORTAL_PATIENTS_DIR", str(portal))
    patient, first = seed(temp_data_dir)
    path = temp_data_dir / "copy.bin"
    path.write_bytes(b"original")
    catalogue.register_artifact(
        patient_uuid=patient.id,
        source_kind="renderer",
        source_id="config-copy",
        logical_family="video",
        original_name=path.name,
        local_path=path,
    )
    assert (
        consolidate_local(patient.label, producers_quiescent=True)["removedBodies"] == 1
    )
    target = path.resolve()
    expected = temp_data_dir / ".content" / patient.id
    assert target.is_relative_to(expected)
    before = reads.current_revision()
    fingerprint = target.stat().st_ctime_ns
    assert consolidate_local(patient.label, producers_quiescent=True) == {
        "removedBodies": 0,
        "savedBytes": 0,
    }
    assert reads.current_revision() == before
    assert target.stat().st_ctime_ns == fingerprint
    with reads.open_local_file(first["fileId"]) as body:
        assert body.read() == b"original"


@pytest.mark.parametrize("mode", ["initial", "archive"])
def test_lightweight_pages_omit_retired_aliases_but_exact_history_resolves(
    temp_data_dir, mode
):
    from backend.clinic_dedup import consolidate_remote
    from backend import clinic_catalogue_reads as reads

    patient, artifact = seed(temp_data_dir)
    canonical = f'patients/{patient.label}/files/{artifact["fileKey"]}'
    retired = f"patients/{patient.label}/files/historic-copy"
    consolidate_remote(
        patient.label,
        canonical,
        [retired],
        hashlib.sha256(b"original").hexdigest(),
        8,
        lambda: [b"original"],
    )
    page = reads.patient_files(
        patient.label, mode=mode, page="1" if mode == "archive" else None
    )
    assert all(
        location["active"] for file in page["files"] for location in file["locations"]
    )
    for file in (
        reads.file_binding(patient.label, file_key="historic-copy"),
        reads.patient_files(patient.label)["files"][0],
    ):
        assert any(
            location["key"] == retired and not location["active"]
            for location in file["locations"]
        )
        assert any(
            location["key"] == canonical and location["verified"]
            for location in file["locations"]
        )


def test_cleanup_historical_alias_uses_engine_patient_label(temp_data_dir):
    from backend.clinic_dedup import consolidate_local

    patient, _ = seed(temp_data_dir)
    path = temp_data_dir / "duplicate-label.bin"
    path.write_bytes(b"original")
    catalogue.register_artifact(
        patient_uuid=patient.id,
        source_kind="renderer",
        source_id="label-copy",
        logical_family="video",
        original_name=path.name,
        local_path=path,
    )
    from backend.clinic_models import ClinicPatientAlias

    with storage.session_scope() as session:
        session.add(ClinicPatientAlias(alias="old-label", patient_uuid=patient.id))
        session.commit()
    consolidate_local("old-label", producers_quiescent=True)
    assert path.resolve().is_relative_to(temp_data_dir / ".content" / patient.id)


@pytest.mark.parametrize("qualified", [False, True])
def test_shared_content_keys_resolve_original_owner(temp_data_dir, qualified):
    from backend.clinic_dedup import consolidate_remote
    from backend import clinic_catalogue_reads as reads
    from backend.clinic_models import CatalogueNotFound

    patient, first = seed(temp_data_dir)
    path = temp_data_dir / "shared-copy.bin"
    path.write_bytes(b"original")
    second = catalogue.register_artifact(
        patient_uuid=patient.id,
        source_kind="renderer",
        source_id="shared-copy",
        logical_family="video",
        original_name=path.name,
        local_path=path,
    )
    prefix = f"patients/{patient.label}/files/"
    consolidate_remote(
        patient.label,
        prefix + first["fileKey"],
        [prefix + second["fileKey"]],
        hashlib.sha256(b"original").hexdigest(),
        8,
        lambda: [b"original"],
    )
    for file in (first, second):
        key = prefix + file["fileKey"] if qualified else file["fileKey"]
        assert (
            reads.file_binding(patient.label, file_key=key)["fileId"] == file["fileId"]
        )
    with pytest.raises(CatalogueNotFound):
        reads.file_binding(
            patient.label, file_key="patients/OTHER/files/" + first["fileKey"]
        )


def test_shared_legacy_key_requires_unambiguous_canonical_target(temp_data_dir):
    from backend.clinic_dedup import consolidate_remote

    patient, first = seed(temp_data_dir)
    path = temp_data_dir / "legacy-copy.bin"
    path.write_bytes(b"original")
    catalogue.register_artifact(
        patient_uuid=patient.id,
        source_kind="renderer",
        source_id="legacy-copy",
        logical_family="video",
        original_name=path.name,
        local_path=path,
    )
    prefix = f"patients/{patient.label}/files/"
    catalogue.add_remote_location(first["fileId"], prefix + "legacy-name.bin")
    with pytest.raises(catalogue.CatalogueConflict, match="database-issued canonical"):
        consolidate_remote(
            patient.label,
            prefix + "legacy-name.bin",
            [prefix + "other.bin"],
            hashlib.sha256(b"original").hexdigest(),
            8,
            lambda: [b"original"],
        )


def test_cleanup_keeps_canonical_body_after_alias_is_registered_again(temp_data_dir):
    from backend.clinic_dedup import consolidate_local

    patient, _ = seed(temp_data_dir)
    path = temp_data_dir / "reimport.bin"
    path.write_bytes(b"original")
    catalogue.register_artifact(
        patient_uuid=patient.id,
        source_kind="renderer",
        source_id="before",
        logical_family="video",
        original_name=path.name,
        local_path=path,
    )
    with pytest.raises(catalogue.CatalogueConflict, match="Stop local producers"):
        consolidate_local(patient.label)
    consolidate_local(patient.label, producers_quiescent=True)
    canonical = path.resolve()
    catalogue.register_artifact(
        patient_uuid=patient.id,
        source_kind="renderer",
        source_id="after",
        logical_family="video",
        original_name=path.name,
        local_path=path,
    )
    consolidate_local(patient.label, producers_quiescent=True)
    assert not canonical.is_symlink() and canonical.read_bytes() == b"original"
    assert path.read_bytes() == b"original"


@pytest.mark.parametrize("legacy", ["files/legacy.bin", ".archive/old.bin"])
def test_publication_does_not_share_legacy_or_archive_targets(temp_data_dir, legacy):
    from backend import clinic_publication as publisher

    patient, first = seed(temp_data_dir)
    path = temp_data_dir / "another.bin"
    path.write_bytes(b"original")
    second = catalogue.register_artifact(
        patient_uuid=patient.id,
        source_kind="renderer",
        source_id="another",
        logical_family="video",
        original_name=path.name,
        local_path=path,
    )
    key = f"patients/{patient.label}/{legacy}"
    catalogue.add_remote_location(first["fileId"], key)
    catalogue.verify_remote_location(first["fileId"], key, lambda: [b"original"])
    assert publisher.prepare_publication(second["fileId"])["item"]["remoteKey"] != key


def test_consolidated_target_survives_corruption_without_recreating_duplicates(
    temp_data_dir,
):
    from backend import clinic_publication as publisher
    from backend.clinic_dedup import consolidate_remote

    patient, first = seed(temp_data_dir)
    path = temp_data_dir / "shared.bin"
    path.write_bytes(b"original")
    second = catalogue.register_artifact(
        patient_uuid=patient.id,
        source_kind="renderer",
        source_id="shared",
        logical_family="video",
        original_name=path.name,
        local_path=path,
    )
    key = f'patients/{patient.label}/files/{first["fileKey"]}'
    old = f'patients/{patient.label}/files/{second["fileKey"]}'
    consolidate_remote(
        patient.label,
        key,
        [old],
        hashlib.sha256(b"original").hexdigest(),
        8,
        lambda: [b"original"],
    )
    with pytest.raises(catalogue.CatalogueConflict, match="Remote bytes differ"):
        catalogue.verify_remote_location(first["fileId"], key, lambda: [b"changed!"])
    for item in (first, second):
        prepared = publisher.prepare_publication(item["fileId"])["item"]
        assert prepared["remoteKey"] == key and not prepared["verified"]


def test_cleanup_includes_unindexed_patient_copies_and_keeps_distinct_files(
    temp_data_dir,
):
    from backend.clinic_dedup import consolidate_local

    patient, _ = seed(temp_data_dir)
    folder = temp_data_dir / "portal_patients" / patient.label
    folder.mkdir(parents=True)
    same = folder / "unindexed.bin"
    same.write_bytes(b"original")
    unique = folder / "unique.bin"
    unique.write_bytes(b"distinct")
    result = consolidate_local(patient.label, producers_quiescent=True)
    assert (
        result["removedBodies"] == 1
        and same.is_symlink()
        and same.read_bytes() == b"original"
    )
    assert unique.read_bytes() == b"distinct" and not unique.is_symlink()


def test_local_shared_body_and_portal_mirror_survive_patient_rekey(temp_data_dir):
    from backend.clinic_dedup import consolidate_local
    from backend import patient_rekey, portal_sync

    patient, _ = seed(temp_data_dir)
    portal = temp_data_dir / "portal_patients"
    folder = portal / patient.label
    folder.mkdir(parents=True)
    path = folder / "report.pdf"
    path.write_bytes(b"original")
    catalogue.register_artifact(
        patient_uuid=patient.id,
        source_kind="renderer",
        source_id="rekey",
        logical_family="video",
        original_name=path.name,
        local_path=path,
    )
    consolidate_local(patient.label, producers_quiescent=True)
    outside = temp_data_dir / "one.bin"
    retained = outside.resolve()
    assert path.resolve().samefile(retained)
    plan = patient_rekey.plan_patient_rekey(
        patient.label, "YZ_01-01-1900", portal_root=portal
    )
    patient_rekey.apply_patient_rekey(plan)
    assert outside.read_bytes() == b"original" and outside.resolve() == retained
    moved = portal / "YZ_01-01-1900"
    assert (moved / "report.pdf").read_bytes() == b"original"
    portal_sync._mirror_tree_with_hardlinks(moved, temp_data_dir / "staged")
    assert (temp_data_dir / "staged" / "report.pdf").read_bytes() == b"original"


def test_remote_cleanup_rejects_historical_patient_prefix(temp_data_dir):
    from backend.clinic_dedup import consolidate_remote
    from backend.clinic_models import ClinicPatientAlias

    patient, first = seed(temp_data_dir)
    with storage.session_scope() as s:
        s.add(ClinicPatientAlias(alias="old-label", patient_uuid=patient.id))
        s.commit()
    with pytest.raises(ValueError, match="canonical patient ID"):
        consolidate_remote(
            "old-label",
            "patients/old-label/files/" + first["fileKey"],
            [],
            hashlib.sha256(b"original").hexdigest(),
            8,
            lambda: [b"original"],
        )


@pytest.mark.parametrize("unselected_link", [False, True])
def test_cleanup_counts_only_released_physical_bodies(temp_data_dir, unselected_link):
    import os
    from backend.clinic_dedup import consolidate_local

    patient, _ = seed(temp_data_dir)
    first = temp_data_dir / "z1.bin"
    first.write_bytes(b"original")
    second = temp_data_dir / "z2.bin"
    os.link(first, second)
    for i, path in enumerate([first] if unselected_link else [first, second]):
        catalogue.register_artifact(
            patient_uuid=patient.id,
            source_kind="renderer",
            source_id=f"linked-{i}",
            logical_family="video",
            original_name=path.name,
            local_path=path,
        )
    result = consolidate_local(patient.label, producers_quiescent=True)
    assert result == {
        "removedBodies": 0 if unselected_link else 1,
        "savedBytes": 0 if unselected_link else 8,
    }
    assert first.read_bytes() == second.read_bytes() == b"original"


def test_local_shared_body_survives_merge_into_existing_patient_folder(temp_data_dir):
    from backend.clinic_dedup import consolidate_local
    from backend import patient_rekey, portal_sync

    patient, _ = seed(temp_data_dir)
    portal = temp_data_dir / "portal_patients"
    folder = portal / patient.label
    folder.mkdir(parents=True)
    path = folder / "report.pdf"
    path.write_bytes(b"original")
    catalogue.register_artifact(
        patient_uuid=patient.id,
        source_kind="renderer",
        source_id="merge",
        logical_family="video",
        original_name=path.name,
        local_path=path,
    )
    consolidate_local(patient.label, producers_quiescent=True)
    new = portal / "YZ_01-01-1900"
    new.mkdir()
    plan = patient_rekey.plan_patient_rekey(
        patient.label, "YZ_01-01-1900", portal_root=portal, merge_into_existing=True
    )
    patient_rekey.apply_patient_rekey(plan)
    assert (
        (new / "report.pdf").read_bytes()
        == (temp_data_dir / "one.bin").read_bytes()
        == b"original"
    )
    portal_sync._mirror_tree_with_hardlinks(new, temp_data_dir / "merge-stage")
    assert (temp_data_dir / "merge-stage" / "report.pdf").read_bytes() == b"original"


@pytest.mark.parametrize("same_inode", [True, False])
def test_merge_coalesces_only_verified_internal_content_anchors(
    temp_data_dir, same_inode
):
    import os
    from backend.clinic_dedup import consolidate_local
    from backend import patient_rekey

    patient, _ = seed(temp_data_dir)
    portal = temp_data_dir / "portal_patients"
    old = portal / patient.label
    old.mkdir(parents=True)
    named = old / "report.pdf"
    named.write_bytes(b"original")
    catalogue.register_artifact(
        patient_uuid=patient.id,
        source_kind="renderer",
        source_id="merge-anchor",
        logical_family="video",
        original_name=named.name,
        local_path=named,
    )
    consolidate_local(patient.label, producers_quiescent=True)
    anchor = named.resolve()
    new = portal / "YZ_01-01-1900"
    target = new / anchor.relative_to(old)
    target.parent.mkdir(parents=True)
    if same_inode:
        os.link(anchor, target)
    else:
        target.write_bytes(b"original")
    plan = patient_rekey.plan_patient_rekey(
        patient.label, "YZ_01-01-1900", portal_root=portal, merge_into_existing=True
    )
    patient_rekey.apply_patient_rekey(plan)
    assert (
        (new / "report.pdf").read_bytes()
        == (temp_data_dir / "one.bin").read_bytes()
        == b"original"
    )


@pytest.mark.parametrize("replan", [False, True])
def test_interrupted_merge_keeps_remaining_aliases_readable_and_resumes(
    temp_data_dir, monkeypatch, replan
):
    from backend.clinic_dedup import consolidate_local
    from backend import patient_rekey

    patient, _ = seed(temp_data_dir)
    portal = temp_data_dir / "portal_patients"
    old = portal / patient.label
    old.mkdir(parents=True)
    for name in ["report1.pdf", "report2.pdf"]:
        path = old / name
        path.write_bytes(b"original")
        catalogue.register_artifact(
            patient_uuid=patient.id,
            source_kind="renderer",
            source_id=name,
            logical_family="video",
            original_name=path.name,
            local_path=path,
        )
    consolidate_local(patient.label, producers_quiescent=True)
    new = portal / "YZ_01-01-1900"
    new.mkdir()

    def plan():
        return patient_rekey.plan_patient_rekey(
            patient.label, "YZ_01-01-1900", portal_root=portal, merge_into_existing=True
        )

    initial = plan()
    replace = patient_rekey.os.replace

    def interrupted(source, target):
        if source == old / "report2.pdf":
            raise OSError("interrupted merge")
        return replace(source, target)

    monkeypatch.setattr(patient_rekey.os, "replace", interrupted)
    with pytest.raises(OSError, match="interrupted merge"):
        patient_rekey.apply_patient_rekey(initial)
    assert (
        (old / "report2.pdf").read_bytes()
        == (new / "report1.pdf").read_bytes()
        == b"original"
    )
    monkeypatch.setattr(patient_rekey.os, "replace", replace)
    patient_rekey.apply_patient_rekey(plan() if replan else initial)
    assert (
        (new / "report1.pdf").read_bytes()
        == (new / "report2.pdf").read_bytes()
        == b"original"
    )
    assert not old.exists()
