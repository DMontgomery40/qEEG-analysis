"""Complete synthetic inventories, honest missing legacy sources and replay."""

import json
import pytest
from sqlalchemy import select, func
from backend import (
    storage,
    clinic_reconciliation as reconcile,
    clinic_catalogue as catalogue,
)
from backend.clinic_models import (
    ClinicArtifact,
    ClinicProjection,
    CatalogueUnavailable,
    CatalogueConflict,
)
from backend.tests.clinic_test_helpers import forbid_clinic_paid  # noqa: F401


def census(keys):
    return [
        {"type": "page", "page": 1, "keys": keys},
        {"type": "complete", "pages": 1, "keyCount": len(keys)},
    ]


def source(root, label, name="one.txt", data=b"original"):
    with storage.session_scope() as s:
        patient = storage.create_patient(s, label=label)
        path = root / name
        if data is not None:
            path.write_bytes(data)
        report = storage.create_report(
            s,
            patient_id=patient.id,
            report_id=name,
            filename=name,
            mime_type="text/plain",
            stored_path=path,
            extracted_text_path=path,
        )
    return patient, report


def test_missing_legacy_retains_original_row_without_fake_artifact(temp_data_dir):
    patient, report = source(temp_data_dir, "legacy-row", data=None)
    inventory = reconcile.build_inventory(
        "legacy",
        remote_events=census([]),
        remote_readback=lambda *a: (),
        max_file_bytes=1024,
    )
    assert inventory["complete"]
    result = reconcile.import_inventory(
        "legacy", remote_readback=lambda *a: (), activate=True
    )
    assert result["retainedUnresolvedSources"][0]["sourceId"] == report.id
    with storage.session_scope() as s:
        assert s.get(storage.Report, report.id).stored_path == report.stored_path
        assert s.scalar(select(func.count()).select_from(ClinicArtifact)) == 0
        assert s.scalar(select(ClinicProjection)).artifact_id is None
    assert (
        reconcile.import_inventory(
            "legacy", remote_readback=lambda *a: (), activate=True
        )
        == result
    )
    with storage.session_scope() as s:
        storage.update_patient(s, patient.id, label="ZZ_01-01-1900")
    with pytest.raises(CatalogueConflict):
        catalogue.complete_catalogue_import(
            {
                k: result[k]
                for k in (
                    "inventoryId",
                    "legacyPatientIds",
                    "retainedUnresolvedSources",
                )
            }
        )


def test_canonical_missing_and_failed_remote_census_never_complete(temp_data_dir):
    source(temp_data_dir, "ZZ_01-01-1900", data=None)
    inventory = reconcile.build_inventory(
        "bad",
        remote_events=[{"type": "page", "page": 1, "keys": []}],
        remote_readback=lambda *a: (),
        max_file_bytes=1024,
    )
    assert not inventory["complete"] and len(inventory["errors"]) == 2
    with pytest.raises(CatalogueUnavailable):
        reconcile.import_inventory("bad", remote_readback=lambda *a: (), activate=True)


@pytest.mark.parametrize(
    "events",
    [
        [],
        [{"type": "complete", "pages": 1, "keyCount": 0}],
        [{"type": "page", "page": 2, "keys": []}],
        [{"type": "page", "page": 1, "keys": ["same", "same"]}],
        [
            {"type": "page", "page": 1, "keys": ["one"]},
            {"type": "complete", "pages": 1, "keyCount": 0},
        ],
    ],
)
def test_malformed_census_is_not_empty_authority(events):
    with pytest.raises(CatalogueUnavailable):
        list(reconcile.validate_remote_census(events))


def test_original_copy_and_remote_only_history_replay_exactly(temp_data_dir):
    patient, report = source(temp_data_dir, "ZZ_01-01-1900")
    key = f"patients/{patient.label}/files/original.txt"
    oldkey = f"patients/{patient.label}/files/older.txt"
    objects = {key: b"original", oldkey: b"remote-only"}
    bindings = {
        key: dict(patientUuid=patient.id, sourceKind="report", sourceId=report.id),
        oldkey: dict(
            patientUuid=patient.id,
            sourceKind="netlify-history",
            sourceId=oldkey,
            metadata={"originalName": "older.txt", "logicalFamily": "historical"},
        ),
    }

    def read(key, limit):
        return iter([objects[key]])

    reconcile.build_inventory(
        "complete",
        remote_events=census(list(objects)),
        remote_readback=read,
        max_file_bytes=1024,
        bindings=bindings,
    )
    a = reconcile.import_inventory("complete", remote_readback=read, activate=True)
    with storage.session_scope() as s:
        assert s.scalar(select(func.count()).select_from(ClinicArtifact)) == 2
    assert (
        reconcile.import_inventory("complete", remote_readback=read, activate=True) == a
    )
    from backend.clinic_catalogue_reads import patient_files

    files = patient_files(patient.label)["files"]
    assert len(files) == 2 and all(f["hashVerified"] for f in files)
    assert (
        next(f for f in files if f["originalName"] == "older.txt")["generatedAt"]
        is None
    )


def test_inventory_retains_symlink_directory_failure(temp_data_dir, monkeypatch):
    portal = temp_data_dir / "portal"
    portal.mkdir()
    (portal / "escaped").symlink_to(temp_data_dir.parent, target_is_directory=True)
    monkeypatch.setattr("backend.portal_sync.portal_patients_dir", lambda: portal)
    result = reconcile.build_inventory(
        "symlink",
        remote_events=census([]),
        remote_readback=lambda *a: (),
        max_file_bytes=100,
    )
    assert not result["complete"]
    assert any(e.get("reason") == "symlink_directory" for e in result["errors"])


def test_inventory_replay_rejects_changed_legacy_evidence(temp_data_dir):
    reconcile.build_inventory(
        "immutable",
        remote_events=census([]),
        remote_readback=lambda *a: (),
        max_file_bytes=100,
        legacy_upload_records=[{"uploadId": "old", "status": "pending"}],
    )
    with pytest.raises(CatalogueConflict):
        reconcile.build_inventory(
            "immutable",
            remote_events=census([]),
            remote_readback=lambda *a: (),
            max_file_bytes=100,
            legacy_upload_records=[{"uploadId": "old", "status": "registered"}],
        )


def test_submission_import_binds_original_registered_sources_without_new_chart(
    temp_data_dir,
):
    from backend.clinic_intake import submit_upload, get_upload
    from backend.tests.test_clinic_upload_import import original

    receipt = original()
    registered = submit_upload(
        key="existing-file",
        identity=receipt["manifest"]["identity"],
        resolution={},
        file_meta=[{}],
        files=[("same.txt", b"one", "text/plain")],
    )["upload"]
    with storage.session_scope() as s:
        patient = s.scalar(
            select(storage.Patient).where(
                storage.Patient.label == registered["patientId"]
            )
        )
    key = receipt["response"]["uploaded"][0]["fileKey"]
    receipt_key = "uploads/submissions/original-id.json"
    objects = {
        key: b"one",
        receipt_key: json.dumps(receipt, separators=(",", ":")).encode(),
    }
    binding = dict(
        patientUuid=patient.id,
        sourceKind="patient-file",
        sourceId=registered["items"][0]["sourceId"],
    )

    def read(key, limit):
        return iter([objects[key]])

    reconcile.build_inventory(
        "submission",
        remote_events=census(list(objects)),
        remote_readback=read,
        max_file_bytes=2048,
        bindings={key: binding},
    )
    reconcile.import_inventory("submission", remote_readback=read, activate=True)
    imported = get_upload("original-id")["upload"]
    assert imported["status"] == "registered"
    assert imported["patientId"] == registered["patientId"]
    assert imported["items"][0]["sourceId"] == registered["items"][0]["sourceId"]
    with storage.session_scope() as s:
        assert s.scalar(select(func.count()).select_from(storage.Patient)) == 1
        assert s.scalar(select(func.count()).select_from(storage.PatientFile)) == 1


def test_original_archived_bytes_and_feedback_hash_survive_import(temp_data_dir):
    import hashlib
    from backend.clinic_feedback import feedback_history

    with storage.session_scope() as session:
        patient = storage.create_patient(session, label="ZZ_01-01-1900")
    key = f"patients/{patient.label}/.archive/old.pdf"
    feedback_key = f"patients/{patient.label}/.archive/feedback/old.pdf.json"
    raw = b'{ "action":"reject", "notes":"original reason", "submittedBy":"Doctor", "submittedAt":123 }'
    objects = {key: b"oldbytes", feedback_key: raw}
    binding = dict(
        patientUuid=patient.id,
        sourceKind="netlify-history",
        sourceId=key,
        metadata=dict(originalName="old.pdf", logicalFamily="original-family"),
    )

    def read(key, limit):
        return iter([objects[key]])

    reconcile.build_inventory(
        "archived",
        remote_events=census(list(objects)),
        remote_readback=read,
        max_file_bytes=1024,
        bindings={key: binding, feedback_key: binding},
    )
    reconcile.import_inventory("archived", remote_readback=read, activate=True)
    with storage.session_scope() as session:
        artifact = session.scalar(select(ClinicArtifact))
        assert artifact.archived
    history = feedback_history(artifact.id)
    expected = (
        "legacy-feedback:"
        + hashlib.sha256(
            (feedback_key + "\n" + hashlib.sha256(raw).hexdigest()).encode()
        ).hexdigest()
    )
    assert len(history) == 1 and history[0]["eventId"] == expected
    assert history[0]["notification"] is None
    assert history[0]["submittedAt"] == 123
    reconcile.import_inventory("archived", remote_readback=read, activate=True)
    assert feedback_history(artifact.id) == history


@pytest.mark.parametrize("relabels", [0, 1, 2])
def test_imported_file_alias_preserves_collision_ownership(temp_data_dir, relabels):
    from backend.clinic_catalogue_reads import file_binding

    old = "ZZ_01-01-1900"
    patients = []
    artifacts = []
    for i in range(2):
        patient, report = source(
            temp_data_dir, old, name=f"file-{i}.txt", data=str(i).encode()
        )
        patients.append(patient)
        with storage.session_scope() as session:
            artifacts.append(
                session.scalar(
                    select(ClinicArtifact).where(ClinicArtifact.source_id == report.id)
                )
            )
    for i in range(relabels):
        with storage.session_scope() as session:
            storage.update_patient(
                session, patients[i].id, label=f"ZZ_01-01-1900_{i+2}"
            )
    for artifact in artifacts:
        evidence = dict(
            patientUuid=artifact.patient_uuid,
            fileId=artifact.id,
            patientAlias=old,
            relativePath="original/file.pdf",
            sha256=artifact.sha256,
            evidence={"journalHash": "original-exact-journal-hash"},
        )
        assert reconcile.import_file_alias(evidence) == artifact.id
        assert reconcile.import_file_alias(evidence) == artifact.id
        if relabels == 0:
            with pytest.raises(CatalogueConflict):
                file_binding(old, file_id=artifact.id)
        else:
            assert file_binding(old, file_id=artifact.id)["fileId"] == artifact.id


def test_original_remote_alias_import_after_both_collision_relabels(temp_data_dir):
    old = "ZZ_01-01-1900"
    patients = [
        source(temp_data_dir, old, name=f"item-{i}.txt", data=str(i).encode())
        for i in range(2)
    ]
    for i, (patient, _) in enumerate(patients):
        with storage.session_scope() as session:
            storage.update_patient(session, patient.id, label=f"ZZ_01-01-1900_{i+2}")
    objects = {f"patients/{old}/files/item-{i}.txt": str(i).encode() for i in range(2)}
    bindings = {
        key: dict(patientUuid=patient.id, sourceKind="report", sourceId=report.id)
        for key, (patient, report) in zip(objects, patients)
    }

    def read(key, limit):
        return iter([objects[key]])

    reconcile.build_inventory(
        "remote-alias",
        remote_events=census(list(objects)),
        remote_readback=read,
        max_file_bytes=10,
        bindings=bindings,
    )
    reconcile.import_inventory("remote-alias", remote_readback=read, activate=True)
    from backend.clinic_catalogue_reads import file_binding

    for key, (patient, _) in zip(objects, patients):
        assert file_binding(old, file_key=key)["patientId"] != old


@pytest.mark.parametrize("boundary", ["before_location", "after_verification"])
def test_real_import_process_replacement_reuses_durable_receipts(
    temp_data_dir, boundary
):
    import os
    import subprocess
    import sys

    patient, report = source(temp_data_dir, "ZZ_01-01-1900")
    key = f"patients/{patient.label}/files/original.txt"
    reconcile.build_inventory(
        "interrupted",
        remote_events=census([key]),
        remote_readback=lambda *a: iter([b"original"]),
        max_file_bytes=100,
        bindings={
            key: dict(patientUuid=patient.id, sourceKind="report", sourceId=report.id)
        },
    )
    code = """
import os, signal, sys
from backend import storage, clinic_reconciliation as reconcile
from backend.paid_transport import PaidSyncTransport, PaidAsyncTransport
def forbidden(*a, **k): raise AssertionError('Paid transport forbidden')
PaidSyncTransport.handle_request = forbidden
PaidAsyncTransport.handle_async_request = forbidden
storage.init_db()
original = reconcile._add_exact_location
def interrupted(*args):
    if sys.argv[1] == 'after_verification': original(*args)
    os.kill(os.getpid(),signal.SIGKILL)
reconcile._add_exact_location = interrupted
reconcile.import_inventory('interrupted',remote_readback=lambda *a: iter([b'original']),activate=True)
"""
    child = subprocess.run(
        [sys.executable, "-c", code, boundary],
        capture_output=True,
        text=True,
        timeout=20,
        env={
            **os.environ,
            "DATA_DIR": str(temp_data_dir),
            "QEEG_ANALYSIS_ROOT": str(temp_data_dir.parent),
        },
    )
    assert child.returncode == -9, child.stderr
    result = reconcile.import_inventory(
        "interrupted", remote_readback=lambda *a: iter([b"original"]), activate=True
    )
    assert result["activated"]
    assert (
        reconcile.import_inventory(
            "interrupted", remote_readback=lambda *a: iter([b"original"]), activate=True
        )
        == result
    )
    with storage.session_scope() as session:
        assert session.scalar(select(func.count()).select_from(ClinicArtifact)) == 1


@pytest.mark.parametrize("source_kind", ["report", "patient-file"])
@pytest.mark.parametrize("replacement", [None, b"replaced", b"different-sized bytes"])
def test_first_import_binds_original_inventory_bytes(
    temp_data_dir, source_kind, replacement
):
    from backend.clinic_models import ClinicCatalogState

    original = b"original"
    path = temp_data_dir / "historical.txt"
    path.write_bytes(original)
    model = storage.Report if source_kind == "report" else storage.PatientFile
    with storage.session_scope() as s:
        patient = storage.create_patient(s, label="ZZ_01-01-1900")
    # Historical original rows precede catalogue hooks. Core insertion models
    # that first-import boundary without creating an artifact beforehand.
    with storage.engine.begin() as connection:
        connection.execute(
            model.__table__.insert().values(
                id="historical-source",
                patient_id=patient.id,
                filename=path.name,
                mime_type="text/plain",
                stored_path=str(path),
                **(
                    {"extracted_text_path": str(path)}
                    if source_kind == "report"
                    else {}
                ),
            )
        )
        connection.execute(
            ClinicCatalogState.__table__.update().values(import_complete=False)
        )
    inventory = reconcile.build_inventory(
        "original-bytes",
        remote_events=census([]),
        remote_readback=lambda *a: (),
        max_file_bytes=1024,
    )
    row = next(
        r
        for r in reconcile.inventory_rows(inventory)
        if r.get("sourceId") == "historical-source"
    )
    if replacement is not None:
        path.write_bytes(replacement)
        for _ in range(2):
            with pytest.raises(CatalogueUnavailable):
                reconcile.import_inventory(
                    "original-bytes", remote_readback=lambda *a: (), activate=True
                )
            with storage.session_scope() as s:
                assert s.scalar(select(func.count()).select_from(ClinicArtifact)) == 0
                assert not s.get(ClinicCatalogState, 1).import_complete
                assert (
                    s.scalar(
                        select(func.count())
                        .select_from(ClinicProjection)
                        .where(ClinicProjection.artifact_id.is_not(None))
                    )
                    == 0
                )
                assert s.get(model, "historical-source").stored_path == str(path)
        path.write_bytes(original)
    # Restoration and untouched sources both admit the original bytes; exact
    # replay retains the same identity/version and successful activation.
    result = reconcile.import_inventory(
        "original-bytes", remote_readback=lambda *a: (), activate=True
    )
    assert result["activated"]
    assert (
        reconcile.import_inventory(
            "original-bytes", remote_readback=lambda *a: (), activate=True
        )
        == result
    )
    with storage.session_scope() as s:
        artifacts = list(s.scalars(select(ClinicArtifact)))
        assert len(artifacts) == 1
        assert (artifacts[0].sha256, artifacts[0].size) == (row["sha256"], row["size"])
        assert (
            artifacts[0].source_kind,
            artifacts[0].source_id,
            artifacts[0].version,
        ) == (source_kind, "historical-source", 1)
        assert s.get(ClinicCatalogState, 1).import_complete


@pytest.mark.parametrize(
    "name",
    [
        ".DS_Store",
        ".qeeg_portal_netlify_sync.lock",
        ".qeeg_portal_sync_state.json",
        "_README.txt",
        ".qeeg_portal_local_pipeline_state.json",
        ".qeeg_portal_sync_watch_state.json",
        ".qeeg_portal_netlify_sync.spawn.lock",
    ],
)
def test_root_operational_inventory_retains_bytes_and_replays(
    temp_data_dir, monkeypatch, name
):
    portal = temp_data_dir / "portal_patients"
    portal.mkdir()
    monkeypatch.setattr("backend.portal_sync.portal_patients_dir", lambda: portal)
    path = portal / name
    path.write_bytes(b"original root operational bytes")
    inventory = reconcile.build_inventory(
        "operational",
        remote_events=census([]),
        remote_readback=lambda *a: (),
        max_file_bytes=1024,
    )
    row = next(reconcile.inventory_rows(inventory))
    assert row["kind"] == "root-operational"
    retained = (
        reconcile._inventory_root("operational") / "objects" / row["rawOperational"]
    )
    assert retained.read_bytes() == path.read_bytes()
    first = reconcile.import_inventory(
        "operational", remote_readback=lambda *a: (), activate=True
    )
    assert (
        reconcile.import_inventory(
            "operational", remote_readback=lambda *a: (), activate=True
        )
        == first
    )
    with storage.session_scope() as session:
        assert session.scalar(select(func.count()).select_from(ClinicArtifact)) == 0
    path.write_bytes(b"changed root operational bytes")
    with pytest.raises(CatalogueUnavailable):
        reconcile.import_inventory(
            "operational", remote_readback=lambda *a: (), activate=True
        )
    assert retained.read_bytes() == b"original root operational bytes"


@pytest.mark.parametrize(
    "relative",
    [
        "unknown.json",
        "ZZ_01-01-1900/.DS_Store.backup",
        "ZZ_01-01-1900/council/ordinary.json",
        "ZZ_01-01-1900/.qeeg_portal_sync_state.json",
    ],
)
def test_operational_retention_never_hides_unknown_or_nested_files(
    temp_data_dir, monkeypatch, relative
):
    portal = temp_data_dir / "portal_patients"
    path = portal / relative
    path.parent.mkdir(parents=True)
    path.write_bytes(b"must retain unresolved ownership")
    monkeypatch.setattr("backend.portal_sync.portal_patients_dir", lambda: portal)
    inventory = reconcile.build_inventory(
        "not-operational",
        remote_events=census([]),
        remote_readback=lambda *a: (),
        max_file_bytes=1024,
    )
    assert next(reconcile.inventory_rows(inventory))["kind"] == "local-file"
    with pytest.raises(CatalogueUnavailable):
        reconcile.import_inventory(
            "not-operational", remote_readback=lambda *a: (), activate=True
        )


@pytest.mark.parametrize(
    "relative", ["ZZ_01-01-1900/.DS_Store", "ZZ_01-01-1900/council/nested/.DS_Store"]
)
def test_nested_filesystem_metadata_retains_exact_bytes_only(
    temp_data_dir, monkeypatch, relative
):
    portal = temp_data_dir / "portal_patients"
    path = portal / relative
    path.parent.mkdir(parents=True)
    raw = b"original filesystem metadata"
    path.write_bytes(raw)
    monkeypatch.setattr("backend.portal_sync.portal_patients_dir", lambda: portal)
    manifest = reconcile.build_inventory(
        "nested-metadata",
        remote_events=census([]),
        remote_readback=lambda *a: (),
        max_file_bytes=1024,
    )
    row = next(reconcile.inventory_rows(manifest))
    assert row["kind"] == "filesystem-metadata"
    assert (
        reconcile._inventory_root("nested-metadata") / "objects" / row["rawOperational"]
    ).read_bytes() == raw
    reconcile.import_inventory(
        "nested-metadata", remote_readback=lambda *a: (), activate=True
    )
    path.write_bytes(b"changed filesystem metadata")
    with pytest.raises(CatalogueUnavailable):
        reconcile.import_inventory(
            "nested-metadata", remote_readback=lambda *a: (), activate=True
        )
    with storage.session_scope() as s:
        assert s.scalar(select(func.count()).select_from(ClinicArtifact)) == 0


@pytest.mark.parametrize(
    "case",
    [
        "current",
        "stale-initials",
        "mismatched",
        "unknown",
        "nested",
        "malformed",
        "not-object",
    ],
)
def test_local_patient_metadata_is_historical_evidence_not_identity(
    temp_data_dir, monkeypatch, case
):
    import json

    label = "ZZ_01-01-1900"
    with storage.session_scope() as s:
        patient = storage.create_patient(
            s, label=label, first_name="Zoe", last_name="Zed"
        )
    portal = temp_data_dir / "portal_patients"
    path = portal / ("XY_01-01-1900" if case == "unknown" else label)
    if case == "nested":
        path /= "council"
    path /= "$meta.json"
    path.parent.mkdir(parents=True)
    raw = json.dumps(
        dict(
            patientId="other" if case == "mismatched" else path.parent.name,
            birthdate="01-01-1900",
            index=1,
            identity={
                "firstInitial": "B" if case == "stale-initials" else "Z",
                "lastInitial": "Z",
            },
        )
    ).encode()
    if case == "malformed":
        raw = b"{"
    if case == "not-object":
        raw = b"[]"
    path.write_bytes(raw)
    monkeypatch.setattr("backend.portal_sync.portal_patients_dir", lambda: portal)
    manifest = reconcile.build_inventory(
        "patient-metadata",
        remote_events=census([]),
        remote_readback=lambda *a: (),
        max_file_bytes=1024,
    )
    row = next(reconcile.inventory_rows(manifest))
    if case in {"current", "stale-initials"}:
        assert row["kind"] == "patient-metadata"
        assert (
            reconcile._inventory_root("patient-metadata")
            / "objects"
            / row["rawOperational"]
        ).read_bytes() == raw
        reconcile.import_inventory(
            "patient-metadata", remote_readback=lambda *a: (), activate=True
        )
        reconcile.import_inventory(
            "patient-metadata", remote_readback=lambda *a: (), activate=True
        )
        with storage.session_scope() as s:
            assert storage.get_patient(s, patient.id).first_name == "Zoe"
            assert s.scalar(select(func.count()).select_from(ClinicArtifact)) == 0
        path.write_bytes(b"{}")
    else:
        assert row["kind"] == "local-file"
    with pytest.raises(CatalogueUnavailable):
        reconcile.import_inventory(
            "patient-metadata", remote_readback=lambda *a: (), activate=True
        )


@pytest.mark.parametrize("change", ["duplicate", "relabel", "symlink"])
def test_patient_metadata_replay_requires_current_unique_owner_and_regular_path(
    temp_data_dir, monkeypatch, change
):
    import json

    label = "ZZ_01-01-1900"
    with storage.session_scope() as s:
        patient = storage.create_patient(s, label=label)
    portal = temp_data_dir / "portal_patients"
    path = portal / label / "$meta.json"
    path.parent.mkdir(parents=True)
    raw = json.dumps({"patientId": label}).encode()
    path.write_bytes(raw)
    monkeypatch.setattr("backend.portal_sync.portal_patients_dir", lambda: portal)
    reconcile.build_inventory(
        "owner-change",
        remote_events=census([]),
        remote_readback=lambda *a: (),
        max_file_bytes=1024,
    )
    if change == "symlink":
        target = temp_data_dir / "unchanged-bytes.json"
        target.write_bytes(raw)
        path.unlink()
        path.symlink_to(target)
    else:
        with storage.session_scope() as s:
            if change == "duplicate":
                storage.create_patient(s, label=label)
            else:
                storage.get_patient(s, patient.id).label = "XY_01-01-1900"
                s.commit()
    with pytest.raises(CatalogueUnavailable):
        reconcile.import_inventory(
            "owner-change", remote_readback=lambda *a: (), activate=True
        )
