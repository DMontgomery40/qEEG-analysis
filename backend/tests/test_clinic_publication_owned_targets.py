"""Publication targets retain database ownership through imports and repair."""

import pytest
from sqlalchemy import select

from backend import (
    clinic_catalogue as catalogue,
    clinic_publication as publisher,
    storage,
)
from backend.clinic_models import ClinicLocation, ClinicPublication
from backend.tests.test_clinic_publication import seed


@pytest.mark.parametrize("prepare_owner_first", [False, True])
@pytest.mark.parametrize("repair_peer_first", [False, True])
def test_imported_target_keeps_original_owner_through_shared_repair(
    temp_data_dir, prepare_owner_first, repair_peer_first
):
    patient, owner = seed(temp_data_dir)
    peer = catalogue.register_artifact(
        patient_uuid=patient.id,
        source_kind="renderer",
        source_id="owned-target:peer",
        logical_family="video",
        original_name="peer.bin",
        local_path=temp_data_dir / "one.bin",
    )
    key = f"patients/{patient.label}/files/{owner['fileKey']}"
    catalogue.add_remote_location(owner["fileId"], key)
    catalogue.verify_remote_location(owner["fileId"], key, lambda: [b"original"])
    if prepare_owner_first:
        assert (
            publisher.prepare_publication(owner["fileId"])["item"]["remoteKey"] == key
        )
    assert publisher.prepare_publication(peer["fileId"])["item"]["remoteKey"] == key
    with storage.session_scope() as session:
        binding = session.get(ClinicPublication, owner["fileId"])
        assert binding is not None and binding.remote_key == key
        assert session.get(ClinicPublication, peer["fileId"]) is None
    with pytest.raises(catalogue.CatalogueConflict, match="Remote bytes differ"):
        catalogue.verify_remote_location(peer["fileId"], key, lambda: [b"changed!"])
    order = (peer, owner) if repair_peer_first else (owner, peer)
    for artifact in order:
        prepared = publisher.prepare_publication(artifact["fileId"])["item"]
        assert prepared["remoteKey"] == key and not prepared["verified"]
    with storage.session_scope() as session:
        assert set(
            session.scalars(
                select(ClinicLocation.key).where(
                    ClinicLocation.kind == "netlify", ClinicLocation.active.is_(True)
                )
            )
        ) == {key}
    catalogue.verify_remote_location(owner["fileId"], key, lambda: [b"original"])
    items = publisher.publication_items(patient.label)["items"]
    assert len(items) == 2
    assert all(item["remoteKey"] == key and item["verified"] for item in items)


@pytest.mark.parametrize("target_kind", ["legacy", "archive", "archive-owned-name"])
def test_current_artifact_import_does_not_become_a_publication_target(
    temp_data_dir, target_kind
):
    patient, artifact = seed(temp_data_dir)
    tail = {
        "legacy": "files/legacy.bin",
        "archive": ".archive/old.bin",
        "archive-owned-name": f".archive/{artifact['fileKey']}",
    }[target_kind]
    imported_key = f"patients/{patient.label}/{tail}"
    catalogue.add_remote_location(artifact["fileId"], imported_key)
    catalogue.verify_remote_location(
        artifact["fileId"], imported_key, lambda: [b"original"]
    )
    assert publisher.publication_items(patient.label)["items"][0]["remoteKey"] is None
    key = f"patients/{patient.label}/files/{artifact['fileKey']}"
    prepared = publisher.prepare_publication(artifact["fileId"])["item"]
    assert prepared["remoteKey"] == key and not prepared["verified"]
    assert publisher.publication_items(patient.label)["items"][0]["remoteKey"] == key


@pytest.mark.parametrize("owner_mismatch", ["patient", "content"])
def test_database_filename_requires_same_patient_and_content_owner(
    temp_data_dir, owner_mismatch
):
    patient, artifact = seed(temp_data_dir)
    owner_patient = patient
    if owner_mismatch == "patient":
        with storage.session_scope() as session:
            owner_patient = storage.create_patient(session, label="AZ_01-01-1900")
    path = temp_data_dir / "other-owner.bin"
    path.write_bytes(b"changed!" if owner_mismatch == "content" else b"original")
    other = catalogue.register_artifact(
        patient_uuid=owner_patient.id,
        source_kind="renderer",
        source_id="owned-target:other",
        logical_family="video",
        original_name=path.name,
        local_path=path,
    )
    key = f"patients/{patient.label}/files/{other['fileKey']}"
    catalogue.add_remote_location(artifact["fileId"], key)
    catalogue.verify_remote_location(artifact["fileId"], key, lambda: [b"original"])
    item = next(
        item
        for item in publisher.publication_items(patient.label)["items"]
        if item["fileId"] == artifact["fileId"]
    )
    assert item["remoteKey"] is None
    prepared = publisher.prepare_publication(artifact["fileId"])["item"]
    assert (
        prepared["remoteKey"] == f"patients/{patient.label}/files/{artifact['fileKey']}"
    )


@pytest.mark.parametrize("prepare_owner_first", [False, True])
@pytest.mark.parametrize("repair_peer_first", [False, True])
def test_imported_relabel_target_replaces_obsolete_owner_receipt(
    temp_data_dir, prepare_owner_first, repair_peer_first
):
    patient, owner = seed(temp_data_dir)
    peer = catalogue.register_artifact(
        patient_uuid=patient.id,
        source_kind="renderer",
        source_id="owned-target:relabel-peer",
        logical_family="video",
        original_name="peer.bin",
        local_path=temp_data_dir / "one.bin",
    )
    old_key = publisher.prepare_publication(owner["fileId"])["item"]["remoteKey"]
    with storage.session_scope() as session:
        storage.update_patient(session, patient.id, label="AZ_01-01-1900")
        old_location = session.scalar(
            select(ClinicLocation).where(
                ClinicLocation.artifact_id == owner["fileId"],
                ClinicLocation.key == old_key,
            )
        )
        old_location.active = False
        session.commit()
    key = f"patients/AZ_01-01-1900/files/{owner['fileKey']}"
    catalogue.add_remote_location(owner["fileId"], key)
    catalogue.verify_remote_location(owner["fileId"], key, lambda: [b"original"])
    if prepare_owner_first:
        assert (
            publisher.prepare_publication(owner["fileId"])["item"]["remoteKey"] == key
        )
    assert publisher.prepare_publication(peer["fileId"])["item"]["remoteKey"] == key
    with storage.session_scope() as session:
        assert session.get(ClinicPublication, owner["fileId"]).remote_key == key
        old_location = session.scalar(
            select(ClinicLocation).where(
                ClinicLocation.artifact_id == owner["fileId"],
                ClinicLocation.key == old_key,
            )
        )
        assert old_location is not None and not old_location.active
    with pytest.raises(catalogue.CatalogueConflict, match="Remote bytes differ"):
        catalogue.verify_remote_location(peer["fileId"], key, lambda: [b"changed!"])
    order = (peer, owner) if repair_peer_first else (owner, peer)
    for artifact in order:
        prepared = publisher.prepare_publication(artifact["fileId"])["item"]
        assert prepared["remoteKey"] == key and not prepared["verified"]
    with storage.session_scope() as session:
        assert set(
            session.scalars(
                select(ClinicLocation.key).where(
                    ClinicLocation.kind == "netlify", ClinicLocation.active.is_(True)
                )
            )
        ) == {key}
