"""Actual producer byte history and retry boundaries, without generation."""

import json
from pathlib import Path
import pytest
from sqlalchemy import select, func
from backend import storage, clinic_producers as producers
from backend.clinic_catalogue_reads import open_local_file
from backend.clinic_models import ClinicArtifact, CatalogueConflict
from backend.tests.test_patient_postprocessing import ready  # noqa: F401
from backend.tests.clinic_test_helpers import forbid_clinic_paid  # noqa: F401


def test_mutable_source_preserves_distinct_original_producer_bytes(temp_data_dir):
    with storage.session_scope() as s:
        patient = storage.create_patient(s, label="ZZ_01-01-1900")
    path = temp_data_dir / "latest.pdf"

    def register(source_id, data):
        path.write_bytes(data)
        return producers.register_original_output(
            patient_uuid=patient.id,
            source_kind="producer",
            source_id=source_id,
            path=path,
            original_name="latest.pdf",
            logical_family="pdf",
        )

    a = register("original-a", b"old")
    b = register("original-b", b"new")
    with open_local_file(a["fileId"]) as stream:
        assert stream.read() == b"old"
    with open_local_file(b["fileId"]) as stream:
        assert stream.read() == b"new"
    assert (a["version"], b["version"]) == (1, 2)
    assert register("original-b", b"new")["fileId"] == b["fileId"]
    with pytest.raises(CatalogueConflict):
        register("original-b", b"different")


@pytest.mark.parametrize("interrupt_kind", [None, "md", "pdf", "meta"])
def test_export_retry_reuses_original_timestamp_snapshots_and_versions(
    ready, monkeypatch, interrupt_kind
):
    from backend.portal_export_manifest import (
        council_export_manifest_payload,
        write_council_export_manifest,
    )

    store, run_id, _ = ready
    with storage.session_scope() as s:
        run = s.get(storage.Run, run_id)
        patient = s.get(storage.Patient, run.patient_id)
        selected = s.scalar(
            select(storage.Artifact).where(
                storage.Artifact.run_id == run_id, storage.Artifact.stage_num == 6
            )
        )
    root = Path(storage.DATA_DIR)
    md = root / "final.md"
    pdf = root / "final.pdf"
    portal = root / "portal_patients" / patient.label
    md.write_bytes(b"original md")
    pdf.write_bytes(b"original pdf")
    payload = council_export_manifest_payload(
        patient_label=patient.label,
        run_id=run_id,
        report_id=run.report_id,
        selected_artifact=selected,
        export_md_path=md,
        export_pdf_path=pdf,
        portal_md_path=None,
        portal_pdf_path=None,
    )
    original_register = producers.register_original_output
    if interrupt_kind:

        def interrupt(**kwargs):
            result = original_register(**kwargs)
            if kwargs["provenance"].get("outputKind") == interrupt_kind:
                raise RuntimeError("Interrupted after committed original output")
            return result

        monkeypatch.setattr(producers, "register_original_output", interrupt)
        with pytest.raises(RuntimeError):
            write_council_export_manifest(
                portal_patient_dir=portal, patient_label=patient.label, payload=payload
            )
        monkeypatch.setattr(producers, "register_original_output", original_register)
    path = write_council_export_manifest(
        portal_patient_dir=portal, patient_label=patient.label, payload=payload
    )
    original_meta = path.read_bytes()
    md.write_bytes(b"rendered again")
    pdf.write_bytes(b"rendered again")
    changed = {**payload, "exported_at": "2099-01-01T00:00:00+00:00"}
    write_council_export_manifest(
        portal_patient_dir=portal, patient_label=patient.label, payload=changed
    )
    assert md.read_bytes() == b"original md" and pdf.read_bytes() == b"original pdf"
    assert path.read_bytes() == original_meta
    assert json.loads(path.read_bytes())["exported_at"] == payload["exported_at"]
    with storage.session_scope() as s:
        assert s.scalar(select(func.count()).select_from(ClinicArtifact)) == 3
    Path(selected.content_path).write_text("changed selected source")
    with pytest.raises(CatalogueConflict):
        write_council_export_manifest(
            portal_patient_dir=portal, patient_label=patient.label, payload=payload
        )


def test_export_interrupted_before_snapshot_rejects_replacement_bytes(
    ready, monkeypatch
):
    from backend.portal_export_manifest import council_export_manifest_payload

    _, run_id, _ = ready
    with storage.session_scope() as s:
        run = s.get(storage.Run, run_id)
        patient = s.get(storage.Patient, run.patient_id)
        selected = s.scalar(
            select(storage.Artifact).where(
                storage.Artifact.run_id == run_id, storage.Artifact.stage_num == 6
            )
        )
    md = Path(storage.DATA_DIR) / "output.md"
    pdf = Path(storage.DATA_DIR) / "output.pdf"
    md.write_bytes(b"original md")
    pdf.write_bytes(b"original pdf")
    payload = council_export_manifest_payload(
        patient_label=patient.label,
        run_id=run_id,
        report_id=run.report_id,
        selected_artifact=selected,
        export_md_path=md,
        export_pdf_path=pdf,
        portal_md_path=None,
        portal_pdf_path=None,
    )
    original = producers.retained_producer_path

    def interrupted(*a, **kw):
        raise RuntimeError("death before first byte snapshot")

    monkeypatch.setattr(producers, "retained_producer_path", interrupted)
    with pytest.raises(RuntimeError):
        producers.freeze_council_export(payload)
    monkeypatch.setattr(producers, "retained_producer_path", original)
    md.write_bytes(b"replacement")
    with pytest.raises(CatalogueConflict):
        producers.freeze_council_export(payload)
