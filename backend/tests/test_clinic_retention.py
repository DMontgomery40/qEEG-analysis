"""Original global-object retention and linear durable import evidence."""

import hashlib
import json
import uuid
import pytest
from sqlalchemy import select, func
from backend import storage, clinic_reconciliation as reconcile
from backend.clinic_models import (
    ClinicArtifact,
    CatalogueConflict,
    CatalogueUnavailable,
)
from backend.tests.test_clinic_reconciliation import census
from backend.tests.clinic_test_helpers import forbid_clinic_paid  # noqa:F401


def evidence(category, data=b"original"):
    key = (
        "videos/original.mp4"
        if category == "legacy-video-library"
        else "uploads/" + str(uuid.uuid4())
    )
    row = dict(
        key=key,
        category=category,
        sha256=hashlib.sha256(data).hexdigest(),
        size=len(data),
    )
    if category == "legacy-unfiled-upload":
        row["metadata"] = dict(
            id=key.split("/")[1],
            filename="original.pdf",
            contentType="application/pdf",
            size=len(data),
            uploadedAt=1234,
            uploadedBy="original-clinic",
            extra="preserved",
        )
    return row


def build(name, rows, keys=None, data=b"original"):
    return reconcile.build_inventory(
        name,
        remote_events=census(keys if keys is not None else [r["key"] for r in rows]),
        remote_readback=lambda *a: iter([data]),
        max_file_bytes=100,
        retained_remote_objects=rows,
    )


@pytest.mark.parametrize("category", ["legacy-video-library", "legacy-unfiled-upload"])
def test_retention_preserves_original_evidence_without_charts(temp_data_dir, category):
    row = evidence(category)
    result = build("retained", [row])
    assert result["complete"] and result["retainedRemoteObjects"] == [row]
    for _ in range(2):
        imported = reconcile.import_inventory(
            "retained", remote_readback=lambda *a: iter([b"original"]), activate=True
        )
        assert imported["outcomes"] == [
            dict(row=0, status="retained", result="retained_remote")
        ]
    with storage.session_scope() as s:
        assert s.scalar(select(func.count()).select_from(ClinicArtifact)) == 0
        assert s.scalar(select(func.count()).select_from(storage.Patient)) == 0
    with pytest.raises(CatalogueUnavailable):
        reconcile.import_inventory(
            "retained", remote_readback=lambda *a: iter([b"replaced"]), activate=True
        )
    changed = json.loads(json.dumps(row))
    changed["sha256"] = "f" * 64
    with pytest.raises(CatalogueConflict):
        build("retained", [changed])


@pytest.mark.parametrize(
    "defect",
    [
        "category",
        "patients",
        "archive",
        "pending",
        "nested",
        "duplicate",
        "metadata",
        "metadata_size",
        "metadata_id",
    ],
)
def test_invalid_retention_never_excludes_patient_or_unproven_objects(
    temp_data_dir, defect
):
    row = evidence("legacy-unfiled-upload")
    if defect == "category":
        row["category"] = "patient-file"
    if defect == "patients":
        row["key"] = "patients/ZZ_01-01-1900/files/file.mp4"
    if defect == "archive":
        row["key"] = ".archive/original.mp4"
    if defect == "pending":
        row["key"] = "uploads/pending/" + str(uuid.uuid4())
    if defect == "nested":
        row.update(category="legacy-video-library", key="videos/subfolder/one.mp4")
    if defect == "metadata":
        row.pop("metadata")
    if defect == "metadata_size":
        row["metadata"]["size"] += 1
    if defect == "metadata_id":
        row["metadata"]["id"] = str(uuid.uuid4())
    with pytest.raises((ValueError, CatalogueConflict)):
        build("invalid", [row, row] if defect == "duplicate" else [row])


@pytest.mark.parametrize("defect", ["missing", "extra", "changed", "read_failure"])
def test_census_and_actual_bytes_must_cover_retention_exactly(temp_data_dir, defect):
    row = evidence("legacy-video-library")
    if defect == "missing":
        result = build("bad", [], keys=[row["key"]])
    elif defect == "extra":
        result = build("bad", [row], keys=[])
    elif defect == "changed":
        result = build("bad", [row], data=b"replaced")
    else:

        def fail(*a):
            raise OSError("unreadable")

        result = reconcile.build_inventory(
            "bad",
            remote_events=census([row["key"]]),
            remote_readback=fail,
            max_file_bytes=100,
            retained_remote_objects=[row],
        )
    assert not result["complete"]
    with pytest.raises(CatalogueUnavailable):
        reconcile.import_inventory("bad", remote_readback=lambda *a: (), activate=True)


def test_progress_serialization_scales_linearly_and_final_is_complete(
    temp_data_dir, monkeypatch
):
    original = reconcile._json
    totals = []
    for count in (40, 80):
        keys = [f"patients/legacy/meta-{i}.json" for i in range(count)]
        reconcile.build_inventory(
            str(count),
            remote_events=census(keys),
            remote_readback=lambda *a: iter([b"{}"]),
            max_file_bytes=100,
        )
        written = []

        def measure(value):
            raw = original(value)
            if (
                isinstance(value, dict)
                and "inventoryId" in value
                and ("outcomes" in value or "delta" in value)
            ):
                written.append(len(raw))
            return raw

        with monkeypatch.context() as m:
            m.setattr(reconcile, "_json", measure)
            reconcile.import_inventory(
                str(count), remote_readback=lambda *a: iter([b"{}"]), activate=False
            )
        totals.append(sum(written))
        progress = json.loads(
            (reconcile._inventory_root(str(count)) / "progress.json").read_bytes()
        )
        assert len(progress["outcomes"]) == count and progress["errors"] == []
    assert totals[1] < totals[0] * 2.8, totals


@pytest.mark.parametrize("boundary", ["before_journal", "after_journal"])
def test_process_death_preserves_linear_journal_and_replay_revalidates_sources(
    temp_data_dir, boundary
):
    import os
    import subprocess
    import sys

    rows = [
        dict(evidence("legacy-video-library"), key=f"videos/{i}.mp4") for i in range(5)
    ]
    build("killed", rows)
    code = """
import os, signal, sys
from backend import storage, clinic_reconciliation as r
storage.init_db()
serialize=r._json
sync=os.fsync
armed=False
def encoded(value):
 global armed
 if isinstance(value,dict) and value.get('sequence')==3 and 'delta' in value:
  if sys.argv[1]=='before_journal': os.kill(os.getpid(),signal.SIGKILL)
  armed=True
 return serialize(value)
def synced(fd):
 sync(fd)
 if armed: os.kill(os.getpid(),signal.SIGKILL)
r._json=encoded
os.fsync=synced
r.import_inventory('killed',remote_readback=lambda *a:iter([b'original']),activate=True)
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
    root = reconcile._inventory_root("killed")
    progress = json.loads((root / "progress.json").read_bytes())
    journal = root / progress["journalFile"]
    events = [json.loads(line) for line in journal.read_bytes().splitlines()]
    assert events[0]["rowsSha256"]
    assert events[-1]["sequence"] == (2 if boundary == "before_journal" else 3)
    assert progress["journalSequence"] == 2
    original_journal = journal.read_bytes()
    result = reconcile.import_inventory(
        "killed", remote_readback=lambda *a: iter([b"original"]), activate=True
    )
    assert len(result["outcomes"]) == 5
    assert journal.read_bytes() == original_journal
    assert (
        json.loads((root / "progress.json").read_bytes())["journalFile"] != journal.name
    )
    with pytest.raises(CatalogueUnavailable):
        reconcile.import_inventory(
            "killed", remote_readback=lambda *a: iter([b"replaced"]), activate=True
        )


def test_unfiled_legacy_record_is_retained_without_patient_or_paid_intent(
    temp_data_dir,
):
    from backend.clinic_records import ClinicLegacyUpload, ClinicUpload

    row = evidence("legacy-unfiled-upload")
    legacy = dict(
        uploadId=row["metadata"]["id"],
        status="needs_reconciliation",
        originalMetadata=row["metadata"],
    )
    reconcile.build_inventory(
        "legacy-upload",
        remote_events=census([row["key"]]),
        remote_readback=lambda *a: iter([b"original"]),
        max_file_bytes=100,
        retained_remote_objects=[row],
        legacy_upload_records=[legacy],
    )
    reconcile.import_inventory(
        "legacy-upload", remote_readback=lambda *a: iter([b"original"]), activate=True
    )
    with storage.session_scope() as s:
        assert (
            json.loads(s.get(ClinicLegacyUpload, legacy["uploadId"]).evidence_json)
            == legacy
        )
        assert s.scalar(select(func.count()).select_from(ClinicUpload)) == 0
        assert s.scalar(select(func.count()).select_from(storage.Patient)) == 0
        assert s.scalar(select(func.count()).select_from(ClinicArtifact)) == 0
