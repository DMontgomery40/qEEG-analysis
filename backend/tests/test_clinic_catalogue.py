from concurrent.futures import ThreadPoolExecutor
import hashlib

import pytest
from sqlalchemy.orm import Session

from backend import storage
from backend import clinic_catalogue as catalogue
from backend import clinic_catalogue_reads as reads
from backend.clinic_naming import canonical_filename


@pytest.fixture
def chart(temp_data_dir):
    with storage.session_scope() as session:
        patient = storage.create_patient(
            session, label="ZZ_01-01-1900_12", first_name="Zoe", last_name="Zero"
        )
    return patient


def register(chart, root, source="source-1", data=b"one", **kwargs):
    path = root / (source + ".pdf")
    path.write_bytes(data)
    return catalogue.register_artifact(
        patient_uuid=chart.id,
        source_kind="test",
        source_id=source,
        original_name="report.pdf",
        logical_family="report",
        local_path=path,
        **kwargs,
    )


@pytest.mark.parametrize(
    ("source", "tail"),
    [
        ("report__09-01-2026.pdf", "report__09-01-2026.pdf"),
        ("report__09-04-2026.pdf", "report__09-04-2026.pdf"),
        (
            "folder/01-01-1983-0__report__09-04-2026.pdf",
            "folder__report__09-04-2026.pdf",
        ),
        ("DM_01-01-1983__09-04-2026.pdf", "09-04-2026.pdf"),
        ("DM_01-01-1983__DM_01-01-1983__09-05-2026.pdf", "09-05-2026.pdf"),
        ("sessions/09-04-2026/report.pdf", "sessions__09-04-2026__report.pdf"),
        ("ZZ_02-02-1990_2__rapport été.pdf", "rapport été.pdf"),
    ],
)
def test_names_preserve_meaning(source, tail):
    assert canonical_filename(source, "ZZ_01-01-1900_12") == "ZZ_01-01-1900_12__" + tail


@pytest.mark.parametrize("ordinal", ["", "_2", "_12", "_999"])
def test_name_bound_and_identity(ordinal):
    patient_id = "ZZ_01-01-1900" + ordinal
    value = canonical_filename(
        "DM_01-01-1983__" + "été" * 400 + "__v3__20260904T120000Z.pdf", patient_id
    )
    assert value.startswith(patient_id + "__") and value.count(patient_id) == 1
    assert len(value.encode()) <= 240 and value.endswith("__v3__20260904T120000Z.pdf")


def test_concurrent_registration_replay_and_distinct_history(chart, temp_data_dir):
    path = temp_data_dir / "same.pdf"
    path.write_bytes(b"one")
    args = dict(
        patient_uuid=chart.id,
        source_kind="test",
        source_id="1",
        original_name="report.pdf",
        logical_family="report",
        local_path=path,
    )
    with ThreadPoolExecutor(max_workers=6) as pool:
        results = list(
            pool.map(lambda _: catalogue.register_artifact(**args), range(12))
        )
    assert len({r["fileId"] for r in results}) == 1
    revision = reads.current_revision()
    assert catalogue.register_artifact(**args)["version"] == 1
    assert reads.current_revision() == revision
    second = catalogue.register_artifact(**{**args, "source_id": "2"})
    assert second["fileId"] != results[0]["fileId"] and second["version"] == 2
    path.write_bytes(b"two")
    with pytest.raises(catalogue.CatalogueConflict):
        catalogue.register_artifact(**args)
    third = catalogue.register_artifact(**{**args, "source_id": "3"})
    assert third["version"] == 3
    with pytest.raises(catalogue.CatalogueUnavailable):
        reads.open_local_file(results[0]["fileId"])


def test_old_producers_pending_recovery_and_revision(chart, temp_data_dir):
    path = temp_data_dir / "later.pdf"
    with storage.session_scope() as session:
        report = storage.create_report(
            session,
            patient_id=chart.id,
            filename="scan.pdf",
            mime_type="application/pdf",
            stored_path=path,
            extracted_text_path=path.with_suffix(".txt"),
        )
    assert reads.roster()["patients"][0]["patientId"] == chart.label
    with pytest.raises(catalogue.CatalogueUnavailable):
        reads.patient_files(chart.label)
    path.write_bytes(b"report")
    file = catalogue.resolve_projection("report", report.id)
    revision = reads.current_revision()
    assert catalogue.resolve_projection("report", report.id)["fileId"] == file["fileId"]
    assert reads.current_revision() == revision
    assert len(reads.patient_files(chart.label)["files"]) == 1


def test_remote_receipt_requires_bytes_and_relabel_alias(chart, temp_data_dir):
    file = register(chart, temp_data_dir)
    key = f"patients/{chart.label}/files/historical.pdf"
    catalogue.add_remote_location(file["fileId"], key)
    assert not reads.file_binding(chart.label, file_key="historical.pdf")["locations"][
        1
    ]["verified"]
    with pytest.raises(catalogue.CatalogueConflict):
        catalogue.verify_remote_location(file["fileId"], key, lambda: iter([b"wrong"]))
    catalogue.verify_remote_location(file["fileId"], key, lambda: iter([b"one"]))
    old = chart.label
    with storage.session_scope() as session:
        storage.update_patient(session, chart.id, label="ZZ_01-01-1900_99")
    binding = reads.file_binding(old, file_key="historical.pdf")
    assert binding["patientId"] == "ZZ_01-01-1900_99"
    assert binding["downloadName"].startswith("ZZ_01-01-1900_99__")


def test_paging_includes_technical_and_rejects_mixed_revision(chart, temp_data_dir):
    for i in range(87):
        register(chart, temp_data_dir, str(i), document_kind="council-export")
    page = reads.patient_files(chart.label, mode="archive", page="1", limit=2)
    all_files = list(page["files"])
    cursor = page["nextCursor"]
    while page["nextCursor"]:
        page = reads.patient_files(
            chart.label, mode="archive", cursor=page["nextCursor"], limit=2
        )
        all_files.extend(page["files"])
    assert len({f["fileId"] for f in all_files}) == 87
    assert page["nextCursor"] is None and not page["truncated"]
    register(chart, temp_data_dir, "new")
    with pytest.raises(catalogue.CatalogueConflict):
        reads.patient_files(chart.label, mode="archive", cursor=cursor, limit=2)


def test_stale_local_binding_loses_verification(chart, temp_data_dir):
    artifact = register(chart, temp_data_dir)
    (temp_data_dir / "source-1.pdf").write_bytes(b"new")
    binding = reads.file_binding(chart.label, file_id=artifact["fileId"])
    assert not binding["hashVerified"]
    assert not binding["locations"][0]["verified"]


def test_download_descriptor_freezes_verified_bytes(chart, temp_data_dir):
    artifact = register(chart, temp_data_dir)
    stream = reads.open_local_file(artifact["fileId"])
    (temp_data_dir / "source-1.pdf").write_bytes(b"new")
    try:
        assert stream.read() == b"one"
    finally:
        stream.close()


def test_direct_orm_patient_change_and_failed_flush_rollback(chart, temp_data_dir):
    revision = reads.current_revision()
    with storage.session_scope() as session:
        row = session.get(storage.Patient, chart.id)
        row.notes = "Recorded directly"
        session.commit()
    assert reads.current_revision() == revision + 1
    assert reads.roster(chart.label)["patient"]["notes"] == "Recorded directly"
    revision = reads.current_revision()
    with storage.session_scope() as session:
        row = session.get(storage.Patient, chart.id)
        row.notes = None  # NOT NULL failure must undo the revision too.
        with pytest.raises(Exception):
            session.commit()
        session.rollback()
    assert reads.current_revision() == revision
    assert reads.roster(chart.label)["patient"]["notes"] == "Recorded directly"


def test_populated_import_requires_source_and_legacy_accounting(chart, temp_data_dir):
    with storage.session_scope() as session:
        from backend.clinic_models import ClinicCatalogState

        session.execute(ClinicCatalogState.__table__.delete())
        session.commit()
    catalogue.initialize_catalogue()
    assert reads.roster()["patients"]
    with pytest.raises(catalogue.CatalogueUnavailable):
        reads.patient_files(chart.label)
    with pytest.raises(catalogue.CatalogueConflict):
        catalogue.complete_catalogue_import(
            {"inventoryId": "census", "legacyPatientIds": ["missing"]}
        )
    catalogue.complete_catalogue_import(
        {"inventoryId": "census", "legacyPatientIds": []}
    )
    assert reads.patient_files(chart.label)["files"] == []
    revision = reads.current_revision()
    catalogue.complete_catalogue_import(
        {"inventoryId": "census", "legacyPatientIds": []}
    )
    assert reads.current_revision() == revision


def test_remote_only_and_alias_import(chart, temp_data_dir):
    digest = hashlib.sha256(b"remote").hexdigest()
    artifact = catalogue.register_artifact(
        patient_uuid=chart.id,
        source_kind="legacy-remote",
        source_id="old-key",
        original_name="patient-facing.pdf",
        logical_family="legacy-report",
        sha256=digest,
        size=6,
        provenance={"history": "unknown"},
        file_key="old-key",
    )
    assert artifact["generatedAt"] is None and artifact["documentKind"] is None
    catalogue.register_patient_alias(chart.id, "01-01-1900-0")
    key = "patients/01-01-1900-0/files/old-key"
    catalogue.add_remote_location(artifact["fileId"], key)
    assert not reads.file_binding("01-01-1900-0", file_key="old-key")["hashVerified"]
    catalogue.verify_remote_location(artifact["fileId"], key, lambda: iter([b"remote"]))
    revision = reads.current_revision()
    catalogue.verify_remote_location(artifact["fileId"], key, lambda: iter([b"remote"]))
    assert reads.current_revision() == revision
    assert (
        reads.file_binding("01-01-1900-0", file_key="old-key")["patientId"]
        == chart.label
    )
    with pytest.raises(catalogue.CatalogueUnavailable):
        reads.open_local_file(artifact["fileId"])


def test_source_error_recovery_does_not_inflate_version(
    chart, temp_data_dir, monkeypatch
):
    path = temp_data_dir / "blocked.pdf"
    path.write_bytes(b"good")
    original = catalogue._read_local
    monkeypatch.setattr(
        catalogue,
        "_read_local",
        lambda _: (_ for _ in ()).throw(
            catalogue.CatalogueUnavailable("source read failure")
        ),
    )
    with storage.session_scope() as session:
        file = storage.create_patient_file(
            session,
            patient_id=chart.id,
            filename="blocked.pdf",
            mime_type="application/pdf",
            size_bytes=4,
            stored_path=path,
        )
    with pytest.raises(catalogue.CatalogueUnavailable):
        catalogue.resolve_projection("patient-file", file.id)
    monkeypatch.setattr(catalogue, "_read_local", original)
    resolved = catalogue.resolve_projection("patient-file", file.id)
    revision = reads.current_revision()
    assert resolved["version"] == 1
    assert (
        catalogue.resolve_projection("patient-file", file.id)["fileId"]
        == resolved["fileId"]
    )
    assert reads.current_revision() == revision


def test_original_generation_metadata_outranks_filename_and_import_time(
    chart, temp_data_dir
):
    older = register(
        chart,
        temp_data_dir,
        "old",
        document_kind="video",
        generated_at=1000,
        uploaded_at=99999,
        provenance={"operationId": "original-op"},
    )
    newer = register(
        chart,
        temp_data_dir,
        "new",
        document_kind="video",
        generated_at=2000,
        uploaded_at=2000,
        provenance={"operationId": "new-op"},
    )
    rows = reads.patient_files(chart.label, mode="initial")["files"]
    assert rows[0]["fileId"] == newer["fileId"]
    assert (
        next(f for f in rows if f["fileId"] == older["fileId"])["provenance"][
            "operationId"
        ]
        == "original-op"
    )
    assert older["uploadedAt"] == 99999 and older["generatedAt"] == 1000
    with pytest.raises(catalogue.CatalogueConflict):
        register(chart, temp_data_dir, "old", document_kind="video", generated_at=2000)


def test_delivery_lookup_unchanged_and_dates(chart, temp_data_dir):
    file = register(
        chart,
        temp_data_dir,
        session_date="2026-09-04",
        provenance={"relativePath": "nested/report.pdf"},
    )
    full = reads.patient_files(chart.label)
    unchanged = reads.patient_files(chart.label, if_index_version=full["indexVersion"])
    assert (
        unchanged["unchanged"]
        and unchanged["files"] == []
        and unchanged["totalFiles"] == 1
    )
    lookup = reads.patient_files(
        chart.label,
        mode="delivery",
        relative_path="nested/report.pdf",
        sha256=file["sha256"],
    )
    assert lookup["files"][0]["fileId"] == file["fileId"]
    assert (
        reads.report_dates([chart.label])["patientReportDates"][chart.label]
        == "2026-09-04"
    )
    for path in ("../report.pdf", "/etc/passwd", "bad\\report.pdf", "x:y"):
        with pytest.raises(ValueError):
            reads.patient_files(
                chart.label, mode="delivery", relative_path=path, sha256=file["sha256"]
            )


def test_legacy_rows_accounted_without_invented_identity(temp_data_dir):
    with storage.session_scope() as session:
        legacy = storage.create_patient(session, label="01-01-1900-0")
    file = register(legacy, temp_data_dir)
    assert file["patientId"] is None and file["downloadName"] is None
    assert reads.roster()["patients"] == []


def test_real_processes_register_one_source_once(temp_data_dir, monkeypatch):
    import os
    import subprocess
    import sys

    root = temp_data_dir / "process-root"
    data = root / "data"
    data.mkdir(parents=True)
    monkeypatch.setattr(storage, "DATA_DIR", data)
    storage.reset_engine(f'sqlite:///{data / "app.db"}')
    storage.init_db()
    with storage.session_scope() as session:
        patient = storage.create_patient(session, label="ZZ_01-01-1900")
    (data / "bytes.pdf").write_bytes(b"shared process bytes")
    env = {**os.environ, "DATA_DIR": str(data), "QEEG_ANALYSIS_ROOT": str(root)}
    script = """
import sys
from backend import storage, clinic_catalogue as c
storage.init_db()
f = c.register_artifact(patient_uuid=sys.argv[1], source_kind='process', source_id='same',
    original_name='bytes.pdf', logical_family='source', local_path=storage.DATA_DIR/'bytes.pdf')
print(f['fileId'], f['version'])
"""
    processes = [
        subprocess.Popen(
            [sys.executable, "-c", script, patient.id],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        for _ in range(5)
    ]
    outputs = []
    for process in processes:
        out, err = process.communicate(timeout=30)
        assert process.returncode == 0, err
        outputs.append(out.strip())
    assert len(set(outputs)) == 1 and outputs[0].endswith(" 1")
    assert reads.current_revision() == 2


def test_noop_patient_flush_does_not_advance_catalogue(chart):
    revision = reads.current_revision()
    with storage.session_scope() as session:
        patient = session.get(storage.Patient, chart.id)
        patient.notes = patient.notes
        session.commit()
    assert reads.current_revision() == revision


def test_paged_reads_project_only_requested_rows(chart, temp_data_dir, monkeypatch):
    for i in range(6):
        register(chart, temp_data_dir, str(i))
    projected = []
    real = reads._artifact_json

    def track(session, artifact, patient):
        projected.append(artifact.id)
        return real(session, artifact, patient)

    monkeypatch.setattr(reads, "_artifact_json", track)
    result = reads.patient_files(chart.label, mode="archive", page="1", limit=2)
    assert result["totalFiles"] == 6 and len(projected) == 2


def test_corrupt_authority_is_unavailable(chart, temp_data_dir):
    file = register(chart, temp_data_dir)
    from backend.clinic_models import ClinicArtifact

    with storage.session_scope() as session:
        artifact = session.get(ClinicArtifact, file["fileId"])
        artifact.provenance_json = "{broken"
        session.commit()
    with pytest.raises(catalogue.CatalogueUnavailable):
        reads.patient_files(chart.label)


def test_unrelated_chart_change_does_not_invalidate_archive(chart, temp_data_dir):
    for i in range(3):
        register(chart, temp_data_dir, str(i))
    first = reads.patient_files(chart.label, mode="archive", page="1", limit=1)
    with storage.session_scope() as session:
        other = storage.create_patient(session, label="AB_02-02-1900")
    register(other, temp_data_dir, "other")
    continued = reads.patient_files(
        chart.label, mode="archive", cursor=first["nextCursor"], limit=1
    )
    assert continued["indexVersion"] == first["indexVersion"]
    assert continued["catalogRevision"] > first["catalogRevision"]
    register(chart, temp_data_dir, "own-change")
    with pytest.raises(catalogue.CatalogueConflict):
        reads.patient_files(
            chart.label, mode="archive", cursor=continued["nextCursor"], limit=1
        )


def test_failed_remote_readback_retires_stale_verification(chart, temp_data_dir):
    file = register(chart, temp_data_dir)
    key = f"patients/{chart.label}/files/replica.pdf"
    catalogue.add_remote_location(file["fileId"], key)
    catalogue.verify_remote_location(file["fileId"], key, lambda: iter([b"one"]))
    before = reads.patient_files(chart.label)["indexVersion"]
    with pytest.raises(catalogue.CatalogueConflict):
        catalogue.verify_remote_location(
            file["fileId"], key, lambda: iter([b"changed"])
        )
    binding = reads.file_binding(chart.label, file_id=file["fileId"])
    assert not next(
        location for location in binding["locations"] if location["kind"] == "netlify"
    )["verified"]
    assert reads.patient_files(chart.label)["indexVersion"] != before


def test_concurrent_distinct_sources_allocate_each_version_once(chart, temp_data_dir):
    path = temp_data_dir / "shared.pdf"
    path.write_bytes(b"shared")

    def admit(index):
        return catalogue.register_artifact(
            patient_uuid=chart.id,
            source_kind="parallel",
            source_id=str(index),
            original_name="same.pdf",
            logical_family="same",
            local_path=path,
        )

    with ThreadPoolExecutor(max_workers=6) as pool:
        results = list(pool.map(admit, range(12)))
    assert sorted(result["version"] for result in results) == list(range(1, 13))
    assert len({result["fileId"] for result in results}) == 12


def test_pre_catalogue_database_can_migrate_before_initialization(temp_data_dir):
    path = temp_data_dir / "old-schema.db"
    storage.reset_engine(f"sqlite:///{path}")
    storage.Patient.__table__.create(storage.engine)
    with storage.session_scope() as session:
        patient = storage.create_patient(session, label="old-identity")
        storage.update_patient(session, patient.id, label="ZZ_01-01-1900")
    storage.init_db()
    assert reads.roster()["patients"][0]["patientId"] == "ZZ_01-01-1900"
    with pytest.raises(catalogue.CatalogueUnavailable):
        reads.patient_files("ZZ_01-01-1900")
    before = reads.current_revision()
    with storage.session_scope() as session:
        storage.update_patient(
            session, patient.id, label="ZZ_01-01-1900", notes="after initialization"
        )
    assert reads.current_revision() == before + 1


def test_orphan_source_is_preserved_as_pending_not_a_new_patient(temp_data_dir):
    from backend.clinic_models import ClinicProjection

    path = temp_data_dir / "orphan.pdf"
    path.write_bytes(b"preserved")
    with storage.session_scope() as session:
        report = storage.create_report(
            session,
            patient_id="missing-original-row",
            filename="orphan.pdf",
            mime_type="application/pdf",
            stored_path=path,
            extracted_text_path=path.with_suffix(".txt"),
        )
        pending = session.query(ClinicProjection).filter_by(source_id=report.id).one()
        assert (
            pending.patient_uuid == "missing-original-row"
            and pending.artifact_id is None
        )
    assert reads.roster()["patients"] == []
    with pytest.raises(catalogue.CatalogueUnavailable):
        catalogue.complete_catalogue_import(
            {"inventoryId": "cannot-claim-complete", "legacyPatientIds": []}
        )


def test_dispose_reset_and_failed_initialization_do_not_activate_wrong_engine(
    temp_data_dir, monkeypatch
):
    from backend.clinic_models import ClinicCatalogState

    original = storage.engine
    assert original in catalogue._initialized_engines
    original.dispose()
    with storage.session_scope() as session:
        storage.create_patient(session, label="ZZ_01-01-1900")
    assert reads.current_revision() == 1
    storage.reset_engine(f'sqlite:///{temp_data_dir / "fresh-engine.db"}')
    replacement = storage.engine
    assert replacement not in catalogue._initialized_engines
    real_get = Session.get

    def fail_state(self, entity, *args, **kwargs):
        if entity is ClinicCatalogState:
            raise RuntimeError("initialization failed")
        return real_get(self, entity, *args, **kwargs)

    monkeypatch.setattr(Session, "get", fail_state)
    with pytest.raises(RuntimeError):
        storage.init_db()
    assert replacement not in catalogue._initialized_engines
    monkeypatch.setattr(Session, "get", real_get)
    storage.init_db()
    assert (
        replacement in catalogue._initialized_engines and reads.current_revision() == 0
    )


@pytest.mark.parametrize("extension", [".pdf", ".abcdefgh"])
def test_impossible_identity_filename_budget_rejects_without_hanging(extension):
    import subprocess
    import sys

    # The parser accepts this shape; the real allocator cannot issue it. Exercise
    # corrupt/imported history under an OS timeout so regressions cannot hang CI.
    script = """
import sys
from backend.clinic_naming import canonical_filename
try:
    canonical_filename('description' + sys.argv[1], 'ZZ_01-01-1900_' + '9' * 240)
except ValueError:
    pass
else:
    raise AssertionError('Impossible identity budget was accepted')
"""
    result = subprocess.run(
        [sys.executable, "-c", script, extension],
        capture_output=True,
        text=True,
        timeout=3,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("description", ["é" * 300, "🧠" * 150, "a" * 500])
@pytest.mark.parametrize("extension", [".pdf", ".abcdefgh"])
def test_multibyte_filename_boundary_preserves_extension_and_revision(
    description, extension
):
    name = canonical_filename(
        description + "__v12__20260904T120000Z" + extension, "ZZ_01-01-1900_999"
    )
    assert len(name.encode("utf-8")) <= 240
    assert name.endswith("__v12__20260904T120000Z" + extension)
    assert name.startswith("ZZ_01-01-1900_999__")


def test_legacy_duplicate_rows_and_alias_ambiguity_survive_relabels(temp_data_dir):
    alias = "ZZ_01-01-1900"
    with storage.session_scope() as session:
        first = storage.create_patient(session, label=alias)
        second = storage.create_patient(session, label=alias)
    with pytest.raises(catalogue.CatalogueConflict):
        reads.roster()
    with pytest.raises(catalogue.CatalogueConflict):
        catalogue.complete_catalogue_import(
            {"inventoryId": "duplicates", "legacyPatientIds": []}
        )
    with storage.session_scope() as session:
        storage.update_patient(session, first.id, label="ZZ_01-01-1900_2")
        storage.update_patient(session, second.id, label="ZZ_01-01-1900_3")
    assert len(reads.roster()["patients"]) == 2
    with pytest.raises(catalogue.CatalogueConflict):
        reads.roster(alias)
    with pytest.raises(catalogue.CatalogueConflict):
        catalogue.register_patient_alias(first.id, alias)
    assert reads.roster("ZZ_01-01-1900_3")["patient"]["patientId"] == "ZZ_01-01-1900_3"


def test_exact_historical_files_keep_original_chart_through_duplicate_relabels(
    temp_data_dir,
):
    old = "ZZ_01-01-1900"
    with storage.session_scope() as session:
        first = storage.create_patient(session, label=old)
    file1 = register(first, temp_data_dir, "first")
    catalogue.add_remote_location(file1["fileId"], f"patients/{old}/files/first.pdf")
    with storage.session_scope() as session:
        second = storage.create_patient(session, label=old)
        healthy = storage.create_patient(session, label="AB_02-02-1900")
    file2 = register(second, temp_data_dir, "second")
    catalogue.add_remote_location(file2["fileId"], f"patients/{old}/files/second.pdf")
    with storage.session_scope() as session:
        storage.update_patient(session, first.id, label="ZZ_01-01-1900_2")
        storage.update_patient(session, second.id, label="ZZ_01-01-1900_3")
    assert (
        reads.file_binding(old, file_key="first.pdf")["patientId"] == "ZZ_01-01-1900_2"
    )
    assert (
        reads.file_binding(old, file_key="second.pdf")["patientId"] == "ZZ_01-01-1900_3"
    )
    assert reads.file_binding(old, file_id=file2["fileId"])["fileId"] == file2["fileId"]
    assert reads.patient_files(healthy.label)["files"] == []
    other = register(healthy, temp_data_dir, "unrelated")
    with pytest.raises(catalogue.CatalogueNotFound):
        reads.file_binding(old, file_id=other["fileId"])
