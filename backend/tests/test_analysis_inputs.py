"""Complete input admission uses real SQLite/assets and fake provider discovery."""

from __future__ import annotations

import json
import uuid
from pathlib import Path

import fitz
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select

from backend import storage
from backend.tests.test_main_invariants import _test_app


def source(tmp_path, patient_id, *, date="2026-01-02", pages=1, value=280):
    rid = str(uuid.uuid4())
    directory = tmp_path / "uploads" / str(uuid.uuid4())  # deliberately != Report.id
    directory.mkdir(parents=True)
    doc = fitz.open()
    text = f"Session 1 ({date})\nPhysical Reaction Time {value} ms 250-360 ms\nAudio P300 Delay 300 ms 250-330 ms"
    for n in range(pages):
        page = doc.new_page()
        page.insert_text((50, 50), text if n == 0 else f"Coherence page {n+1}")
    doc.save(directory / "original.pdf")
    (directory / "pages").mkdir()
    (directory / "sources").mkdir()
    sections = []
    for n, page in enumerate(doc, 1):
        body = page.get_text()
        sections.append(f"=== PAGE {n} / {pages} ===\n{body}")
        page.get_pixmap().save(directory / "pages" / f"page-{n}.png")
        for engine in ["pypdf", "pymupdf", "apple_vision", "tesseract"]:
            (directory / "sources" / f"page-{n}.{engine}.txt").write_text(body)
    doc.close()
    for filename in ["extracted.txt", "extracted_enhanced.txt"]:
        (directory / filename).write_text("\n\n".join(sections))
    (directory / "metadata.json").write_text(
        json.dumps(
            {
                "schema_version": 2,
                "page_count": pages,
                "engines": {
                    "pypdf": True,
                    "pymupdf": True,
                    "apple_vision": True,
                    "tesseract": True,
                },
                "pages": [{"page": n} for n in range(1, pages + 1)],
            }
        )
    )
    with storage.session_scope() as s:
        return storage.create_report(
            s,
            report_id=rid,
            patient_id=patient_id,
            filename="report.pdf",
            mime_type="application/pdf",
            stored_path=directory / "original.pdf",
            extracted_text_path=directory / "extracted.txt",
        )


@pytest.fixture
def admission(temp_data_dir, monkeypatch):
    app, main = _test_app(temp_data_dir, monkeypatch)
    with storage.session_scope() as s:
        p = storage.create_patient(s, label="ZZ_01-01-1900")
    payload = {
        "patient_id": p.id,
        "council_model_ids": ["mock-council-a"],
        "consolidator_model_id": "mock-consolidator",
    }
    with TestClient(app, raise_server_exceptions=False) as client:
        yield client, payload, temp_data_dir, main


@pytest.mark.parametrize("count", [1, 2, 3])
def test_complete_source_admission_and_operation_retry(admission, count):
    client, payload, tmp, _ = admission
    sources = [
        source(tmp, payload["patient_id"], date=f"2026-0{count-i}-02", pages=6)
        for i in range(count)
    ]
    ids = [s.id for s in sources]
    req = {
        **payload,
        "report_ids": ids,
        "special_instructions": "  Address work fatigue.\nKeep café context.  \n",
        "operation_id": "lost-response",
    }
    first = client.post("/api/runs", json=req)
    assert first.status_code == 200, first.text
    run = first.json()
    assert run["source_report_ids"] == ids
    assert run["special_instructions"] == req["special_instructions"]
    assert len(run["analysis_input_fingerprint"]) == 64
    assert client.post("/api/runs", json=req).json()["id"] == run["id"]
    with storage.session_scope() as s:
        execution = storage.get_report(s, run["report_id"])
        assert len(list(s.scalars(select(storage.Run)))) == 1
    if count == 1:
        assert execution.id == ids[0]
    else:
        assert execution.id not in ids
        directory = Path(execution.stored_path).parent
        assert len(list((directory / "pages").glob("*.png"))) == count * 6
        assert len(list((directory / "sources").glob("*.txt"))) == count * 6 * 4
        with fitz.open(execution.stored_path) as pdf:
            assert len(pdf) == count * 6
        manifest = run["source_manifest"]
        assert [x["report_id"] for x in manifest["sources"]] == ids
        assert manifest["sources"][0]["session_aliases"] == {"1": count}
        assert len(manifest["page_map"]) == count * 6
        assert [p["source_report_id"] for p in manifest["page_map"][::6]] == ids
    for changed in [
        {"special_instructions": "changed"},
        {"report_ids": ids[::-1] if count > 1 else []},
        {"council_model_ids": ["mock-council-b"]},
    ]:
        response = client.post("/api/runs", json={**req, **changed})
        assert response.status_code in (400, 409)
    new = client.post("/api/runs", json={**req, "operation_id": "intentional-new"})
    assert new.status_code == 200
    assert new.json()["id"] != run["id"]


@pytest.mark.parametrize(
    "variant",
    [
        "legacy",
        "equivalent",
        "empty",
        "duplicate",
        "contradictory",
        "missing",
        "foreign",
        "missing-original",
    ],
)
def test_source_validation(admission, variant):
    client, payload, tmp, _ = admission
    report = source(tmp, payload["patient_id"])
    request = {**payload, "report_ids": [report.id]}
    expected = 400
    if variant == "legacy":
        request.pop("report_ids")
        request["report_id"] = report.id
        expected = 200
    elif variant == "equivalent":
        request["report_id"] = report.id
        expected = 200
    elif variant == "empty":
        request["report_ids"] = []
    elif variant == "duplicate":
        request["report_ids"] *= 2
    elif variant == "contradictory":
        request["report_id"] = "other"
    elif variant == "missing":
        request["report_ids"] += ["missing"]
        expected = 404
    elif variant == "missing-original":
        Path(report.stored_path).unlink()
    elif variant == "foreign":
        with storage.session_scope() as s:
            other = storage.create_patient(s, label="AB_01-01-1900")
        request["report_ids"] += [source(tmp, other.id).id]
    response = client.post("/api/runs", json=request)
    assert response.status_code == expected, response.text


@pytest.mark.parametrize("date", ["unknown", "2026-02-30", "01/02/2026"])
def test_mapping_ambiguity_has_source_evidence(admission, date):
    client, payload, tmp, _ = admission
    a = source(tmp, payload["patient_id"])
    b = source(tmp, payload["patient_id"], date=date, value=299)
    response = client.post("/api/runs", json={**payload, "report_ids": [a.id, b.id]})
    assert response.status_code == 409, response.text
    detail = response.json()["detail"]
    assert detail["code"] == "ANALYSIS_SESSION_MAPPING_REQUIRED"
    assert {s["report_id"] for s in detail["sources"]} == {a.id, b.id}
    assert detail["reason"]


def test_same_date_consistent_repeated_visit_is_merged(admission):
    client, payload, tmp, _ = admission
    reports = [
        source(tmp, payload["patient_id"], date=date)
        for date in ["01/02/2026", "2026-01-02"]
    ]
    response = client.post(
        "/api/runs", json={**payload, "report_ids": [r.id for r in reports]}
    )
    assert response.status_code == 200, response.text
    assert [
        s["session_aliases"] for s in response.json()["source_manifest"]["sources"]
    ] == [{"1": 1}, {"1": 1}]


@pytest.mark.parametrize("breakpoint", ["compose", "promotion", "registration", "run"])
def test_interrupted_admission_recovers_reserved_identity(
    admission, monkeypatch, breakpoint
):
    from backend import analysis_inputs as inputs

    client, payload, tmp, _ = admission
    originals = [
        source(tmp, payload["patient_id"], date=d) for d in ["2026-01-02", "2026-02-03"]
    ]
    req = {
        **payload,
        "report_ids": [r.id for r in originals],
        "operation_id": "interrupted",
    }
    if breakpoint == "compose":
        owner, name = inputs, "_write_combined_report"
    elif breakpoint == "promotion":
        owner, name = inputs.os, "replace"
    elif breakpoint == "registration":
        owner, name = storage, "create_report"
    else:
        owner, name = storage, "create_run"
    saved = getattr(owner, name)

    def fail(*args, **kwargs):
        raise RuntimeError("simulated process exit")

    monkeypatch.setattr(owner, name, fail)
    response = client.post("/api/runs", json=req)
    assert response.status_code == 500
    with storage.session_scope() as s:
        reservation = s.get(storage.AnalysisInputReservation, "interrupted")
        assert reservation is not None
        assert not list(s.scalars(select(storage.Run)))
    monkeypatch.setattr(owner, name, saved)
    response = client.post("/api/runs", json=req)
    assert response.status_code == 200, response.text
    assert response.json()["id"] == reservation.run_id
    assert response.json()["report_id"] == reservation.report_id
    with storage.session_scope() as s:
        assert len(list(s.scalars(select(storage.Run)))) == 1
        assert len(list(s.scalars(select(storage.Report)))) == 3


@pytest.mark.parametrize("damage", ["page", "metadata", "text", "directory"])
def test_combined_asset_recovery_uses_original_sources(admission, monkeypatch, damage):
    from backend import analysis_inputs as inputs

    client, payload, tmp, _ = admission
    originals = [
        source(tmp, payload["patient_id"], date=d, pages=2)
        for d in ["2026-02-02", "2026-01-02"]
    ]
    original_bytes = [Path(r.stored_path).read_bytes() for r in originals]
    req = {**payload, "report_ids": [r.id for r in originals], "operation_id": "repair"}
    created = client.post("/api/runs", json=req).json()
    with storage.session_scope() as s:
        report = storage.get_report(s, created["report_id"])
    directory = Path(report.stored_path).parent
    before = inputs._asset_inventory(directory)
    if damage == "directory":
        import shutil

        shutil.rmtree(directory)
    else:
        (
            directory
            / {
                "page": "pages/page-3.png",
                "metadata": "metadata.json",
                "text": "extracted_enhanced.txt",
            }[damage]
        ).unlink()

    def forbid(*args, **kwargs):
        pytest.fail(
            "Generic OCR must not be called for combined or already extracted sources"
        )

    monkeypatch.setattr(inputs.reports, "extract_pdf_full", forbid)
    assert inputs.repair_combined_report(report, run_id=created["id"])
    assert inputs._asset_inventory(directory) == before
    assert [Path(r.stored_path).read_bytes() for r in originals] == original_bytes


@pytest.mark.parametrize(
    "change",
    ["order", "original", "extraction", "page", "alias", "instructions", "models"],
)
@pytest.mark.parametrize("reserved_only", [False, True])
def test_operation_rejects_every_immutable_input_change(
    admission, monkeypatch, change, reserved_only
):
    client, payload, tmp, _ = admission
    originals = [
        source(tmp, payload["patient_id"], date=d) for d in ["2026-02-02", "2026-01-02"]
    ]
    req = {**payload, "report_ids": [r.id for r in originals], "operation_id": "frozen"}
    create = storage.create_run
    if reserved_only:

        def crash(*args, **kwargs):
            raise RuntimeError("exit before run registration")

        monkeypatch.setattr(storage, "create_run", crash)
    first = client.post("/api/runs", json=req)
    assert first.status_code == (500 if reserved_only else 200)
    monkeypatch.setattr(storage, "create_run", create)
    with storage.session_scope() as session:
        reservation = session.get(storage.AnalysisInputReservation, "frozen")
        saved_manifest = reservation.manifest_json
    if change == "order":
        req["report_ids"].reverse()
    elif change == "instructions":
        req["special_instructions"] = "focus"
    elif change == "models":
        req["council_model_ids"] = ["mock-council-b"]
    elif change == "alias":
        req["source_session_aliases"] = {
            r.id: {"1": i + 7} for i, r in enumerate(originals)
        }
    elif change == "original":
        with Path(originals[0].stored_path).open("ab") as out:
            out.write(b"\nchanged")
    elif change == "extraction":
        with (Path(originals[0].stored_path).parent / "extracted_enhanced.txt").open(
            "a"
        ) as out:
            out.write("\nnew context")
    elif change == "page":
        pixmap = fitz.Pixmap(
            str(Path(originals[0].stored_path).parent / "pages/page-1.png")
        )
        pixmap.clear_with(0)
        pixmap.save(Path(originals[0].stored_path).parent / "pages/page-1.png")
    response = client.post("/api/runs", json=req)
    if not reserved_only and change in ["original", "extraction", "page"]:
        # The receipt survives source drift; starting new generation still validates it.
        assert response.status_code == 200, response.text
        assert response.json() == first.json()
        assert client.post(f"/api/runs/{reservation.run_id}/start").status_code == 409
    else:
        assert response.status_code == 409, response.text
        assert response.json()["detail"]["code"] == "ANALYSIS_OPERATION_CONFLICT"
    with storage.session_scope() as session:
        assert (
            session.get(storage.AnalysisInputReservation, "frozen").manifest_json
            == saved_manifest
        )
        assert bool(storage.get_run(session, reservation.run_id)) is not reserved_only


@pytest.mark.parametrize(
    "aliases",
    ["incomplete", "unknown", "zero", "bool", "float", "conflicting", "distinct"],
)
def test_explicit_session_mapping_validation(admission, aliases):
    client, payload, tmp, _ = admission
    a = source(tmp, payload["patient_id"], value=280)
    b = source(tmp, payload["patient_id"], value=299)
    mapping = {a.id: {"1": 1}, b.id: {"1": 2}}
    if aliases == "incomplete":
        mapping.pop(b.id)
    elif aliases == "unknown":
        mapping["unknown"] = {"1": 3}
    elif aliases == "zero":
        mapping[a.id]["1"] = 0
    elif aliases == "bool":
        mapping[a.id]["1"] = True
    elif aliases == "float":
        mapping[a.id]["1"] = 1.5
    elif aliases == "conflicting":
        mapping[b.id]["1"] = 1
    response = client.post(
        "/api/runs",
        json={**payload, "report_ids": [a.id, b.id], "source_session_aliases": mapping},
    )
    expected = (
        200 if aliases == "distinct" else (422 if aliases in ["bool", "float"] else 409)
    )
    assert response.status_code == expected, response.text


def test_source_drift_blocks_start_before_scheduling(admission, monkeypatch):
    client, payload, tmp, main = admission
    report = source(tmp, payload["patient_id"])
    created = client.post("/api/runs", json={**payload, "report_id": report.id}).json()
    Path(report.extracted_text_path).write_text("changed source snapshot")
    calls = []
    monkeypatch.setattr(main, "_spawn_task", lambda *a, **k: calls.append(True))
    response = client.post("/api/runs/" + created["id"] + "/start")
    assert response.status_code == 409
    assert response.json()["detail"]["code"] == "ANALYSIS_INPUT_CHANGED"
    assert calls == []
    with storage.session_scope() as s:
        assert storage.get_run(s, created["id"]).status == "created"


def test_generic_reextract_endpoint_preserves_combined_identity(admission, monkeypatch):
    client, payload, tmp, main = admission
    reports = [
        source(tmp, payload["patient_id"], date=date)
        for date in ["2026-03-02", "2026-01-02"]
    ]
    run = client.post(
        "/api/runs", json={**payload, "report_ids": [r.id for r in reports]}
    ).json()
    with storage.session_scope() as s:
        execution = storage.get_report(s, run["report_id"])
    directory = Path(execution.stored_path).parent
    (directory / "pages/page-1.png").unlink()
    monkeypatch.setattr(
        main,
        "extract_pdf_full",
        lambda *args: pytest.fail("Generic merged OCR was called"),
    )
    response = client.post("/api/reports/" + execution.id + "/reextract")
    assert response.status_code == 200
    assert response.json()["page_images_written"] == 2
    assert "local=1 global=2" in (directory / "extracted_enhanced.txt").read_text()


@pytest.mark.parametrize("interrupted", [False, True])
def test_admission_repairs_original_extraction_once_before_run_registration(
    admission, monkeypatch, interrupted
):
    from backend import reports
    from backend.reports import PdfFullExtraction

    client, payload, tmp, _ = admission
    original = source(tmp, payload["patient_id"])
    directory = Path(original.stored_path).parent
    saved_text = (directory / "extracted_enhanced.txt").read_text()
    image_bytes = (directory / "pages/page-1.png").read_bytes()
    (directory / "pages/page-1.png").unlink()
    calls = []

    def extract(path):
        import base64

        calls.append(path)
        return PdfFullExtraction(
            enhanced_text=saved_text,
            page_images=[
                {"page": 1, "base64_png": base64.b64encode(image_bytes).decode()}
            ],
            per_page_sources=[
                {
                    "page": 1,
                    "pypdf_text": saved_text,
                    "pymupdf_text": saved_text,
                    "vision_ocr_text": saved_text,
                    "tesseract_ocr_text": saved_text,
                }
            ],
            metadata={
                "schema_version": 2,
                "page_count": 1,
                "pages": [{"page": 1}],
                "engines": {
                    "pypdf": True,
                    "pymupdf": True,
                    "apple_vision": True,
                    "tesseract": True,
                },
            },
        )

    monkeypatch.setattr(reports, "extract_pdf_full", extract)
    request = {**payload, "report_id": original.id, "operation_id": "repaired-original"}
    if interrupted:
        from backend import analysis_inputs as inputs

        replace = inputs.os.replace
        writes = []

        def interrupt_after_partial_repair(src, dst):
            writes.append(str(dst))
            if len(writes) == 5:
                raise RuntimeError("interrupted source extraction repair")
            return replace(src, dst)

        monkeypatch.setattr(inputs.os, "replace", interrupt_after_partial_repair)
        assert client.post("/api/runs", json=request).status_code == 500
        monkeypatch.setattr(inputs.os, "replace", replace)
    first = client.post("/api/runs", json=request)
    assert first.status_code == 200, first.text
    second = client.post("/api/runs", json=request)
    assert second.status_code == 200, second.text
    assert first.json()["id"] == second.json()["id"]
    assert calls == [Path(original.stored_path)] * (2 if interrupted else 1)
    assert (directory / "pages/page-1.png").read_bytes() == image_bytes


def test_legacy_storage_migration_preserves_banked_run(temp_data_dir):
    from sqlalchemy import create_engine

    legacy = create_engine(f"sqlite:///{temp_data_dir}/legacy.db")
    with legacy.begin() as conn:
        conn.exec_driver_sql(
            "CREATE TABLE runs (id VARCHAR PRIMARY KEY, report_id VARCHAR NOT NULL)"
        )
        conn.exec_driver_sql("INSERT INTO runs VALUES ('banked', 'source')")
    storage.reset_engine(f"sqlite:///{temp_data_dir}/legacy.db")
    storage.init_db()
    storage.init_db()
    with storage.engine.begin() as conn:
        row = conn.exec_driver_sql(
            "SELECT source_report_ids_json, source_manifest_json, special_instructions, operation_id FROM runs WHERE id = 'banked'"
        ).one()
    assert json.loads(row[0]) == ["source"]
    assert json.loads(row[1]) == {"legacy": True}
    assert row[2:] == ("", None)


@pytest.mark.parametrize("change", ["missing-source", "different-patient"])
def test_reserved_operation_conflicts_before_repairing_changed_input(admission, change):
    client, payload, tmp, _ = admission
    report = source(tmp, payload["patient_id"])
    req = {**payload, "report_ids": [report.id], "operation_id": "reserved-input"}
    assert client.post("/api/runs", json=req).status_code == 200
    if change == "missing-source":
        req["report_ids"] = ["missing"]
    elif change == "different-patient":
        req["patient_id"] = "another"
    response = client.post("/api/runs", json=req)
    assert response.status_code == 409, response.text
    assert response.json()["detail"]["code"] == "ANALYSIS_OPERATION_CONFLICT"


@pytest.mark.parametrize(
    "legend",
    [
        "Session 1: 2026-02-02",
        "Session 1 Date: 02/02/2026",
        "Session 1 (2026-02-02)\nSession 1 (2026-03-02)",
        "Session 1 (2026-02-02)\nSession 2 (2026-02-02)",
    ],
)
def test_source_date_evidence_resolves_only_unambiguous_visits(admission, legend):
    client, payload, tmp, _ = admission
    a = source(tmp, payload["patient_id"])
    b = source(tmp, payload["patient_id"], date="2026-02-02")
    path = Path(b.stored_path).parent / "extracted_enhanced.txt"
    path.write_text(path.read_text().replace("Session 1 (2026-02-02)", legend))
    response = client.post("/api/runs", json={**payload, "report_ids": [b.id, a.id]})
    if "\n" in legend:
        assert response.status_code == 409, response.text
        assert response.json()["detail"]["code"] == "ANALYSIS_SESSION_MAPPING_REQUIRED"
    else:
        assert response.status_code == 200, response.text
        assert response.json()["source_manifest"]["sources"][0]["session_aliases"] == {
            "1": 2
        }


def test_concurrent_same_operation_creates_one_report_and_run(admission, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    from backend import analysis_inputs as inputs

    client, payload, tmp, _ = admission
    originals = [
        source(tmp, payload["patient_id"], date=date, pages=3)
        for date in ["2026-03-02", "2026-01-02"]
    ]
    req = {
        **payload,
        "report_ids": [r.id for r in originals],
        "operation_id": "concurrent-operation",
    }
    calls = []
    compose = inputs._write_combined_report

    def recording_compose(**kwargs):
        calls.append(kwargs["report_id"])
        return compose(**kwargs)

    monkeypatch.setattr(inputs, "_write_combined_report", recording_compose)
    with ThreadPoolExecutor(max_workers=2) as workers:
        results = list(
            workers.map(lambda _: client.post("/api/runs", json=req), [1, 2])
        )
    assert [r.status_code for r in results] == [200, 200]
    assert results[0].json()["id"] == results[1].json()["id"]
    assert len(calls) == 1
    with storage.session_scope() as session:
        assert len(list(session.scalars(select(storage.Run)))) == 1
        assert len(list(session.scalars(select(storage.Report)))) == 3


@pytest.mark.parametrize("explicit", [False, True])
@pytest.mark.parametrize(
    "left_observation,right_observation,expected",
    [
        ("Alpha power at Cz: 5 uV2", "Alpha power at Cz: 40 uV2", 409),
        ("Coherence F3-F4: 0.20", "Coherence F3-F4: 0.95", 409),
        ("Theta peak at Fz: 4 Hz", "Theta peak at Fz: 7 Hz", 409),
        ("Alpha power at Cz: 5 uV2", "Alpha power at Cz: 5 uV2", 200),
        ("Alpha power at Cz: 5 uV2", "", 409),
    ],
)
def test_same_date_merge_checks_complete_observed_evidence(
    admission, explicit, left_observation, right_observation, expected
):
    client, payload, tmp, _ = admission
    originals = [source(tmp, payload["patient_id"]) for _ in range(2)]
    for original, observation in zip(originals, [left_observation, right_observation]):
        directory = Path(original.stored_path).parent
        for path in [
            directory / "extracted_enhanced.txt",
            *sorted((directory / "sources").glob("*.txt")),
        ]:
            path.write_text(path.read_text() + "\n" + observation)
    request = {**payload, "report_ids": [r.id for r in originals]}
    if explicit:
        request["source_session_aliases"] = {r.id: {"1": 1} for r in originals}
    response = client.post("/api/runs", json=request)
    assert response.status_code == expected, response.text
    if expected == 409:
        assert response.json()["detail"]["code"] == "ANALYSIS_SESSION_MAPPING_REQUIRED"
        with storage.session_scope() as session:
            assert not list(session.scalars(select(storage.Run)))


@pytest.mark.parametrize("stream", ["pypdf", "pymupdf", "apple_vision", "tesseract"])
def test_merge_preserves_conflicting_evidence_from_every_ocr_stream(admission, stream):
    client, payload, tmp, _ = admission
    originals = [source(tmp, payload["patient_id"]) for _ in range(2)]
    for original, value in zip(originals, [5, 40]):
        path = Path(original.stored_path).parent / "sources" / f"page-1.{stream}.txt"
        path.write_text(path.read_text() + f"\nAlpha power at Cz: {value} uV2\n")
    response = client.post(
        "/api/runs", json={**payload, "report_ids": [r.id for r in originals]}
    )
    assert response.status_code == 409, response.text
    assert response.json()["detail"]["code"] == "ANALYSIS_SESSION_MAPPING_REQUIRED"


@pytest.mark.parametrize(
    "dates,labels,expected",
    [
        (["2026-01-02", "2026-02-02"], [2, 1], 409),
        (["2026-01-02", "2026-02-02"], [1, 2], 200),
        (["2026-02-02", "2026-01-02"], [2, 1], 200),
        (["2026-01-02", "2026-01-02"], [2, 1], 200),
        (["unknown", "2026-02-02"], [2, 1], 200),
        (["2026-01-02", "unknown", "2026-02-02"], [3, 2, 1], 409),
        (["2026-01-02", "unknown", "2026-02-02"], [1, 3, 2], 200),
    ],
)
def test_explicit_mapping_preserves_every_known_chronological_relation(
    admission, dates, labels, expected
):
    client, payload, tmp, _ = admission
    originals = [source(tmp, payload["patient_id"], date=date) for date in dates]
    request = {
        **payload,
        "report_ids": [r.id for r in originals],
        "source_session_aliases": {
            r.id: {"1": label} for r, label in zip(originals, labels)
        },
    }
    response = client.post("/api/runs", json=request)
    assert response.status_code == expected, response.text
    if expected == 409:
        assert response.json()["detail"]["code"] == "ANALYSIS_SESSION_MAPPING_REQUIRED"


@pytest.mark.parametrize("explicit", [False, True])
def test_partial_multisession_evidence_needs_operator_mapping(admission, explicit):
    client, payload, tmp, _ = admission
    originals = [source(tmp, payload["patient_id"]) for _ in range(2)]
    for report, dates, values in zip(
        originals,
        [("2026-01-02", "2026-02-02"), ("2026-02-02", "2026-03-02")],
        [(280, 290), (290, 300)],
    ):
        directory = Path(report.stored_path).parent
        body = f"Session 1 ({dates[0]})\nSession 2 ({dates[1]})\nPhysical Reaction Time {values[0]} ms {values[1]} ms 250-360 ms\n"
        (directory / "extracted_enhanced.txt").write_text("=== PAGE 1 / 1 ===\n" + body)
        for path in (directory / "sources").glob("*.txt"):
            path.write_text(body)
    request = {**payload, "report_ids": [r.id for r in originals]}
    if explicit:
        request["source_session_aliases"] = {
            originals[0].id: {"1": 1, "2": 2},
            originals[1].id: {"1": 2, "2": 3},
        }
    response = client.post("/api/runs", json=request)
    assert response.status_code == (200 if explicit else 409), response.text


def test_stricter_admission_preserves_identical_banked_operation_receipt(
    admission, monkeypatch
):
    from backend import analysis_inputs as inputs

    client, payload, tmp, _ = admission
    originals = [source(tmp, payload["patient_id"]) for _ in range(2)]
    for original, value in zip(originals, [5, 40]):
        path = Path(original.stored_path).parent / "extracted_enhanced.txt"
        path.write_text(path.read_text() + f"\nAlpha power at Cz: {value} uV2\n")
    request = {
        **payload,
        "report_ids": [r.id for r in originals],
        "operation_id": "pre-fix-receipt",
    }
    # Model the prior admission version's partial-evidence decision, then restore
    # current validation. The underlying stored source snapshots stay exact.
    observed = inputs._observed_session_evidence
    monkeypatch.setattr(
        inputs,
        "_observed_session_evidence",
        lambda *args: (True, (("prior matching subset",),)),
    )
    first = client.post("/api/runs", json=request)
    assert first.status_code == 200, first.text
    monkeypatch.setattr(inputs, "_observed_session_evidence", observed)
    repeated = client.post("/api/runs", json=request)
    assert repeated.status_code == 200, repeated.text
    assert repeated.json()["id"] == first.json()["id"]
    assert repeated.json()["source_manifest"] == first.json()["source_manifest"]
    new = client.post("/api/runs", json={**request, "operation_id": "new-admission"})
    assert new.status_code == 409
    assert new.json()["detail"]["code"] == "ANALYSIS_SESSION_MAPPING_REQUIRED"


def test_banked_receipt_does_not_ignore_removed_explicit_aliases(admission):
    client, payload, tmp, _ = admission
    original = source(tmp, payload["patient_id"])
    request = {
        **payload,
        "report_ids": [original.id],
        "operation_id": "explicit-original",
        "source_session_aliases": {original.id: {"1": 3}},
    }
    first = client.post("/api/runs", json=request)
    assert first.status_code == 200, first.text
    assert client.post("/api/runs", json=request).json()["id"] == first.json()["id"]
    request.pop("source_session_aliases")
    changed = client.post("/api/runs", json=request)
    assert changed.status_code == 409, changed.text
    assert changed.json()["detail"]["code"] == "ANALYSIS_OPERATION_CONFLICT"


@pytest.mark.parametrize("count", [1, 2])
@pytest.mark.parametrize("status", ["created", "running", "complete"])
def test_exact_rejoin_survives_catalogue_and_asset_loss(
    admission, monkeypatch, count, status
):
    from backend import analysis_inputs as inputs, runtime_identity

    client, payload, tmp, main = admission
    originals = [
        source(tmp, payload["patient_id"], date=f"2026-0{i+1}-02") for i in range(count)
    ]
    request = {
        **payload,
        "report_ids": [r.id for r in originals],
        "special_instructions": "  exact café\n",
        "operation_id": "lost-create",
    }
    first = client.post("/api/runs", json=request).json()
    with storage.session_scope() as s:
        run = storage.get_run(s, first["id"])
        run.status = status
        s.commit()
    expected = client.get(f"/api/runs/{first['id']}").json()
    import shutil

    for original in originals:
        shutil.rmtree(Path(original.stored_path).parent)
    main.DISCOVERED_MODEL_IDS.clear()

    def forbidden(*args, **kwargs):
        raise AssertionError(
            "A banked receipt must not read sources, runtime, or schedule work"
        )

    monkeypatch.setattr(inputs, "_source_snapshot", forbidden)
    monkeypatch.setattr(runtime_identity, "current_runtime_identity", forbidden)
    monkeypatch.setattr(main, "_spawn_task", forbidden)
    storage.reset_engine(f"sqlite:///{tmp / 'app.db'}")
    storage.init_db()
    repeated = client.post("/api/runs", json=request)
    assert repeated.status_code == 200, repeated.text
    assert repeated.json() == expected


@pytest.mark.parametrize("reappears", [False, True])
@pytest.mark.parametrize("interrupted", [False, True])
def test_exact_rejoin_preserves_fallback_and_reserved_models(
    admission, monkeypatch, reappears, interrupted
):
    from backend import analysis_inputs as inputs

    client, payload, tmp, main = admission
    originals = [
        source(tmp, payload["patient_id"], date=d) for d in ["2026-02-02", "2026-01-02"]
    ]
    request = {
        **payload,
        "report_ids": [r.id for r in originals],
        "operation_id": "saved-fallback",
        "council_model_ids": ["preferred"],
        "allowed_model_fallbacks": {"preferred": "mock-council-a"},
    }
    compose = inputs._compose

    def crash(*args):
        raise RuntimeError("exit after reservation")

    if interrupted:
        monkeypatch.setattr(inputs, "_compose", crash)
    first = client.post("/api/runs", json=request)
    assert first.status_code == (500 if interrupted else 200)
    with storage.session_scope() as s:
        reservation = s.get(storage.AnalysisInputReservation, request["operation_id"])
        manifest = reservation.manifest_json
    monkeypatch.setattr(inputs, "_compose", compose)
    main.DISCOVERED_MODEL_IDS.clear()
    if reappears:
        main.DISCOVERED_MODEL_IDS.update(["preferred", "mock-consolidator"])
    repeated = client.post("/api/runs", json=request)
    assert repeated.status_code == 200, repeated.text
    run = repeated.json()
    assert run["id"] == reservation.run_id
    assert run["report_id"] == reservation.report_id
    assert run["source_manifest"] == json.loads(manifest)
    assert run["resolved_model_ids"] == ["mock-council-a", "mock-consolidator"]
    with storage.session_scope() as s:
        assert len(list(s.scalars(select(storage.Run)))) == 1


@pytest.mark.parametrize(
    "change",
    [
        "add-alias",
        "remove-alias",
        "change-alias",
        "add-fallback",
        "remove-fallback",
        "change-fallback",
        "models",
        "instructions",
        "order",
        "patient",
    ],
)
def test_exact_operation_rejects_changed_authorization_before_catalogue(
    admission, change
):
    client, payload, tmp, main = admission
    originals = [
        source(tmp, payload["patient_id"], date=d) for d in ["2026-01-02", "2026-02-02"]
    ]
    ids = [r.id for r in originals]
    request = {
        **payload,
        "report_ids": ids,
        "operation_id": "immutable-request",
        "allowed_model_fallbacks": {"mock-council-a": "mock-council-b"},
    }
    aliases = {rid: {"1": i + 1} for i, rid in enumerate(ids)}
    if change in ["remove-alias", "change-alias"]:
        request["source_session_aliases"] = aliases
    first = client.post("/api/runs", json=request)
    assert first.status_code == 200, first.text
    if change == "add-alias":
        request["source_session_aliases"] = aliases
    elif change == "remove-alias":
        request.pop("source_session_aliases")
    elif change == "change-alias":
        request["source_session_aliases"] = {ids[0]: {"1": 3}, ids[1]: {"1": 4}}
    elif change == "add-fallback":
        request["allowed_model_fallbacks"]["unused"] = "mock-council-b"
    elif change == "remove-fallback":
        request.pop("allowed_model_fallbacks")
    elif change == "change-fallback":
        request["allowed_model_fallbacks"]["mock-council-a"] = "mock-consolidator"
    elif change == "models":
        request["council_model_ids"] = ["unknown"]
    elif change == "instructions":
        request["special_instructions"] = " "
    elif change == "order":
        request["report_ids"] = ids[::-1]
    else:
        request["patient_id"] = "other"
    main.DISCOVERED_MODEL_IDS.clear()
    response = client.post("/api/runs", json=request)
    assert response.status_code == 409
    assert response.json()["detail"]["code"] == "ANALYSIS_OPERATION_CONFLICT"
    assert client.get(f"/api/runs/{first.json()['id']}").json() == first.json()


@pytest.mark.parametrize(
    "status", ["created", "failed", "needs_auth", "running", "complete"]
)
def test_start_availability_gate_only_for_new_dispatch(admission, monkeypatch, status):
    client, payload, tmp, main = admission
    original = source(tmp, payload["patient_id"])
    first = client.post("/api/runs", json={**payload, "report_id": original.id}).json()
    with storage.session_scope() as s:
        run = storage.get_run(s, first["id"])
        run.status = status
        s.commit()

    def forbidden(*args, **kwargs):
        raise AssertionError("No generation may be scheduled")

    monkeypatch.setattr(main, "_spawn_task", forbidden)
    main.DISCOVERED_MODEL_IDS.clear()
    response = client.post(f"/api/runs/{first['id']}/start")
    if status == "complete":
        assert response.status_code == 200, response.text
        assert response.json()["status"] == status
    else:
        assert response.status_code == 409
        assert response.json()["detail"]["code"] == (
            "ANALYSIS_MODEL_MISMATCH"
            if status == "created"
            else "LEGACY_RECONCILIATION_REQUIRED"
        )
    assert client.get(f"/api/runs/{first['id']}").json()["start_requested_at"] is None


def test_admission_snapshot_migration_preserves_legacy_reservation_twice(temp_data_dir):
    from sqlalchemy import create_engine

    path = temp_data_dir / "legacy-reservation.db"
    legacy = create_engine(f"sqlite:///{path}")
    with legacy.begin() as conn:
        conn.exec_driver_sql(
            "CREATE TABLE analysis_input_reservations (operation_id VARCHAR PRIMARY KEY, request_fingerprint VARCHAR NOT NULL, envelope_fingerprint VARCHAR NOT NULL, manifest_json TEXT NOT NULL, report_id VARCHAR NOT NULL, run_id VARCHAR NOT NULL UNIQUE, created_at DATETIME)"
        )
        conn.exec_driver_sql(
            "INSERT INTO analysis_input_reservations VALUES ('old-op', 'request-hash', 'envelope-hash', '{\"legacy\":false}', 'old-report', 'old-run', '2026-01-01')"
        )
        original = conn.exec_driver_sql(
            "SELECT * FROM analysis_input_reservations"
        ).one()
    storage.reset_engine(f"sqlite:///{path}")
    for _ in range(2):
        storage.init_db()
        with storage.engine.begin() as conn:
            migrated = conn.exec_driver_sql(
                "SELECT * FROM analysis_input_reservations"
            ).one()
        assert tuple(migrated[:7]) == tuple(original)
        assert tuple(migrated[7:]) == (None, None)


@pytest.mark.parametrize(
    "change", ["same", "assets", "aliases", "fallback", "catalogue", "models"]
)
def test_legacy_reservation_remains_strict_without_invented_snapshot(admission, change):
    client, payload, tmp, main = admission
    report = source(tmp, payload["patient_id"])
    request = {**payload, "report_id": report.id, "operation_id": "legacy-snapshot"}
    first = client.post("/api/runs", json=request).json()
    # Reproduce a real pre-migration reservation: original fingerprints/manifest
    # remain available, while the new exact request/resolution fields are unknown.
    with storage.session_scope() as s:
        reservation = s.get(storage.AnalysisInputReservation, request["operation_id"])
        reservation.immutable_request_json = None
        reservation.model_fields_json = None
        s.commit()
        original_identity = (
            reservation.run_id,
            reservation.report_id,
            reservation.manifest_json,
            reservation.envelope_fingerprint,
            reservation.request_fingerprint,
        )
    if change == "assets":
        Path(report.stored_path).unlink()
    elif change == "aliases":
        request["source_session_aliases"] = {report.id: {"1": 2}}
    elif change == "fallback":
        request["allowed_model_fallbacks"] = {"mock-council-a": "mock-council-b"}
    elif change == "catalogue":
        main.DISCOVERED_MODEL_IDS.clear()
    elif change == "models":
        request["council_model_ids"] = ["mock-council-b"]
    repeated = client.post("/api/runs", json=request)
    assert repeated.status_code == (200 if change == "same" else 409), repeated.text
    if change == "same":
        assert repeated.json() == first
    with storage.session_scope() as s:
        reservation = s.get(storage.AnalysisInputReservation, request["operation_id"])
        assert (
            reservation.run_id,
            reservation.report_id,
            reservation.manifest_json,
            reservation.envelope_fingerprint,
            reservation.request_fingerprint,
        ) == original_identity
        assert reservation.immutable_request_json is None
        assert reservation.model_fields_json is None


def test_reserved_recovery_keeps_saved_mapping_despite_policy_change(
    admission, monkeypatch
):
    from backend import analysis_inputs as inputs

    client, payload, tmp, main = admission
    originals = [
        source(tmp, payload["patient_id"], date=d) for d in ["2026-02-02", "2026-01-02"]
    ]
    request = {
        **payload,
        "report_ids": [r.id for r in originals],
        "operation_id": "saved-mapping",
    }
    compose = inputs._compose

    def crash(*args):
        raise RuntimeError("exit after reservation")

    monkeypatch.setattr(inputs, "_compose", crash)
    assert client.post("/api/runs", json=request).status_code == 500
    with storage.session_scope() as s:
        prior = s.get(storage.AnalysisInputReservation, request["operation_id"])
        saved_models = json.loads(prior.model_fields_json)
    monkeypatch.setattr(inputs, "_compose", compose)

    def forbidden(*args):
        raise AssertionError("Mapping already reserved")

    monkeypatch.setattr(inputs, "resolve_aliases", forbidden)
    main.DISCOVERED_MODEL_IDS.clear()
    response = client.post("/api/runs", json=request)
    assert response.status_code == 200, response.text
    assert response.json()["source_manifest"] == json.loads(prior.manifest_json)
    for key, value in saved_models.items():
        assert response.json()[key] == value
