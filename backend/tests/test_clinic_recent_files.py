"""Global recent cards use bounded SQL and the existing exact file policy."""

import pytest
from sqlalchemy import event
from backend import (
    storage,
    clinic_catalogue as catalogue,
    clinic_catalogue_reads as reads,
)
from backend.clinic_recent_files import recent_files
from backend.clinic_models import CatalogueConflict
from backend.tests.test_clinic_api import live_api  # noqa: F401
from backend.tests.clinic_test_helpers import forbid_clinic_paid  # noqa: F401


def seed(
    root, patient, index, *, kind="video", generated=1000, content_type="video/mp4"
):
    path = root / f"file-{index}"
    path.write_bytes(str(index).encode())
    return catalogue.register_artifact(
        patient_uuid=patient.id,
        source_kind="recent-test",
        source_id=str(index),
        logical_family="recent:" + kind,
        original_name=f"file-{index}.mp4",
        local_path=path,
        document_kind=kind,
        generated_at=generated,
        content_type=content_type,
        session_date="1900-01-01",
    )


def test_recent_cross_chart_order_limit_and_public_bindings(temp_data_dir, monkeypatch):
    from backend.clinic_feedback import record_feedback

    with storage.session_scope() as session:
        first = storage.create_patient(session, label="AA_01-01-1900")
        second = storage.create_patient(session, label="BB_01-01-1900")
        legacy = storage.create_patient(session, label="legacy")
    files = [
        seed(
            temp_data_dir,
            first if n % 2 else second,
            n,
            generated=1000 if n < 5 else None,
        )
        for n in range(8)
    ]
    seed(temp_data_dir, legacy, 90, generated=9999)
    pdf = seed(
        temp_data_dir,
        second,
        91,
        kind="patient-summary",
        content_type="application/pdf",
    )
    seed(
        temp_data_dir, second, 92, kind="patient-summary", content_type="text/markdown"
    )
    record_feedback(
        key="original-review",
        patient_id=files[0]["patientId"],
        file_id=files[0]["fileId"],
        version=files[0]["version"],
        action="approve",
        actor="Synthetic",
    )
    expected = sorted(
        files,
        key=lambda f: (
            f["generatedAt"] if f["generatedAt"] is not None else -1,
            f["sessionDate"],
            f["version"],
            f["fileId"],
        ),
        reverse=True,
    )
    calls = []

    def track(_conn, _cursor, statement, *_args):
        if statement.lstrip().upper().startswith("SELECT"):
            calls.append(statement)

    # The drawer must never visit each chart or hash bytes.
    monkeypatch.setattr(
        reads, "patient_files", lambda *a, **k: pytest.fail("per-chart traversal")
    )
    monkeypatch.setattr(
        catalogue, "_read_local", lambda *a, **k: pytest.fail("byte hashing")
    )
    event.listen(storage.engine, "before_cursor_execute", track)
    try:
        result = recent_files(kind="video", limit=3)
        small_count = len(calls)
        calls.clear()
        all_rows = recent_files(kind="video", limit=120)
        assert len(calls) == small_count <= 6
    finally:
        event.remove(storage.engine, "before_cursor_execute", track)
    assert [f["fileId"] for f in result["files"]] == [f["fileId"] for f in expected[:3]]
    assert result["truncated"] and result["returnedFiles"] == 3
    assert len(all_rows["files"]) == 8 and not all_rows["truncated"]
    for row in all_rows["files"]:
        exact = reads.file_binding(row["patientId"], file_id=row["fileId"])
        assert all(exact[k] == v for k, v in row.items())
    assert (
        recent_files(kind="patient-summary", content_type="application/pdf")["files"][
            0
        ]["fileId"]
        == pdf["fileId"]
    )
    with storage.session_scope() as session:
        storage.create_patient(session, label=first.label)
    with pytest.raises(CatalogueConflict):
        recent_files(kind="video")


@pytest.mark.parametrize(
    "query",
    [
        {"kind": "unknown"},
        {"kind": "video", "limit": "0"},
        {"kind": "video", "limit": "121"},
        {"kind": "video", "limit": "1.5"},
        {"kind": "video", "limit": "bad"},
        {"kind": "video", "patientId": "invented"},
    ],
)
def test_recent_http_rejects_invalid_query(live_api, query):
    client, _, _ = live_api
    assert client.get("/recent-files", params=query).status_code == 400


def test_recent_http_exact_current_chart_binding(live_api):
    client, chart, root = live_api
    artifact = seed(root, chart, "http")
    response = client.get("/recent-files", params={"kind": "video", "limit": "1"})
    assert response.status_code == 200
    assert response.json()["files"] == [artifact]
    assert response.headers["x-clinic-schema-version"] == "clinic-v1"
