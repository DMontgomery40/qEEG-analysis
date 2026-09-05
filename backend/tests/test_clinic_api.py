"""Exercise actual loopback HTTP with the production router and real scratch bytes."""

import socket
import threading
import time
from urllib.parse import quote

import httpx
import pytest
import uvicorn

from backend import storage, clinic_catalogue as catalogue
from backend import clinic_catalogue_reads as reads


@pytest.fixture
def live_api(temp_data_dir):
    with storage.session_scope() as session:
        chart = storage.create_patient(session, label="ZZ_01-01-1900_12")
    from backend import main

    app = main.app
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    server = uvicorn.Server(
        uvicorn.Config(app, log_level="error", lifespan="off", ws="none")
    )
    thread = threading.Thread(
        target=server.run, kwargs={"sockets": [sock]}, daemon=True
    )
    thread.start()
    for _ in range(200):
        if server.started:
            break
        time.sleep(0.01)
    assert server.started
    with httpx.Client(
        base_url=f"http://127.0.0.1:{sock.getsockname()[1]}/api/clinic"
    ) as client:
        yield client, chart, temp_data_dir
    server.should_exit = True
    thread.join(5)
    sock.close()
    assert not thread.is_alive()


def test_http_bytes_range_head_and_exact_binding(live_api):
    client, chart, root = live_api
    path = root / "video.mp4"
    data = bytes(range(256)) * 1024
    path.write_bytes(data)
    artifact = catalogue.register_artifact(
        patient_uuid=chart.id,
        source_kind="renderer",
        source_id="operation-1",
        logical_family="video",
        original_name="01-01-1983-0__vidéo.mp4",
        local_path=path,
        document_kind="video",
        generated_at=1234,
        provenance={"operationId": "operation-1"},
    )
    params = {"patientId": chart.label, "fileId": artifact["fileId"]}
    response = client.get("/file", params=params)
    assert response.status_code == 200 and response.content == data
    assert response.headers["content-type"] == "video/mp4"
    assert (
        quote(artifact["downloadName"], safe="")
        in response.headers["content-disposition"]
    )
    assert response.headers["content-length"] == str(len(data))
    for byte_range, expected in [
        ("bytes=10-29", data[10:30]),
        ("bytes=-9", data[-9:]),
        ("bytes=100-", data[100:]),
    ]:
        result = client.get("/file", params=params, headers={"Range": byte_range})
        assert result.status_code == 206 and result.content == expected
        head = client.head("/file", params=params, headers={"Range": byte_range})
        assert head.status_code == 206 and not head.content
        assert head.headers["content-length"] == str(len(expected))
    for invalid in ("bytes=999999-", "bytes=2-1", "bytes=0-1,3-4", "bytes=-0"):
        assert (
            client.get("/file", params=params, headers={"Range": invalid}).status_code
            == 416
        )
    assert (
        client.get(
            "/file", params=params, headers={"If-None-Match": response.headers["etag"]}
        ).status_code
        == 304
    )
    assert (
        client.get(
            "/file", params=params, headers={"Range": "bytes=0-2", "If-Range": '"old"'}
        ).content
        == data
    )
    path.write_bytes(b"changed")
    assert client.get("/file", params=params).status_code == 503
    assert (
        client.get(
            "/file", params={"patientId": chart.label, "fileKey": "/etc/passwd"}
        ).status_code
        == 404
    )


def test_read_routes_policy_roster_actor_and_pending_scope(live_api):
    client, chart, root = live_api
    assert client.get("/patients").json()["patients"][0]["patientId"] == chart.label
    policy = client.get("/policy")
    assert policy.json()["policy"]["tts"]["voice"] == "Charon"
    assert int(policy.headers["x-clinic-catalog-revision"]) >= 1
    assert client.get("/patients/" + chart.label).json()["patient"]["index"] == 12
    headers = {"X-Clinic-Actor": quote("Zoë Staff"), "X-Clinic-Principal": "workbench"}
    assert client.get("/patients", headers=headers).status_code == 200
    for actor in ("%zz", "%ff", "%0a", "", "a" * 129):
        assert (
            client.get(
                "/patients", headers={**headers, "X-Clinic-Actor": actor}
            ).status_code
            == 400
        )
    with storage.session_scope() as session:
        other = storage.create_patient(session, label="ZZ_01-01-1900_2")
        storage.create_patient_file(
            session,
            patient_id=other.id,
            filename="missing.pdf",
            mime_type="application/pdf",
            size_bytes=3,
            stored_path=root / "missing.pdf",
        )
    assert len(client.get("/patients").json()["patients"]) == 2
    assert (
        client.get("/patient-files", params={"patientId": chart.label}).json()["files"]
        == []
    )
    assert (
        client.get("/patient-files", params={"patientId": other.label}).status_code
        == 503
    )
    dates = client.post("/patient-report-dates", json={"patientIds": [chart.label]})
    assert dates.json()["patientReportDates"] == {chart.label: None}
    assert (
        client.post(
            "/patient-report-dates", json={"patientIds": [other.label]}
        ).status_code
        == 503
    )
    assert client.post("/catalogue/register", json={}).status_code == 404


@pytest.mark.asyncio
async def test_snapshot_closed_when_transport_fails_before_iteration():
    import io
    from backend.clinic_api import _SnapshotResponse

    stream = io.BytesIO(b"fixed")
    response = _SnapshotResponse(stream, iter([b"fixed"]))

    async def send(message):
        raise RuntimeError("disconnected before headers")

    async def receive():
        return {"type": "http.disconnect"}

    with pytest.raises(RuntimeError):
        await response({"type": "http", "asgi": {"spec_version": "2.4"}}, receive, send)
    assert stream.closed


def test_snapshot_closed_for_head_errors_and_complete_response(live_api, monkeypatch):
    client, chart, root = live_api
    path = root / "small.pdf"
    path.write_bytes(b"hello")
    artifact = catalogue.register_artifact(
        patient_uuid=chart.id,
        source_kind="test",
        source_id="close",
        original_name="small.pdf",
        logical_family="file",
        local_path=path,
    )
    opened = []
    real_open = reads.open_local_file

    def tracked(file_id):
        stream = real_open(file_id)
        opened.append(stream)
        return stream

    monkeypatch.setattr(reads, "open_local_file", tracked)
    params = {"patientId": chart.label, "fileId": artifact["fileId"]}
    assert client.head("/file", params=params).status_code == 200
    assert (
        client.get("/file", params=params, headers={"Range": "bytes=-0"}).status_code
        == 416
    )
    assert client.get("/file", params=params).content == b"hello"
    for _ in range(100):
        if all(stream.closed for stream in opened):
            break
        time.sleep(0.01)
    assert len(opened) == 3 and all(stream.closed for stream in opened)
    assert (
        client.post("/patient-report-dates", json={"patientIds": [{}]}).status_code
        == 400
    )


def test_existing_patient_and_attachment_http_apis_feed_shared_reads(live_api):
    client, _chart, root = live_api
    # These routes are the original production application, not a replacement writer.
    created = client.post(
        "/../patients",
        json={"first_name": "Ada", "last_name": "Baker", "birthdate": "02-02-1900"},
    )
    assert created.status_code == 200, created.text
    old_shape = created.json()
    patient_id = old_shape["patient_id"]
    assert old_shape["id"] != patient_id
    assert any(
        p["patientId"] == patient_id for p in client.get("/patients").json()["patients"]
    )
    attached = client.post(
        "/../patients/" + old_shape["id"] + "/files",
        files={"file": ("notes.txt", b"Original attachment", "text/plain")},
    )
    assert attached.status_code == 200, attached.text
    listing = client.get("/patient-files", params={"patientId": patient_id})
    assert listing.status_code == 200, listing.text
    row = listing.json()["files"][0]
    assert (
        client.get(
            "/file", params={"patientId": patient_id, "fileId": row["fileId"]}
        ).content
        == b"Original attachment"
    )
