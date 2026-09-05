from __future__ import annotations

from pathlib import Path


def test_register_patient_file_copies_and_upserts(temp_data_dir):
    from backend import storage
    from scripts.register_patient_file import register_patient_file

    with storage.session_scope() as session:
        storage.create_patient(session, label="EN_05-13-1947", notes="")

    source = Path(temp_data_dir) / "video.mp4"
    source.write_bytes(b"first")

    first = register_patient_file(
        patient_label="EN_05-13-1947",
        src_path=source,
        filename="EN_05-13-1947.mp4",
        mime_type="video/mp4",
    )
    assert first["filename"] == "EN_05-13-1947.mp4"
    assert first["mime_type"] == "video/mp4"
    assert Path(str(first["stored_path"])).read_bytes() == b"first"

    source.write_bytes(b"second")
    second = register_patient_file(
        patient_label="EN_05-13-1947",
        src_path=source,
        filename="EN_05-13-1947.mp4",
        mime_type="video/mp4",
    )
    assert second["id"] == first["id"]
    assert Path(str(second["stored_path"])).read_bytes() == b"second"

    with storage.session_scope() as session:
        files = storage.list_patient_files(session, first["patient_id"])
    assert len(files) == 1
