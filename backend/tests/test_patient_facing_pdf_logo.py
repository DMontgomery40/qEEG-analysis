from __future__ import annotations


def test_patient_facing_logo_respects_env_jpeg(monkeypatch, tmp_path):
    from backend import patient_facing_pdf

    logo_path = tmp_path / "custom-logo.JPG"
    logo_path.write_bytes(b"\xff\xd8\xff\xe0")
    monkeypatch.setenv("QEEG_PATIENT_FACING_LOGO_PATH", str(logo_path))

    data_uri = patient_facing_pdf._get_logo_base64()
    assert data_uri.startswith("data:image/jpeg;base64,")


def test_patient_facing_logo_default_not_site_mapping(monkeypatch):
    from backend import patient_facing_pdf

    monkeypatch.delenv("QEEG_PATIENT_FACING_LOGO_PATH", raising=False)
    logo_path = patient_facing_pdf._resolve_logo_path()
    assert logo_path is not None
    assert "site-mapping-reference" not in logo_path.name.lower()
