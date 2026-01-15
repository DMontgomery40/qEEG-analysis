from __future__ import annotations

import mimetypes
import os
import shutil
from pathlib import Path
from typing import BinaryIO

from .config import PATIENT_FILES_DIR


def _safe_suffix(filename: str) -> str:
    base = os.path.basename(filename or "")
    _, ext = os.path.splitext(base)
    ext = (ext or "").lower()
    if ext and len(ext) <= 10:
        return ext
    return ""


def _guess_mime_type(filename: str, provided: str | None) -> str:
    if provided and isinstance(provided, str) and provided.strip():
        return provided
    guessed, _ = mimetypes.guess_type(filename)
    return guessed or "application/octet-stream"


def patient_file_dir(patient_id: str, file_id: str) -> Path:
    return PATIENT_FILES_DIR / patient_id / file_id


def patient_file_original_path(patient_id: str, file_id: str, filename: str) -> Path:
    ext = _safe_suffix(filename)
    if not ext:
        ext = ".bin"
    return patient_file_dir(patient_id, file_id) / f"original{ext}"


def save_patient_file_upload(
    *,
    patient_id: str,
    file_id: str,
    filename: str,
    provided_mime_type: str | None,
    src: BinaryIO,
) -> tuple[Path, str, int]:
    mime_type = _guess_mime_type(filename, provided_mime_type)
    out_dir = patient_file_dir(patient_id, file_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    original_path = patient_file_original_path(patient_id, file_id, filename)
    with original_path.open("wb") as f:
        shutil.copyfileobj(src, f)
    size_bytes = int(original_path.stat().st_size)
    return original_path, mime_type, size_bytes

