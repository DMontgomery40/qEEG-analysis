"""Shared filing, feedback and original-operation references in existing app.db."""

from sqlalchemy import ForeignKey, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column
from .storage import Base


class ClinicUpload(Base):
    __tablename__ = "clinic_uploads"
    id: Mapped[str] = mapped_column(String, primary_key=True)
    admission_key: Mapped[str] = mapped_column(String, unique=True)
    manifest_json: Mapped[str] = mapped_column(Text)
    patient_uuid: Mapped[str | None] = mapped_column(ForeignKey("patients.id"))
    status: Mapped[str] = mapped_column(default="pending")
    conflict_json: Mapped[str | None] = mapped_column(Text)
    resolution_json: Mapped[str | None] = mapped_column(Text)
    uploaded_at: Mapped[int]
    uploaded_by: Mapped[str | None]
    uploaded_principal: Mapped[str | None]
    analysis_json: Mapped[str | None] = mapped_column(Text)


class ClinicUploadItem(Base):
    __tablename__ = "clinic_upload_items"
    __table_args__ = (UniqueConstraint("upload_id", "position"),)
    id: Mapped[str] = mapped_column(String, primary_key=True)
    upload_id: Mapped[str] = mapped_column(ForeignKey("clinic_uploads.id"))
    position: Mapped[int]
    metadata_json: Mapped[str] = mapped_column(Text)
    staging_path: Mapped[str] = mapped_column(Text)
    source_id: Mapped[str] = mapped_column(String)
    source_kind: Mapped[str]
    status: Mapped[str] = mapped_column(default="pending")
    artifact_id: Mapped[str | None] = mapped_column(ForeignKey("clinic_artifacts.id"))
    error: Mapped[str | None] = mapped_column(Text)
    projection_path: Mapped[str | None] = mapped_column(Text)


class ClinicMutation(Base):
    __tablename__ = "clinic_mutations"
    key: Mapped[str] = mapped_column(String, primary_key=True)
    material_json: Mapped[str] = mapped_column(Text)
    result_json: Mapped[str] = mapped_column(Text)


class ClinicLegacyUpload(Base):
    __tablename__ = "clinic_legacy_uploads"
    id: Mapped[str] = mapped_column(String, primary_key=True)
    evidence_json: Mapped[str] = mapped_column(Text)
    record_json: Mapped[str] = mapped_column(Text)


class ClinicFeedback(Base):
    __tablename__ = "clinic_feedback"
    id: Mapped[str] = mapped_column(String, primary_key=True)
    material_json: Mapped[str] = mapped_column(Text)
    artifact_id: Mapped[str] = mapped_column(
        ForeignKey("clinic_artifacts.id"), index=True
    )
    action: Mapped[str]
    notes: Mapped[str] = mapped_column(Text)
    author: Mapped[str | None]
    principal: Mapped[str | None]
    created_at: Mapped[int]
    sequence: Mapped[int] = mapped_column(unique=True)
    notification_json: Mapped[str | None] = mapped_column(Text)


class ClinicOperation(Base):
    __tablename__ = "clinic_operations"
    id: Mapped[str] = mapped_column(String, primary_key=True)
    patient_uuid: Mapped[str] = mapped_column(ForeignKey("patients.id"))
    producer: Mapped[str]
    kind: Mapped[str]
    original_json: Mapped[str] = mapped_column(Text)
    generation: Mapped[int]
    sequence: Mapped[int]
    payload_json: Mapped[str] = mapped_column(Text)
