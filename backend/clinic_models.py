"""Additive catalogue records in the existing engine database."""

from sqlalchemy import ForeignKey, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column
from .storage import Base


class ClinicCatalogState(Base):
    __tablename__ = "clinic_catalog_state"
    id: Mapped[int] = mapped_column(primary_key=True)
    revision: Mapped[int] = mapped_column(default=0)
    import_complete: Mapped[bool] = mapped_column(default=False)
    import_manifest: Mapped[str | None] = mapped_column(Text)
    updated_at: Mapped[int] = mapped_column(default=0)


class ClinicPatientAlias(Base):
    __tablename__ = "clinic_patient_aliases"
    alias: Mapped[str] = mapped_column(String, primary_key=True)
    ambiguous: Mapped[bool] = mapped_column(default=False)
    patient_uuid: Mapped[str] = mapped_column(ForeignKey("patients.id"))


class ClinicArtifact(Base):
    __tablename__ = "clinic_artifacts"
    __table_args__ = (
        UniqueConstraint("source_kind", "source_id"),
        UniqueConstraint("patient_uuid", "logical_family", "version"),
        UniqueConstraint("patient_uuid", "file_key"),
    )
    id: Mapped[str] = mapped_column(String, primary_key=True)
    patient_uuid: Mapped[str] = mapped_column(ForeignKey("patients.id"), index=True)
    source_kind: Mapped[str] = mapped_column(String)
    source_id: Mapped[str] = mapped_column(String)
    logical_family: Mapped[str] = mapped_column(String)
    version: Mapped[int]
    file_key: Mapped[str] = mapped_column(String)
    original_name: Mapped[str] = mapped_column(Text)
    sha256: Mapped[str] = mapped_column(String)
    size: Mapped[int]
    content_type: Mapped[str] = mapped_column(String)
    document_kind: Mapped[str | None] = mapped_column(String)
    session_date: Mapped[str | None] = mapped_column(String)
    generated_at: Mapped[int | None]
    registered_at: Mapped[int]
    uploaded_at: Mapped[int | None]
    uploaded_by: Mapped[str | None] = mapped_column(String)
    provenance_json: Mapped[str] = mapped_column(Text)
    archived: Mapped[bool] = mapped_column(default=False)


class ClinicLocation(Base):
    __tablename__ = "clinic_locations"
    __table_args__ = (UniqueConstraint("artifact_id", "kind", "key"),)
    id: Mapped[str] = mapped_column(String, primary_key=True)
    artifact_id: Mapped[str] = mapped_column(
        ForeignKey("clinic_artifacts.id"), index=True
    )
    kind: Mapped[str] = mapped_column(String)
    key: Mapped[str] = mapped_column(Text)
    patient_alias: Mapped[str] = mapped_column(String)
    verified: Mapped[bool] = mapped_column(default=False)
    active: Mapped[bool] = mapped_column(default=True)
    verified_at: Mapped[int | None]
    fingerprint: Mapped[str | None] = mapped_column(Text)


class ClinicProjection(Base):
    """Incomplete source metadata, not a worker or execution state machine."""

    __tablename__ = "clinic_projections"
    __table_args__ = (UniqueConstraint("source_kind", "source_id"),)
    id: Mapped[str] = mapped_column(String, primary_key=True)
    patient_uuid: Mapped[str] = mapped_column(ForeignKey("patients.id"), index=True)
    source_kind: Mapped[str] = mapped_column(String)
    source_id: Mapped[str] = mapped_column(String)
    payload_json: Mapped[str] = mapped_column(Text)
    error: Mapped[str | None] = mapped_column(String)
    artifact_id: Mapped[str | None] = mapped_column(ForeignKey("clinic_artifacts.id"))


class CatalogueUnavailable(RuntimeError):
    pass


class CatalogueConflict(ValueError):
    pass


class CatalogueNotFound(LookupError):
    pass


class ClinicPatientCatalogState(Base):
    """Last committed catalogue change for one existing Patient row."""

    __tablename__ = "clinic_patient_catalog_state"
    patient_uuid: Mapped[str] = mapped_column(
        ForeignKey("patients.id"), primary_key=True
    )
    revision: Mapped[int]
    updated_at: Mapped[int]


class ClinicPublication(Base):
    """One immutable transfer target for an existing artifact, never a queue."""

    __tablename__ = "clinic_publications"
    artifact_id: Mapped[str] = mapped_column(
        ForeignKey("clinic_artifacts.id"), primary_key=True
    )
    remote_key: Mapped[str] = mapped_column(Text, unique=True)
