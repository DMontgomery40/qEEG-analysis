from __future__ import annotations

import json
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Literal

from sqlalchemy import (
    DateTime,
    ForeignKey,
    UniqueConstraint,
    String,
    Text,
    create_engine,
    event,
    func,
    select,
    update,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, relationship
from sqlalchemy.engine import make_url

from .config import DATA_DIR, ensure_data_dirs


RunStatus = Literal["created", "running", "complete", "failed", "needs_auth"]


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _new_id() -> str:
    return str(uuid.uuid4())


class Base(DeclarativeBase):
    pass


class Patient(Base):
    __tablename__ = "patients"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_new_id)
    # The canonical clinic ID (XX_MM-DD-YYYY[_N]) lives here. `id` above stays an
    # internal relational key and never reaches a folder, filename, or clinic screen.
    label: Mapped[str] = mapped_column(String, nullable=False)
    notes: Mapped[str] = mapped_column(Text, nullable=False, default="")
    # Nullable so patients created before the identity columns existed still read back.
    birthdate: Mapped[str | None] = mapped_column(String, nullable=True, default=None)
    first_name: Mapped[str | None] = mapped_column(String, nullable=True, default=None)
    last_name: Mapped[str | None] = mapped_column(String, nullable=True, default=None)
    first_initial: Mapped[str | None] = mapped_column(String, nullable=True, default=None)
    last_initial: Mapped[str | None] = mapped_column(String, nullable=True, default=None)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)

    reports: Mapped[list["Report"]] = relationship(back_populates="patient")
    files: Mapped[list["PatientFile"]] = relationship(back_populates="patient")
    runs: Mapped[list["Run"]] = relationship(back_populates="patient")


class PatientIdReservation(Base):
    """Every canonical clinic ID ever issued, keyed by the ID itself.

    Rows are never deleted or compacted. A patient being deleted or relabelled
    retires its ID; it must never be handed to a different person afterwards.
    This is operational state: it belongs in database backups and no rebuild
    command may clear it.
    """

    __tablename__ = "patient_id_reservations"

    patient_id: Mapped[str] = mapped_column(String, primary_key=True)
    first_initial: Mapped[str] = mapped_column(String, nullable=False)
    last_initial: Mapped[str] = mapped_column(String, nullable=False)
    birthdate: Mapped[str] = mapped_column(String, nullable=False)
    ordinal: Mapped[int] = mapped_column(nullable=False, default=1)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)


class Report(Base):
    __tablename__ = "reports"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_new_id)
    patient_id: Mapped[str] = mapped_column(ForeignKey("patients.id"), nullable=False)
    filename: Mapped[str] = mapped_column(String, nullable=False)
    mime_type: Mapped[str] = mapped_column(String, nullable=False)
    stored_path: Mapped[str] = mapped_column(String, nullable=False)
    extracted_text_path: Mapped[str] = mapped_column(String, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)

    patient: Mapped[Patient] = relationship(back_populates="reports")
    runs: Mapped[list["Run"]] = relationship(back_populates="report")


class PatientFile(Base):
    __tablename__ = "patient_files"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_new_id)
    patient_id: Mapped[str] = mapped_column(ForeignKey("patients.id"), nullable=False)
    filename: Mapped[str] = mapped_column(String, nullable=False)
    mime_type: Mapped[str] = mapped_column(String, nullable=False)
    size_bytes: Mapped[int] = mapped_column(nullable=False, default=0)
    stored_path: Mapped[str] = mapped_column(String, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)

    patient: Mapped[Patient] = relationship(back_populates="files")


class Run(Base):
    __tablename__ = "runs"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_new_id)
    patient_id: Mapped[str] = mapped_column(ForeignKey("patients.id"), nullable=False)
    report_id: Mapped[str] = mapped_column(ForeignKey("reports.id"), nullable=False)
    status: Mapped[str] = mapped_column(String, nullable=False, default="created")
    council_model_ids_json: Mapped[str] = mapped_column(Text, nullable=False, default="[]")
    consolidator_model_id: Mapped[str] = mapped_column(String, nullable=False, default="")
    requested_model_ids_json: Mapped[str] = mapped_column(Text, nullable=False, default="[]")
    resolved_model_ids_json: Mapped[str] = mapped_column(Text, nullable=False, default="[]")
    creating_instance_id: Mapped[str] = mapped_column(String, nullable=False, default="")
    model_catalogue_fingerprint: Mapped[str] = mapped_column(String, nullable=False, default="")
    label_map_json: Mapped[str] = mapped_column(Text, nullable=False, default="{}")
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    selected_artifact_id: Mapped[str | None] = mapped_column(String, nullable=True)
    error_message: Mapped[str] = mapped_column(Text, nullable=False, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)

    source_report_ids_json: Mapped[str] = mapped_column(Text, nullable=False, default="[]")
    source_manifest_json: Mapped[str] = mapped_column(Text, nullable=False, default='{"legacy":true}')
    special_instructions: Mapped[str] = mapped_column(Text, nullable=False, default="")
    analysis_input_fingerprint: Mapped[str] = mapped_column(String, nullable=False, default="")
    operation_id: Mapped[str | None] = mapped_column(String, nullable=True, unique=True)

    # Durable execution is explicitly admitted; historical rows remain inactive.
    start_requested_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    execution_state: Mapped[str | None] = mapped_column(String, nullable=True)
    owner_token: Mapped[str | None] = mapped_column(String, nullable=True)
    owner_generation: Mapped[int] = mapped_column(
        nullable=False, default=0, server_default="0"
    )
    owner_pid: Mapped[int | None] = mapped_column(nullable=True)
    owner_started_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    next_check_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    blocked_reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    execution_manifest_path: Mapped[str | None] = mapped_column(String, nullable=True)
    execution_manifest_hash: Mapped[str | None] = mapped_column(String, nullable=True)

    patient: Mapped[Patient] = relationship(back_populates="runs")
    report: Mapped[Report] = relationship(back_populates="runs")
    artifacts: Mapped[list["Artifact"]] = relationship(back_populates="run")


class PaidRequest(Base):
    """E2 journal metadata; exact bodies remain immutable files, never row blobs.

    E2 owns dispatch classification/file reconciliation. All writes must run in
    RunOwner.transaction(), including orphan-file reconciliation after takeover.
    """

    __tablename__ = "paid_requests"
    run_id: Mapped[str] = mapped_column(ForeignKey("runs.id"), primary_key=True)
    scope_key: Mapped[str] = mapped_column(String, primary_key=True)
    dispatch_ordinal: Mapped[int] = mapped_column(primary_key=True)
    request_path: Mapped[str] = mapped_column(String, nullable=False)
    request_hash: Mapped[str] = mapped_column(String, nullable=False)
    route_json: Mapped[str] = mapped_column(Text, nullable=False)
    execution_manifest_hash: Mapped[str] = mapped_column(String, nullable=False)
    input_fingerprint: Mapped[str] = mapped_column(String, nullable=False)
    owner_token: Mapped[str] = mapped_column(String, nullable=False)
    owner_generation: Mapped[int] = mapped_column(nullable=False)
    state: Mapped[str] = mapped_column(String, nullable=False, default="prepared")
    response_path: Mapped[str | None] = mapped_column(String, nullable=True)
    response_hash: Mapped[str | None] = mapped_column(String, nullable=True)
    http_status: Mapped[int | None] = mapped_column(nullable=True)
    response_metadata_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    error_classification: Mapped[str | None] = mapped_column(String, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow
    )
    dispatched_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    response_saved_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )


class StageReceipt(Base):
    """E4 verified stage policy/member/artifact receipt stored in a hashed file.

    Insertion is E4's commit of the unchanged clinical policy, after checking all
    member terminal outcomes and artifact hashes. No legacy artifact inference.
    """

    __tablename__ = "stage_receipts"
    run_id: Mapped[str] = mapped_column(ForeignKey("runs.id"), primary_key=True)
    stage_num: Mapped[int] = mapped_column(primary_key=True)
    receipt_path: Mapped[str] = mapped_column(String, nullable=False)
    receipt_hash: Mapped[str] = mapped_column(String, nullable=False)
    execution_manifest_hash: Mapped[str] = mapped_column(String, nullable=False)
    input_fingerprint: Mapped[str] = mapped_column(String, nullable=False)
    policy_version: Mapped[str] = mapped_column(String, nullable=False)
    owner_token: Mapped[str] = mapped_column(String, nullable=False)
    owner_generation: Mapped[int] = mapped_column(nullable=False)
    completed_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow
    )


class PostObligation(Base):
    """One pinned post-run obligation per kind, independent of clinical status."""

    __tablename__ = "post_obligations"
    run_id: Mapped[str] = mapped_column(ForeignKey("runs.id"), primary_key=True)
    kind: Mapped[str] = mapped_column(String, primary_key=True)
    manifest_path: Mapped[str] = mapped_column(String, nullable=False)
    manifest_hash: Mapped[str] = mapped_column(String, nullable=False)
    owner_token: Mapped[str] = mapped_column(String, nullable=False)
    owner_generation: Mapped[int] = mapped_column(nullable=False)
    state: Mapped[str] = mapped_column(String, nullable=False, default="pending")
    receipt_path: Mapped[str | None] = mapped_column(String, nullable=True)
    receipt_hash: Mapped[str | None] = mapped_column(String, nullable=True)
    next_check_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    blocked_reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow
    )


class AnalysisInputReservation(Base):
    """Stable identities survive a crash before free composition or run registration."""

    __tablename__ = "analysis_input_reservations"
    operation_id: Mapped[str] = mapped_column(String, primary_key=True)
    request_fingerprint: Mapped[str] = mapped_column(String, nullable=False)
    envelope_fingerprint: Mapped[str] = mapped_column(String, nullable=False)
    immutable_request_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    model_fields_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    manifest_json: Mapped[str] = mapped_column(Text, nullable=False)
    report_id: Mapped[str] = mapped_column(String, nullable=False)
    run_id: Mapped[str] = mapped_column(String, nullable=False, unique=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)


class Artifact(Base):
    __tablename__ = "artifacts"
    __table_args__ = (
        UniqueConstraint("run_id", "operation_key", name="uq_artifacts_run_operation"),
    )

    operation_key: Mapped[str | None] = mapped_column(String, nullable=True)

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_new_id)
    run_id: Mapped[str] = mapped_column(ForeignKey("runs.id"), nullable=False)
    stage_num: Mapped[int] = mapped_column(nullable=False)
    stage_name: Mapped[str] = mapped_column(String, nullable=False)
    model_id: Mapped[str] = mapped_column(String, nullable=False, default="")
    kind: Mapped[str] = mapped_column(String, nullable=False)
    content_path: Mapped[str] = mapped_column(String, nullable=False)
    content_type: Mapped[str] = mapped_column(String, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)

    run: Mapped[Run] = relationship(back_populates="artifacts")


def get_db_path() -> Path:
    ensure_data_dirs()
    return DATA_DIR / "app.db"


def create_sqlite_engine(db_url: str):
    """Identical durability for production and scratch; retain sqlite3 transaction mode.

    The installed SQLAlchemy pysqlite dialect passes timeout to sqlite3 and
    supports per-connection listeners. No isolation_level/BEGIN/pool/WAL change.
    """
    url = make_url(db_url)
    if url.database and url.database != ":memory:" and not url.query:
        # pysqlite resolves filenames when creating its connection closure. Keep
        # the URL equally stable for ownership stores constructed after chdir.
        url = url.set(database=str(Path(url.database).resolve()))
    db_engine = create_engine(
        url,
        future=True,
        connect_args={"check_same_thread": False, "timeout": 5.0},
    )

    @event.listens_for(db_engine, "connect")
    def configure_connection(connection, _record):
        cursor = connection.cursor()
        try:
            cursor.execute("PRAGMA busy_timeout=5000")
            cursor.execute("PRAGMA synchronous=FULL")
        finally:
            cursor.close()

    return db_engine


engine = create_sqlite_engine(f"sqlite:///{get_db_path()}")


def reset_engine(db_url: str) -> None:
    """
    Test helper: point storage at a different SQLite URL (e.g. sqlite:////tmp/test.db).
    """
    global engine
    try:
        engine.dispose()
    except Exception:
        pass
    engine = create_sqlite_engine(db_url)


# Added to `patients` after the table shipped, so they need an in-place upgrade.
# All nullable: existing rows keep their values and read back as unset identity.
_PATIENT_IDENTITY_COLUMNS = {
    "birthdate": "VARCHAR",
    "first_name": "VARCHAR",
    "last_name": "VARCHAR",
    "first_initial": "VARCHAR",
    "last_initial": "VARCHAR",
}

_RUN_ATTESTATION_COLUMNS = {
    "requested_model_ids_json": "TEXT NOT NULL DEFAULT '[]'",
    "resolved_model_ids_json": "TEXT NOT NULL DEFAULT '[]'",
    "creating_instance_id": "VARCHAR NOT NULL DEFAULT ''",
    "model_catalogue_fingerprint": "VARCHAR NOT NULL DEFAULT ''",
}


def _ensure_patient_identity_columns() -> None:
    """Add identity columns to a `patients` table created before they existed.

    `create_all` only creates missing tables, so the clinic's live database
    would otherwise keep the old five-column shape forever.
    """
    with engine.begin() as conn:
        present = {row[1] for row in conn.exec_driver_sql("PRAGMA table_info(patients)")}
        for column, sql_type in _PATIENT_IDENTITY_COLUMNS.items():
            if column not in present:
                conn.exec_driver_sql(
                    f"ALTER TABLE patients ADD COLUMN {column} {sql_type}"
                )


def _ensure_run_attestation_columns() -> None:
    with engine.begin() as conn:
        present = {row[1] for row in conn.exec_driver_sql("PRAGMA table_info(runs)")}
        for column, sql_type in _RUN_ATTESTATION_COLUMNS.items():
            if column not in present:
                conn.exec_driver_sql(f"ALTER TABLE runs ADD COLUMN {column} {sql_type}")


def _ensure_analysis_input_columns() -> None:
    columns = {
        "source_report_ids_json": "TEXT NOT NULL DEFAULT '[]'",
        "source_manifest_json": "TEXT NOT NULL DEFAULT '{\"legacy\":true}'",
        "special_instructions": "TEXT NOT NULL DEFAULT ''",
        "analysis_input_fingerprint": "VARCHAR NOT NULL DEFAULT ''",
        "operation_id": "VARCHAR",
    }
    with engine.begin() as conn:
        present = {row[1] for row in conn.exec_driver_sql("PRAGMA table_info(runs)")}
        for column, sql_type in columns.items():
            if column not in present:
                conn.exec_driver_sql(f"ALTER TABLE runs ADD COLUMN {column} {sql_type}")
        conn.exec_driver_sql(
            "CREATE UNIQUE INDEX IF NOT EXISTS ix_runs_operation_id ON runs(operation_id)"
        )
        # Nullable snapshots distinguish legacy reservations without inventing their
        # original authorization or resolution. Repeated startup preserves every row.
        reservation_columns = {
            row[1]
            for row in conn.exec_driver_sql(
                "PRAGMA table_info(analysis_input_reservations)"
            )
        }
        for column in ("immutable_request_json", "model_fields_json"):
            if column not in reservation_columns:
                conn.exec_driver_sql(
                    f"ALTER TABLE analysis_input_reservations ADD COLUMN {column} TEXT"
                )
        # Existing runs retain their original report, provenance, and empty instructions.
        rows = conn.exec_driver_sql(
            "SELECT id, report_id FROM runs WHERE source_report_ids_json = '[]'"
        ).fetchall()
        for row in rows:
            conn.exec_driver_sql(
                "UPDATE runs SET source_report_ids_json = ? WHERE id = ?",
                (json.dumps([row[1]]), row[0]),
            )


def _ensure_run_execution_columns() -> None:
    columns = {
        "start_requested_at": "DATETIME",
        "execution_state": "VARCHAR",
        "owner_token": "VARCHAR",
        "owner_generation": "INTEGER NOT NULL DEFAULT 0",
        "owner_pid": "INTEGER",
        "owner_started_at": "DATETIME",
        "next_check_at": "DATETIME",
        "blocked_reason": "TEXT",
        "execution_manifest_path": "VARCHAR",
        "execution_manifest_hash": "VARCHAR",
    }
    with engine.begin() as conn:
        present = {row[1] for row in conn.exec_driver_sql("PRAGMA table_info(runs)")}
        for column, sql_type in columns.items():
            if column not in present:
                conn.exec_driver_sql(f"ALTER TABLE runs ADD COLUMN {column} {sql_type}")


def _ensure_artifact_operation_key() -> None:
    with engine.begin() as conn:
        present = {
            row[1] for row in conn.exec_driver_sql("PRAGMA table_info(artifacts)")
        }
        if "operation_key" not in present:
            conn.exec_driver_sql(
                "ALTER TABLE artifacts ADD COLUMN operation_key VARCHAR"
            )
        conn.exec_driver_sql(
            "CREATE UNIQUE INDEX IF NOT EXISTS uq_artifacts_run_operation ON artifacts(run_id, operation_key)"
        )


def _ensure_clinic_location_lookup_index() -> None:
    with engine.begin() as conn:
        conn.exec_driver_sql(
            "CREATE INDEX IF NOT EXISTS ix_clinic_locations_kind_key_active_artifact "
            "ON clinic_locations(kind, key, active, artifact_id)"
        )


def init_db() -> None:
    from . import clinic_records  # noqa: F401 - additive tables in the original Base
    from .clinic_catalogue import initialize_catalogue

    ensure_data_dirs()
    Base.metadata.create_all(engine)
    _ensure_patient_identity_columns()
    _ensure_run_attestation_columns()
    _ensure_analysis_input_columns()
    _ensure_run_execution_columns()
    _ensure_artifact_operation_key()
    _ensure_clinic_location_lookup_index()
    initialize_catalogue()


@contextmanager
def session_scope() -> Iterable[Session]:
    # Avoid returning expired/detached ORM objects that later trigger lazy refresh
    # after the session has been closed (common in background workflows).
    with Session(engine, expire_on_commit=False) as session:
        yield session


def _touch_updated_at(obj: Any) -> None:
    if hasattr(obj, "updated_at"):
        obj.updated_at = _utcnow()


def list_patients(session: Session) -> list[Patient]:
    return list(session.scalars(select(Patient).order_by(Patient.created_at.desc())))


def find_patients_by_label(session: Session, label: str) -> list[Patient]:
    normalized = (label or "").strip()
    if not normalized:
        return []
    return list(session.scalars(select(Patient).where(func.lower(Patient.label) == normalized.lower())))


def get_patient(session: Session, patient_id: str) -> Patient | None:
    return session.get(Patient, patient_id)


def create_patient(
    session: Session,
    *,
    label: str,
    notes: str = "",
    birthdate: str | None = None,
    first_name: str | None = None,
    last_name: str | None = None,
    first_initial: str | None = None,
    last_initial: str | None = None,
    commit: bool = True,
) -> Patient:
    patient = Patient(
        label=label,
        notes=notes,
        birthdate=birthdate,
        first_name=first_name,
        last_name=last_name,
        first_initial=first_initial,
        last_initial=last_initial,
    )
    session.add(patient)
    if commit:
        session.commit()
    else:
        session.flush()
    session.refresh(patient)
    return patient


def update_patient(
    session: Session,
    patient_id: str,
    *,
    label: str,
    notes: str | None = None,
    birthdate: str | None = None,
    first_name: str | None = None,
    last_name: str | None = None,
    first_initial: str | None = None,
    last_initial: str | None = None,
    commit: bool = True,
) -> Patient | None:
    """Update a patient. Fields left as None keep their stored value.

    That includes ``notes``: they hold what the agent has learned about this
    patient, so a caller that simply does not mention them must not erase them.
    Passing a string — including ``""`` — still replaces what is stored.
    """
    patient = session.get(Patient, patient_id)
    if patient is None:
        return None
    patient.label = label
    for field, value in (
        ("notes", notes),
        ("birthdate", birthdate),
        ("first_name", first_name),
        ("last_name", last_name),
        ("first_initial", first_initial),
        ("last_initial", last_initial),
    ):
        if value is not None:
            setattr(patient, field, value)
    _touch_updated_at(patient)
    if commit:
        session.commit()
    else:
        session.flush()
    session.refresh(patient)
    return patient


def list_reports(session: Session, patient_id: str) -> list[Report]:
    return list(
        session.scalars(
            select(Report).where(Report.patient_id == patient_id).order_by(Report.created_at.desc())
        )
    )


def list_patient_files(session: Session, patient_id: str) -> list[PatientFile]:
    return list(
        session.scalars(
            select(PatientFile)
            .where(PatientFile.patient_id == patient_id)
            .order_by(PatientFile.created_at.desc())
        )
    )


def create_patient_file(
    session: Session,
    *,
    file_id: str | None = None,
    patient_id: str,
    filename: str,
    mime_type: str,
    size_bytes: int,
    stored_path: Path,
    commit: bool = True,
) -> PatientFile:
    f = PatientFile(
        id=file_id if file_id else _new_id(),
        patient_id=patient_id,
        filename=filename,
        mime_type=mime_type,
        size_bytes=int(size_bytes),
        stored_path=str(stored_path),
    )
    session.add(f)
    if commit:
        session.commit()
    else:
        session.flush()
    session.refresh(f)
    return f


def get_patient_file(session: Session, file_id: str) -> PatientFile | None:
    return session.get(PatientFile, file_id)


def delete_patient_file(session: Session, file_id: str) -> PatientFile | None:
    f = session.get(PatientFile, file_id)
    if f is None:
        return None
    session.delete(f)
    session.commit()
    return f


def create_report(
    session: Session,
    *,
    report_id: str | None = None,
    patient_id: str,
    filename: str,
    mime_type: str,
    stored_path: Path,
    extracted_text_path: Path,
    commit: bool = True,
) -> Report:
    report = Report(
        id=report_id if report_id else _new_id(),
        patient_id=patient_id,
        filename=filename,
        mime_type=mime_type,
        stored_path=str(stored_path),
        extracted_text_path=str(extracted_text_path),
    )
    session.add(report)
    if commit:
        session.commit()
    else:
        session.flush()
    session.refresh(report)
    return report


def get_report(session: Session, report_id: str) -> Report | None:
    return session.get(Report, report_id)


def list_runs(session: Session, patient_id: str) -> list[Run]:
    return list(
        session.scalars(select(Run).where(Run.patient_id == patient_id).order_by(Run.created_at.desc()))
    )


def get_run(session: Session, run_id: str) -> Run | None:
    return session.get(Run, run_id)


def create_run(
    session: Session,
    *,
    patient_id: str,
    report_id: str,
    council_model_ids: list[str],
    consolidator_model_id: str,
    requested_model_ids: list[str] | None = None,
    resolved_model_ids: list[str] | None = None,
    creating_instance_id: str = "",
    model_catalogue_fingerprint: str = "",
    source_report_ids: list[str] | None = None,
    source_manifest: dict[str, Any] | None = None,
    special_instructions: str = "",
    analysis_input_fingerprint: str = "",
    operation_id: str | None = None,
    run_id: str | None = None,
) -> Run:
    requested = list(requested_model_ids or [])
    resolved = list(resolved_model_ids or [])
    run = Run(
        id=run_id or _new_id(),
        source_report_ids_json=json.dumps(source_report_ids or [report_id]),
        source_manifest_json=json.dumps(source_manifest or {"legacy": True}),
        special_instructions=special_instructions,
        analysis_input_fingerprint=analysis_input_fingerprint,
        operation_id=operation_id,
        patient_id=patient_id,
        report_id=report_id,
        status="created",
        council_model_ids_json=json.dumps(council_model_ids),
        consolidator_model_id=consolidator_model_id,
        requested_model_ids_json=json.dumps(requested),
        resolved_model_ids_json=json.dumps(resolved),
        creating_instance_id=creating_instance_id,
        model_catalogue_fingerprint=model_catalogue_fingerprint,
        label_map_json="{}",
        started_at=None,
        completed_at=None,
        selected_artifact_id=None,
    )
    session.add(run)
    session.commit()
    session.refresh(run)
    return run


def claim_run_start(session: Session, run_id: str) -> bool:
    result = session.execute(
        update(Run)
        .where(
            Run.id == run_id,
            Run.status.in_(("created", "failed", "needs_auth")),
            Run.start_requested_at.is_(None),
            Run.execution_state.is_(None),
            Run.execution_manifest_hash.is_(None),
            ~select(PostObligation.run_id)
            .where(PostObligation.run_id == Run.id)
            .exists(),
        )
        .values(
            status="running",
            error_message="",
            started_at=_utcnow(),
            completed_at=None,
        )
    )
    session.commit()
    return bool(result.rowcount)


def update_run_status(
    session: Session, run_id: str, *, status: RunStatus, error_message: str = ""
) -> Run | None:
    run = session.get(Run, run_id)
    if run is None:
        return None
    run.status = status
    run.error_message = error_message
    if status == "running" and run.started_at is None:
        run.started_at = _utcnow()
    if status in {"complete", "failed", "needs_auth"}:
        run.completed_at = _utcnow()
    session.commit()
    session.refresh(run)
    return run


def set_run_label_map(session: Session, run_id: str, label_map: dict[str, str]) -> None:
    run = session.get(Run, run_id)
    if run is None:
        return
    run.label_map_json = json.dumps(label_map, sort_keys=True)
    session.commit()


def select_artifact(session: Session, run_id: str, artifact_id: str) -> Run | None:
    run = session.get(Run, run_id)
    if run is None:
        return None
    artifact = session.get(Artifact, artifact_id)
    if artifact is None or artifact.run_id != run_id:
        return None
    run.selected_artifact_id = artifact_id
    session.commit()
    session.refresh(run)
    return run


def create_artifact(
    session: Session,
    *,
    run_id: str,
    stage_num: int,
    stage_name: str,
    model_id: str,
    kind: str,
    content_path: Path,
    content_type: str,
) -> Artifact:
    artifact = Artifact(
        run_id=run_id,
        stage_num=stage_num,
        stage_name=stage_name,
        model_id=model_id,
        kind=kind,
        content_path=str(content_path),
        content_type=content_type,
    )
    session.add(artifact)
    session.commit()
    session.refresh(artifact)
    return artifact


def list_artifacts(session: Session, run_id: str) -> list[Artifact]:
    return list(
        session.scalars(
            select(Artifact)
            .where(Artifact.run_id == run_id)
            .order_by(Artifact.stage_num.asc(), Artifact.created_at.asc())
        )
    )


def get_artifact(session: Session, artifact_id: str) -> Artifact | None:
    return session.get(Artifact, artifact_id)


def find_artifact(
    session: Session,
    *,
    run_id: str,
    stage_num: int,
    model_id: str,
    kind: str,
) -> Artifact | None:
    return session.scalars(
        select(Artifact).where(
            Artifact.run_id == run_id,
            Artifact.stage_num == stage_num,
            Artifact.model_id == model_id,
            Artifact.kind == kind,
        )
    ).first()
