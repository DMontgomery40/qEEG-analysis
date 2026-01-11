from __future__ import annotations

import json
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Literal

from sqlalchemy import DateTime, ForeignKey, String, Text, create_engine, select
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, relationship

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
    label: Mapped[str] = mapped_column(String, nullable=False)
    notes: Mapped[str] = mapped_column(Text, nullable=False, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)

    reports: Mapped[list["Report"]] = relationship(back_populates="patient")
    runs: Mapped[list["Run"]] = relationship(back_populates="patient")


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


class Run(Base):
    __tablename__ = "runs"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_new_id)
    patient_id: Mapped[str] = mapped_column(ForeignKey("patients.id"), nullable=False)
    report_id: Mapped[str] = mapped_column(ForeignKey("reports.id"), nullable=False)
    status: Mapped[str] = mapped_column(String, nullable=False, default="created")
    council_model_ids_json: Mapped[str] = mapped_column(Text, nullable=False, default="[]")
    consolidator_model_id: Mapped[str] = mapped_column(String, nullable=False, default="")
    label_map_json: Mapped[str] = mapped_column(Text, nullable=False, default="{}")
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    selected_artifact_id: Mapped[str | None] = mapped_column(String, nullable=True)
    error_message: Mapped[str] = mapped_column(Text, nullable=False, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)

    patient: Mapped[Patient] = relationship(back_populates="runs")
    report: Mapped[Report] = relationship(back_populates="runs")
    artifacts: Mapped[list["Artifact"]] = relationship(back_populates="run")


class Artifact(Base):
    __tablename__ = "artifacts"

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


engine = create_engine(
    f"sqlite:///{get_db_path()}",
    future=True,
    connect_args={"check_same_thread": False},
)


def reset_engine(db_url: str) -> None:
    """
    Test helper: point storage at a different SQLite URL (e.g. sqlite:////tmp/test.db).
    """
    global engine
    try:
        engine.dispose()
    except Exception:
        pass
    engine = create_engine(db_url, future=True, connect_args={"check_same_thread": False})


def init_db() -> None:
    ensure_data_dirs()
    Base.metadata.create_all(engine)


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


def get_patient(session: Session, patient_id: str) -> Patient | None:
    return session.get(Patient, patient_id)


def create_patient(session: Session, *, label: str, notes: str = "") -> Patient:
    patient = Patient(label=label, notes=notes)
    session.add(patient)
    session.commit()
    session.refresh(patient)
    return patient


def update_patient(session: Session, patient_id: str, *, label: str, notes: str) -> Patient | None:
    patient = session.get(Patient, patient_id)
    if patient is None:
        return None
    patient.label = label
    patient.notes = notes
    _touch_updated_at(patient)
    session.commit()
    session.refresh(patient)
    return patient


def list_reports(session: Session, patient_id: str) -> list[Report]:
    return list(
        session.scalars(
            select(Report).where(Report.patient_id == patient_id).order_by(Report.created_at.desc())
        )
    )


def create_report(
    session: Session,
    *,
    patient_id: str,
    filename: str,
    mime_type: str,
    stored_path: Path,
    extracted_text_path: Path,
) -> Report:
    report = Report(
        patient_id=patient_id,
        filename=filename,
        mime_type=mime_type,
        stored_path=str(stored_path),
        extracted_text_path=str(extracted_text_path),
    )
    session.add(report)
    session.commit()
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
) -> Run:
    run = Run(
        patient_id=patient_id,
        report_id=report_id,
        status="created",
        council_model_ids_json=json.dumps(council_model_ids),
        consolidator_model_id=consolidator_model_id,
        label_map_json="{}",
        started_at=None,
        completed_at=None,
        selected_artifact_id=None,
    )
    session.add(run)
    session.commit()
    session.refresh(run)
    return run


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
