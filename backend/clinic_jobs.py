"""Read original engine receipts and registered supervised operation references."""

import json
from sqlalchemy import select
from . import storage
from .clinic_catalogue import _write, _bump
from .clinic_catalogue_reads import _patient, _envelope, _json, _millis
from .clinic_records import ClinicOperation, ClinicUpload
from .clinic_models import CatalogueConflict, CatalogueNotFound
from .clinic_intake import require_key, _upload_json


def register_operation(operation_id, *, patient_id, producer, kind, original):
    require_key(operation_id)
    if (
        producer not in ("workbench", "renderer")
        or kind not in ("video", "analysis", "patient-summary")
        or not isinstance(original, dict)
        or not original
    ):
        raise ValueError("Original supervised operation reference required")
    with _write() as s:
        p = _patient(s, patient_id)
        row = s.get(ClinicOperation, operation_id)
        if row:
            if (row.patient_uuid, row.producer, row.kind, row.original_json) != (
                p.id,
                producer,
                kind,
                _json(original),
            ):
                raise CatalogueConflict("Original operation identity changed")
            return
        s.add(
            ClinicOperation(
                id=operation_id,
                patient_uuid=p.id,
                producer=producer,
                kind=kind,
                original_json=_json(original),
                generation=0,
                sequence=0,
                payload_json=_json({"status": "unknown"}),
            )
        )
        _bump(s, p.id)


def update_operation(operation_id, *, producer, generation, sequence, payload):
    if (
        type(generation) is not int
        or type(sequence) is not int
        or min(generation, sequence) < 0
        or not isinstance(payload, dict)
        or payload.get("status")
        not in (
            "unknown",
            "pending",
            "running",
            "complete",
            "failed",
            "cancelled",
            "needs_operator_answer",
        )
    ):
        raise ValueError("Invalid producer receipt")
    if set(payload) - {
        "status",
        "fileId",
        "runId",
        "detail",
        "clinicalState",
        "generationState",
        "deliveryState",
    }:
        raise ValueError("Unsupported producer payload")
    with _write() as s:
        row = s.get(ClinicOperation, operation_id)
        if not row:
            raise CatalogueNotFound("Original operation is not registered")
        if row.producer != producer:
            raise CatalogueConflict("Operation producer differs")
        incoming = (generation, sequence)
        prior = (row.generation, row.sequence)
        if incoming < prior:
            return False
        if incoming == prior:
            if row.payload_json != _json(payload):
                raise CatalogueConflict("Producer sequence binds different evidence")
            return False
        old = json.loads(row.payload_json)
        if old["status"] in ("complete", "failed", "cancelled") and payload[
            "status"
        ] not in ("complete", "failed", "cancelled"):
            raise CatalogueConflict("Terminal operation evidence cannot regress")
        if payload.get("runId"):
            run = s.get(storage.Run, payload["runId"])
            if not run or run.patient_id != row.patient_uuid:
                raise CatalogueConflict("Run belongs to another chart or is unknown")
        if payload.get("fileId"):
            from .clinic_models import ClinicArtifact

            f = s.get(ClinicArtifact, payload["fileId"])
            if not f or f.patient_uuid != row.patient_uuid:
                raise CatalogueConflict("File belongs to another chart or is unknown")
        row.generation = generation
        row.sequence = sequence
        row.payload_json = _json(payload)
        _bump(s, row.patient_uuid)
        return True


def patient_jobs(patient_id):
    with storage.session_scope() as s:
        p = _patient(s, patient_id)
        jobs = []
        operation_ids = set()
        for run in s.scalars(
            select(storage.Run)
            .where(storage.Run.patient_id == p.id)
            .order_by(storage.Run.created_at, storage.Run.id)
        ):
            operation_ids.add(run.operation_id)
            stages = [
                dict(
                    stage=r.stage_num,
                    receiptHash=r.receipt_hash,
                    policyVersion=r.policy_version,
                    completedAt=_millis(r.completed_at),
                )
                for r in s.scalars(
                    select(storage.StageReceipt)
                    .where(storage.StageReceipt.run_id == run.id)
                    .order_by(storage.StageReceipt.stage_num)
                )
            ]
            post = [
                dict(
                    kind=r.kind,
                    state=r.state,
                    receiptHash=r.receipt_hash,
                    blockedReason=r.blocked_reason,
                )
                for r in s.scalars(
                    select(storage.PostObligation)
                    .where(storage.PostObligation.run_id == run.id)
                    .order_by(storage.PostObligation.kind)
                )
            ]
            jobs.append(
                dict(
                    jobId=run.id,
                    runId=run.id,
                    operationId=run.operation_id,
                    kind="analysis",
                    producer="engine",
                    status=run.status,
                    executionState=run.execution_state,
                    clinicalState=run.status,
                    stages=stages,
                    post=post,
                )
            )
        for row in s.scalars(
            select(ClinicOperation)
            .where(ClinicOperation.patient_uuid == p.id)
            .order_by(ClinicOperation.id)
        ):
            jobs.append(
                dict(
                    jobId=row.id,
                    operationId=row.id,
                    kind=row.kind,
                    producer=row.producer,
                    generation=row.generation,
                    sequence=row.sequence,
                    original=json.loads(row.original_json),
                    **json.loads(row.payload_json),
                )
            )
        for u in s.scalars(
            select(ClinicUpload).where(
                ClinicUpload.patient_uuid == p.id,
                ClinicUpload.analysis_json.is_not(None),
            )
        ):
            a = _upload_json(s, u)["analysis"]
            if a["operationId"] not in operation_ids:
                jobs.append(
                    dict(
                        jobId=a["operationId"],
                        kind="analysis",
                        producer="engine",
                        uploadId=u.id,
                        **a,
                    )
                )
        return _envelope(s, jobs=jobs)
