"""Loopback-only original producer calls, excluded from the clinic public bridge."""

import ipaddress
from pathlib import Path
from fastapi import APIRouter, Depends, HTTPException, Request
from . import storage
from .clinic_api import _call, _error, _object_json
from .clinic_catalogue_reads import _envelope, _patient
from .clinic_producers import register_original_output
from .clinic_models import CatalogueConflict, CatalogueNotFound
from .clinic_records import ClinicOperation
from . import clinic_publication as publication
from . import clinic_jobs


def local_producer(request: Request):
    try:
        peer = ipaddress.ip_address(request.client.host)
    except (ValueError, AttributeError):
        raise HTTPException(403, "Local producer access required") from None
    if (
        not peer.is_loopback
        or request.url.hostname not in ("localhost", "127.0.0.1", "::1")
        or "origin" in request.headers
        or "x-clinic-principal" in request.headers
        or any(
            k == "forwarded" or k.startswith("x-forwarded-") for k in request.headers
        )
    ):
        raise HTTPException(403, "Local producer access required")
    if (
        request.method != "GET"
        and request.headers.get("content-type", "").split(";")[0].lower()
        != "application/json"
    ):
        raise HTTPException(415, "JSON producer request required")


router = APIRouter(
    prefix="/api/clinic/internal", dependencies=[Depends(local_producer)]
)


async def object_body(request):
    data = bytearray()
    async for chunk in request.stream():
        data.extend(chunk)
        if len(data) > 1048576:
            raise ValueError("Producer request exceeds limit")
    return _object_json(bytes(data))


@router.get("/publication")
def publication_list(request: Request):
    return _call(
        publication.publication_items,
        request.query_params.get("patientId"),
        limit=request.query_params.get("limit", 100),
        cursor=request.query_params.get("cursor"),
    )


@router.post("/publication/{file_id}/prepare")
async def publication_prepare(file_id: str, request: Request):
    try:
        if await object_body(request) != {}:
            raise ValueError("Expected empty preparation body")
    except ValueError as error:
        return _error(error)
    return _call(publication.prepare_publication, file_id)


@router.post("/publication/{file_id}/verify")
async def publication_verify(file_id: str, request: Request):
    try:
        body = await object_body(request)
        if set(body) != {"remoteKey"} or not isinstance(body["remoteKey"], str):
            raise ValueError("Expected registered remoteKey")
    except ValueError as error:
        return _error(error)
    import asyncio
    import threading
    from .council.execution import drain_task

    stop = threading.Event()
    task = asyncio.create_task(
        asyncio.to_thread(
            _call,
            publication.verify_publication,
            file_id,
            body["remoteKey"],
            stop_event=stop,
        )
    )
    try:
        while not task.done():
            await asyncio.wait({task}, timeout=0.1)
            if await request.is_disconnected():
                stop.set()
        return await task
    finally:
        stop.set()
        await drain_task(task)


def operation_mutation(body, operation_id=None):
    if operation_id is None:
        if set(body) != {"operationId", "patientId", "producer", "kind", "original"}:
            raise ValueError("Expected original operation")
        clinic_jobs.register_operation(
            body["operationId"],
            patient_id=body["patientId"],
            producer=body["producer"],
            kind=body["kind"],
            original=body["original"],
        )
    else:
        if set(body) != {"producer", "generation", "sequence", "payload"}:
            raise ValueError("Expected original receipt")
        clinic_jobs.update_operation(operation_id, **body)
    with storage.session_scope() as s:
        return _envelope(s)


@router.post("/operations")
async def operation_register(request: Request):
    try:
        body = await object_body(request)
    except ValueError as error:
        return _error(error)
    return _call(operation_mutation, body)


@router.post("/operations/{operation_id}")
async def operation_update(operation_id: str, request: Request):
    try:
        body = await object_body(request)
    except ValueError as error:
        return _error(error)
    return _call(operation_mutation, body, operation_id)


def producer_artifact(body):
    from .portal_sync import portal_patients_dir

    required = {
        "patientId",
        "operationId",
        "outputId",
        "relativePath",
        "originalName",
        "logicalFamily",
    }
    optional = {
        "documentKind",
        "sessionDate",
        "generatedAt",
        "provenance",
        "expectedSha256",
        "expectedSize",
    }
    if set(body) - required - optional or not required <= set(body):
        raise ValueError("Expected original output binding")
    if any(not isinstance(body[k], str) or not body[k] for k in required):
        raise ValueError("Invalid output identity")
    if "expectedSha256" in body or "expectedSize" in body:
        import re

        if (
            not isinstance(body.get("expectedSha256"), str)
            or not re.fullmatch(r"[a-f0-9]{64}", body["expectedSha256"])
            or type(body.get("expectedSize")) is not int
            or body["expectedSize"] < 0
        ):
            raise ValueError("Expected SHA-256 and size must be supplied together")
    with storage.session_scope() as s:
        patient = _patient(s, body["patientId"])
        operation = s.get(ClinicOperation, body["operationId"])
        if operation is None:
            raise CatalogueNotFound("Original operation not registered")
        if operation.patient_uuid != patient.id:
            raise CatalogueConflict("Operation belongs to another chart")
        patient_uuid = patient.id
        root = (portal_patients_dir() / patient.label).resolve(strict=True)
    relative = Path(body["relativePath"])
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("Invalid original relative path")
    path = (root / relative).resolve(strict=True)
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError("Output escapes patient directory")
    provenance = body.get("provenance", {})
    if not isinstance(provenance, dict):
        raise ValueError("Invalid original provenance")
    artifact = register_original_output(
        patient_uuid=patient_uuid,
        source_kind=operation.producer,
        source_id=__import__("json").dumps(
            [body["operationId"], body["outputId"]], separators=(",", ":")
        ),
        original_name=body["originalName"],
        logical_family=body["logicalFamily"],
        path=path,
        expected_sha256=body.get("expectedSha256"),
        expected_size=body.get("expectedSize"),
        document_kind=body.get("documentKind"),
        session_date=body.get("sessionDate"),
        generated_at=body.get("generatedAt"),
        provenance=dict(
            original=provenance, operationId=operation.id, outputId=body["outputId"]
        ),
    )
    with storage.session_scope() as s:
        return _envelope(s, artifact=artifact)


@router.post("/artifacts")
async def artifact_register(request: Request):
    try:
        body = await object_body(request)
    except ValueError as error:
        return _error(error)
    return _call(producer_artifact, body)
