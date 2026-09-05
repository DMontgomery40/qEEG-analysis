"""Narrow read-only clinic transport; internal registration stays Python-only."""

from __future__ import annotations

import re
from urllib.parse import quote, unquote_to_bytes

from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from sqlalchemy.exc import SQLAlchemyError

from . import clinic_catalogue_reads as catalogue
from .clinic_naming import POLICY, POLICY_REVISION


def trusted_actor(request: Request):
    value = request.headers.get("x-clinic-actor")
    if value is None:
        return None
    try:
        if re.search(r"%(?![0-9a-fA-F]{2})", value):
            raise ValueError()
        actor = unquote_to_bytes(value).decode("utf-8")
        if (
            not actor
            or len(actor.encode()) > 128
            or any(not c.isprintable() for c in actor)
        ):
            raise ValueError()
        principal = request.headers.get("x-clinic-principal")
        if principal not in ("thrylen-service", "workbench"):
            raise ValueError()
        return actor
    except (ValueError, UnicodeError) as error:
        from fastapi import HTTPException

        raise HTTPException(400, "Invalid clinic actor") from error


router = APIRouter(prefix="/api/clinic", dependencies=[Depends(trusted_actor)])


def _headers(payload):
    return {
        "X-Clinic-Schema-Version": "clinic-v1",
        "X-Clinic-Policy-Revision": POLICY_REVISION,
        "X-Clinic-Catalog-Revision": str(payload["catalogRevision"]),
    }


def _error(error):
    if isinstance(error, catalogue.CatalogueNotFound):
        code, message = 404, str(error)
    elif isinstance(error, catalogue.CatalogueConflict):
        code, message = 409, str(error)
    elif isinstance(error, (catalogue.CatalogueUnavailable, SQLAlchemyError, OSError)):
        code, message = 503, "Clinic authority or exact file bytes are unavailable"
    else:
        code, message = 400, str(error)
    return JSONResponse(dict(ok=False, message=message), status_code=code)


def _call(function, *args, **kwargs):
    try:
        payload = function(*args, **kwargs)
        return JSONResponse(payload, headers=_headers(payload))
    except (
        ValueError,
        LookupError,
        catalogue.CatalogueUnavailable,
        SQLAlchemyError,
        OSError,
    ) as error:
        return _error(error)


@router.get("/policy")
def policy():
    return _call(
        lambda: dict(
            ok=True,
            schemaVersion="clinic-v1",
            policyRevision=POLICY_REVISION,
            catalogRevision=catalogue.current_revision(),
            policy=POLICY,
        )
    )


@router.get("/patients")
def patients():
    return _call(catalogue.roster)


@router.get("/patients/{patient_id}")
def patient(patient_id: str):
    return _call(catalogue.roster, patient_id)


@router.get("/patient-files")
def patient_files(request: Request):
    query = request.query_params
    return _call(
        catalogue.patient_files,
        query.get("patientId"),
        mode=query.get("mode", "full"),
        limit=query.get("limit", "500"),
        page=query.get("page"),
        cursor=query.get("cursor"),
        relative_path=query.get("relativePath"),
        sha256=query.get("sha256"),
        if_index_version=query.get("ifIndexVersion"),
    )


@router.post("/patient-report-dates")
async def patient_report_dates(request: Request):
    try:
        raw = await request.body()
        if len(raw) > 100_000:
            raise ValueError("Report-date batch is too large")
        import json

        body = json.loads(raw)
        if not isinstance(body, dict):
            raise ValueError("Expected a report-date batch")
    except ValueError as error:
        return _error(error)
    return _call(catalogue.report_dates, body.get("patientIds"))


@router.get("/file-binding")
def file_binding(request: Request):
    query = request.query_params
    return _call(
        catalogue.file_binding,
        query.get("patientId"),
        file_key=query.get("fileKey"),
        file_id=query.get("fileId"),
    )


def _range(value, size):
    match = re.fullmatch(r"bytes=(\d*)-(\d*)", value)
    if not match or not any(match.groups()) or size == 0:
        raise ValueError("Unsatisfiable byte range")
    first, last = match.groups()
    if not first:
        suffix = int(last)
        if suffix <= 0:
            raise ValueError("Unsatisfiable byte range")
        return max(0, size - suffix), size - 1
    start = int(first)
    end = min(int(last), size - 1) if last else size - 1
    if start > end or start >= size:
        raise ValueError("Unsatisfiable byte range")
    return start, end


class _SnapshotResponse(StreamingResponse):
    def __init__(self, stream, *args, **kwargs):
        self.snapshot = stream
        super().__init__(*args, **kwargs)

    async def __call__(self, scope, receive, send):
        try:
            await super().__call__(scope, receive, send)
        finally:
            self.snapshot.close()


@router.api_route("/file", methods=["GET", "HEAD"])
def file_bytes(request: Request):
    query = request.query_params
    stream = None
    try:
        binding = catalogue.file_binding(
            query.get("patientId"),
            file_key=query.get("fileKey"),
            file_id=query.get("fileId"),
        )
        stream = catalogue.open_local_file(binding["fileId"])
        etag = '"' + binding["sha256"] + '"'
        headers = {
            **_headers(binding),
            "ETag": etag,
            "Accept-Ranges": "bytes",
            "Content-Type": binding["contentType"],
            "Content-Disposition": "attachment; filename=\"patient-file\"; filename*=UTF-8''"
            + quote(binding["downloadName"], safe=""),
        }
        if request.headers.get("if-none-match") in (etag, "*"):
            stream.close()
            return Response(status_code=304, headers=headers)
        size, status, start, end = binding["size"], 200, 0, binding["size"] - 1
        range_header = request.headers.get("range")
        if range_header and request.headers.get("if-range", etag) == etag:
            try:
                start, end = _range(range_header, size)
            except ValueError:
                stream.close()
                return Response(
                    status_code=416,
                    headers={
                        **headers,
                        "Content-Range": f"bytes */{size}",
                        "Content-Length": "0",
                    },
                )
            status = 206
            headers["Content-Range"] = f"bytes {start}-{end}/{size}"
        length = max(0, end - start + 1)
        headers["Content-Length"] = str(length)
        if request.method == "HEAD":
            stream.close()
            return Response(status_code=status, headers=headers)
        stream.seek(start)

        def chunks():
            try:
                remaining = length
                while remaining:
                    data = stream.read(min(1024 * 1024, remaining))
                    if not data:
                        raise OSError("Verified file ended early")
                    remaining -= len(data)
                    yield data
            finally:
                stream.close()

        return _SnapshotResponse(stream, chunks(), status_code=status, headers=headers)
    except (
        ValueError,
        LookupError,
        catalogue.CatalogueUnavailable,
        SQLAlchemyError,
        OSError,
    ) as error:
        if stream:
            stream.close()
        return _error(error)
