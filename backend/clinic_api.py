"""Narrow read-only clinic transport; internal registration stays Python-only."""

from __future__ import annotations

import re
from urllib.parse import quote, unquote_to_bytes

from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from sqlalchemy.exc import SQLAlchemyError

from . import clinic_catalogue_reads as catalogue
from .clinic_naming import POLICY_REVISION


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
    from .clinic_analysis_intents import public_current_policy

    return _call(
        lambda: dict(
            ok=True,
            schemaVersion="clinic-v1",
            policyRevision=POLICY_REVISION,
            catalogRevision=catalogue.current_revision(),
            **public_current_policy(),
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


@router.get("/uploads")
def clinic_uploads():
    from .clinic_intake import list_uploads

    return _call(list_uploads)


@router.get("/uploads/{upload_id}")
def clinic_upload(upload_id: str):
    from .clinic_intake import get_upload

    return _call(get_upload, upload_id)


def _object_json(raw):
    import json

    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("Duplicate JSON field")
            result[key] = value
        return result

    value = json.loads(
        raw,
        object_pairs_hook=unique,
        parse_constant=lambda _: (_ for _ in ()).throw(
            ValueError("Invalid JSON number")
        ),
    )
    if not isinstance(value, dict):
        raise ValueError("Expected JSON object")
    return value


@router.post("/uploads")
async def clinic_upload_submit(request: Request):
    from starlette.concurrency import run_in_threadpool
    from starlette.datastructures import UploadFile
    from .clinic_intake import submit_upload, require_key

    form = None
    try:
        key = require_key(request.headers.get("idempotency-key"))
        form = await request.form()
        allowed = {
            "files",
            "fileMeta",
            "patientId",
            "firstName",
            "lastName",
            "firstInitial",
            "lastInitial",
            "birthdate",
            "resolution",
            "submissionId",
            "analysisIntent",
        }
        if set(form) - allowed:
            raise ValueError("Unsupported upload field")
        if any(len(form.getlist(k)) != 1 for k in set(form) - {"files", "fileMeta"}):
            raise ValueError("Repeated singleton upload field")
        if form.get("submissionId", key) != key:
            raise ValueError("submissionId must equal Idempotency-Key")
        uploads = form.getlist("files")
        metas = form.getlist("fileMeta")
        if (
            not uploads
            or len(uploads) != len(metas)
            or not all(isinstance(f, UploadFile) for f in uploads)
        ):
            raise ValueError("Each file requires ordered fileMeta")
        files = [
            (f.filename, await f.read(), f.content_type or "application/octet-stream")
            for f in uploads
        ]
        kwargs = dict(
            key=key,
            identity={
                k: form[k]
                for k in (
                    "firstName",
                    "lastName",
                    "firstInitial",
                    "lastInitial",
                    "birthdate",
                )
                if k in form
            },
            files=files,
            file_meta=[_object_json(m) for m in metas],
            actor=trusted_actor(request),
            principal=request.headers.get("x-clinic-principal"),
            patient_id=form.get("patientId"),
            resolution=_object_json(form["resolution"])
            if "resolution" in form
            else None,
            analysis_intent=_object_json(form["analysisIntent"])
            if "analysisIntent" in form
            else None,
        )
        return await run_in_threadpool(_call, submit_upload, **kwargs)
    except (ValueError, TypeError) as error:
        return _error(error)
    finally:
        if form is not None:
            await form.close()


@router.post("/uploads/{upload_id}/resolution")
async def clinic_upload_resolve(upload_id: str, request: Request):
    from starlette.concurrency import run_in_threadpool
    from .clinic_intake import resolve_upload

    try:
        body = _object_json(await request.body())
    except (ValueError, TypeError) as error:
        return _error(error)
    return await run_in_threadpool(
        _call,
        resolve_upload,
        upload_id,
        key=request.headers.get("idempotency-key"),
        resolution=body,
        actor=trusted_actor(request),
    )


@router.patch("/patients/{patient_id}")
async def clinic_patient_patch(patient_id: str, request: Request):
    from .clinic_patient_updates import patch_patient

    try:
        body = _object_json(await request.body())
    except (ValueError, TypeError) as error:
        return _error(error)
    return _call(
        patch_patient,
        patient_id,
        key=request.headers.get("idempotency-key"),
        changes=body,
        actor=trusted_actor(request),
    )


@router.post("/feedback")
async def clinic_feedback(request: Request):
    from .clinic_feedback import record_feedback

    try:
        body = _object_json(await request.body())
        if set(body) - {"patientId", "fileId", "version", "action", "notes"} or not {
            "patientId",
            "fileId",
            "version",
            "action",
        } <= set(body):
            raise ValueError("Exact patient, file, version and action required")
    except (ValueError, TypeError) as error:
        return _error(error)
    return _call(
        record_feedback,
        key=request.headers.get("idempotency-key"),
        patient_id=body["patientId"],
        file_id=body["fileId"],
        version=body["version"],
        action=body["action"],
        notes=body.get("notes", ""),
        actor=trusted_actor(request),
        principal=request.headers.get("x-clinic-principal"),
    )


@router.get("/jobs")
def clinic_jobs(request: Request):
    from .clinic_jobs import patient_jobs

    return _call(patient_jobs, request.query_params.get("patientId"))


@router.post("/identity-preview")
async def clinic_identity_preview(request: Request):
    from .clinic_identity_preview import preview_identities

    try:
        body = _object_json(await request.body())
    except (ValueError, TypeError) as error:
        return _error(error)
    return _call(preview_identities, body)


def _notification_actor(request):
    from fastapi import HTTPException

    if request.headers.get(
        "x-clinic-principal"
    ) != "thrylen-service" or not trusted_actor(request):
        raise HTTPException(403, "Thrylen notification receipt required")


@router.post("/feedback/{event_id}/notification/claim")
async def clinic_notification_claim(event_id: str, request: Request):
    from .clinic_feedback import claim_notification

    _notification_actor(request)
    try:
        body = _object_json(await request.body())
        if (
            set(body) != {"claimId"}
            or request.headers.get("idempotency-key") != body["claimId"]
        ):
            raise ValueError("Expected matching notification attempt key")
    except ValueError as error:
        return _error(error)
    return _call(claim_notification, event_id, claim_id=body["claimId"])


@router.post("/feedback/{event_id}/notification")
async def clinic_notification_receipt(event_id: str, request: Request):
    from .clinic_feedback import record_notification

    _notification_actor(request)
    try:
        body = _object_json(await request.body())
        if (
            set(body) - {"claimId", "status", "detail"}
            or not {"claimId", "status"} <= set(body)
            or body["status"] not in ("sent", "failed", "unknown")
        ):
            raise ValueError("Expected original notification outcome")
        if request.headers.get("idempotency-key") != body["claimId"]:
            raise ValueError("Expected matching notification attempt key")
        detail = body.get("detail", "")
        if not isinstance(detail, str) or len(detail.encode()) > 2048:
            raise ValueError("Invalid notification detail")
    except (ValueError, TypeError) as error:
        return _error(error)
    return _call(
        record_notification,
        event_id,
        claim_id=body["claimId"],
        status=body["status"],
        detail=detail,
    )


@router.get("/recent-files")
def clinic_recent_files(request: Request):
    from .clinic_recent_files import recent_files

    query = request.query_params
    if set(query) - {"kind", "contentType", "limit"}:
        return _error(ValueError("Unsupported recent file query"))
    return _call(
        recent_files,
        kind=query.get("kind"),
        content_type=query.get("contentType"),
        limit=query.get("limit", "30"),
    )
