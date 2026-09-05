"""Opt-in council result, artifact and six-stage completion receipts.

The retained RunOwner fences short file/metadata operations. Network awaits stay
in the original stage implementation. Files are immutable and legacy rows never
constitute a completed owned stage.
"""

from __future__ import annotations

import base64
from functools import wraps
import inspect
import json
from pathlib import Path
from uuid import NAMESPACE_URL, uuid5

from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError
from datetime import datetime, timezone

from .. import storage
from ..run_execution import ExecutionConflict
from .execution import current_execution, _hash, _json, _publish
from .paths import _artifact_path
from .constants import STAGES

POLICY_VERSION = "council-six-stages-v1"
_ROLES = {
    1: "member",
    2: "reviewer",
    3: "member",
    4: "consolidation",
    5: "reviewer",
    6: "writer",
}


def _storage_guarded(function):
    @wraps(function)
    def wrapped(*args, **kwargs):
        try:
            return function(*args, **kwargs)
        except (OSError, SQLAlchemyError) as error:
            raise ExecutionConflict(
                "completion persistence failed; acknowledged work remains recoverable"
            ) from error

    return wrapped


def artifact_order(artifact):
    key = artifact.operation_key or ""
    parts = key.split("/")
    index = int(parts[2]) if len(parts) > 2 and parts[2].isdigit() else -1
    return (artifact.stage_num, bool(key), index, key, artifact.created_at, artifact.id)


def project_run_status(session, run_id, *, status, error_message=""):
    ctx = current_execution()
    if ctx is None:
        return storage.update_run_status(
            session, run_id, status=status, error_message=error_message
        )
    if run_id != ctx.owner.run_id:
        raise ExecutionConflict("run status projection belongs to another owner")
    with ctx.owner.transaction() as owned_session:
        run = owned_session.get(storage.Run, run_id)
        run.status, run.error_message = status, error_message
        if status == "running" and run.started_at is None:
            run.started_at = datetime.now(timezone.utc)
        if status in {"complete", "failed", "needs_auth"}:
            run.completed_at = datetime.now(timezone.utc)
    return run


def project_label_map(session, run_id, label_map):
    ctx = current_execution()
    if ctx is None:
        return storage.set_run_label_map(session, run_id, label_map)
    if run_id != ctx.owner.run_id:
        raise ExecutionConflict("label projection belongs to another owner")
    with ctx.owner.transaction() as owned_session:
        owned_session.get(storage.Run, run_id).label_map_json = json.dumps(
            label_map, sort_keys=True
        )


def _binding(key):
    ctx = current_execution()
    return {
        "key": key,
        "run_id": ctx.owner.run_id,
        "execution_manifest_hash": ctx.manifest_hash,
        "input_fingerprint": ctx.manifest["admission"]["analysis_input_fingerprint"],
    }


def _path(key):
    return (
        current_execution().manifest_path.parent
        / "completion"
        / (_hash(key.encode()) + ".json")
    )


def _file_hash(path):
    try:
        return _hash(Path(path).read_bytes())
    except OSError as error:
        raise ExecutionConflict(
            "required completion file is missing or unreadable"
        ) from error


@_storage_guarded
def _paid(prefix, *, exact=False):
    """Snapshot acknowledged original requests, outside any file IO transaction."""
    ctx = current_execution()
    with ctx.owner.transaction() as session:
        rows = list(
            session.scalars(
                select(storage.PaidRequest)
                .where(storage.PaidRequest.run_id == ctx.owner.run_id)
                .order_by(
                    storage.PaidRequest.scope_key, storage.PaidRequest.dispatch_ordinal
                )
            )
        )
        fields = (
            "scope_key",
            "dispatch_ordinal",
            "state",
            "request_path",
            "request_hash",
            "route_json",
            "execution_manifest_hash",
            "input_fingerprint",
            "response_path",
            "response_hash",
            "http_status",
            "response_metadata_json",
            "error_classification",
        )
        result = [
            {f: getattr(row, f) for f in fields}
            for row in rows
            if row.scope_key == prefix
            or (not exact and row.scope_key.startswith(prefix + "/"))
        ]
    for row in result:
        if row["state"] not in {"response_saved", "rejected"}:
            raise ExecutionConflict(
                "unfinished paid work cannot acknowledge a semantic unit"
            )
        for stem in ("request", "response"):
            if (
                not row[stem + "_path"]
                or _file_hash(row[stem + "_path"]) != row[stem + "_hash"]
            ):
                raise ExecutionConflict("original paid receipt bytes changed")
    return result


@_storage_guarded
def _read(key):
    path = _path(key)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_bytes())
    except (OSError, ValueError) as error:
        raise ExecutionConflict("invalid completion record") from error
    if data.get("binding") != _binding(key):
        raise ExecutionConflict("completion identity changed")
    return data


@_storage_guarded
def _save(key, **fields):
    record = {"binding": _binding(key), **fields}
    _publish(current_execution().owner, _path(key), _json(record))
    return record


@_storage_guarded
def record_semantic_result(key, result):
    """Bind successful parse/repair output to its complete original paid sequence.

    Re-entry still executes the original coroutine through E2 replay, preserving
    its exact request validation and local mutation/validation recipe.
    """
    if current_execution() is not None and "/sdk-request/" not in key:
        if hasattr(result, "model_dump"):
            result = result.model_dump(mode="json")
        _save("semantic/" + key, result=result, paid=_paid(key))


def save_product(kind, text):
    from .types import StageDef
    from .paths import _data_pack_path, _vision_transcript_path

    ctx = current_execution()
    if ctx is None:
        return
    json_product = kind == "data_pack"
    stage = StageDef(
        1,
        STAGES[0].name,
        kind,
        "application/json" if json_product else "text/markdown",
        ".json" if json_product else ".md",
    )
    prefix = "s1/data-pack" if json_product else "s1/transcript"
    # Product scope includes all extractor candidates and targeted repairs.
    return save_artifact(
        stage=stage,
        model_id="_" + kind,
        text=text,
        operation_key=prefix,
        path=(_data_pack_path if json_product else _vision_transcript_path)(
            ctx.owner.run_id
        ),
    )


def load_product(kind):
    ctx = current_execution()
    if ctx is None:
        return None
    prefix = "s1/data-pack" if kind == "data_pack" else "s1/transcript"
    record = _read("artifact/" + prefix)
    if record is None:
        return None
    if record["paid"] != _paid(prefix):
        raise ExecutionConflict("extraction product paid bindings changed")
    with ctx.owner.transaction() as session:
        completed = session.get(storage.StageReceipt, (ctx.owner.run_id, 1)) is not None
    row = _reconcile_artifact(
        record, reconstruct=not completed and not _path("stage/1").exists()
    )
    return Path(row.content_path).read_text(encoding="utf-8")


def member_key(stage_num, index, model_id):
    if stage_num == 4:
        return "s4/consolidation"
    return f"s{stage_num}/{_ROLES[stage_num]}/{index}/{model_id}"


def _artifact_metadata(artifact):
    return {
        name: getattr(artifact, name)
        for name in (
            "id",
            "run_id",
            "operation_key",
            "stage_num",
            "stage_name",
            "model_id",
            "kind",
            "content_path",
            "content_type",
        )
    }


def _artifact_target(stage, index, model_id):
    ctx = current_execution()
    models = (
        ctx.roles["writers"]
        if stage.num == 6
        else [ctx.manifest["admission"]["consolidator_model_id"]]
        if stage.num == 4
        else json.loads(ctx.manifest["admission"]["council_model_ids_json"])
    )
    target = _artifact_path(ctx.owner.run_id, stage.num, model_id, stage.ext)
    defaults = [
        _artifact_path(ctx.owner.run_id, stage.num, m, stage.ext) for m in models
    ]
    # The first member keeps the original path. Every later duplicate or sanitized
    # collision gets a suffix absent from ALL original paths and earlier suffixes.
    occupied = set()
    for i, default in enumerate(defaults[: index + 1]):
        candidate = default
        ordinal = 0
        while candidate in occupied:
            suffix = f".member-{i}" + (f"-{ordinal}" if ordinal else "")
            candidate = default.with_name(default.stem + suffix + default.suffix)
            ordinal += 1
            if candidate in defaults:
                occupied.add(candidate)
        occupied.add(candidate)
        if i == index:
            target = candidate

    return target


@_storage_guarded
def _reconcile_artifact(record, *, reconstruct):
    ctx = current_execution()
    metadata = record["artifact"]
    content = base64.b64decode(record["content_base64"], validate=True)
    if _hash(content) != record["content_hash"]:
        raise ExecutionConflict("unit artifact content hash changed")
    path = Path(metadata["content_path"])
    if not reconstruct and _file_hash(path) != record["content_hash"]:
        raise ExecutionConflict("completed artifact hash changed")
    if reconstruct:
        _publish(ctx.owner, path, content)
    if _file_hash(path) != record["content_hash"]:
        raise ExecutionConflict("artifact hash changed")
    with ctx.owner.transaction() as session:
        row = session.scalar(
            select(storage.Artifact).where(
                storage.Artifact.run_id == ctx.owner.run_id,
                storage.Artifact.operation_key == metadata["operation_key"],
            )
        )
        if row is None:
            if not reconstruct:
                raise ExecutionConflict("completed artifact registration missing")
            row = storage.Artifact(**metadata)
            session.add(row)
            session.flush()
        elif _artifact_metadata(row) != metadata:
            raise ExecutionConflict("artifact registration changed")
    return row


@_storage_guarded
def save_artifact(*, stage, model_id, text, operation_key, path=None):
    """File intent -> immutable artifact -> unique row; every replay validates bytes."""
    ctx = current_execution()
    key = "artifact/" + operation_key
    metadata = dict(
        id=str(uuid5(NAMESPACE_URL, ctx.manifest_hash + "/" + operation_key)),
        run_id=ctx.owner.run_id,
        operation_key=operation_key,
        stage_num=stage.num,
        stage_name=stage.name,
        model_id=model_id,
        kind=stage.kind,
        content_path=str(
            path or _artifact_path(ctx.owner.run_id, stage.num, model_id, stage.ext)
        ),
        content_type=stage.content_type,
    )
    content = text.encode("utf-8")
    record = _save(
        key,
        artifact=metadata,
        content_base64=base64.b64encode(content).decode(),
        content_hash=_hash(content),
        paid=_paid(operation_key),
    )
    return _reconcile_artifact(record, reconstruct=True)


@_storage_guarded
def finish_member(stage, index, model_id, text):
    """Save a successful validated member before any completion event await."""
    ctx = current_execution()
    if ctx is None:
        return
    key = member_key(stage.num, index, model_id)
    row = save_artifact(
        stage=stage,
        model_id=model_id,
        text=text,
        operation_key=key,
        path=_artifact_target(stage, index, model_id),
    )
    _save(
        "member/" + key,
        disposition="success",
        artifact_id=row.id,
        artifact_record_hash=_file_hash(_path("artifact/" + key)),
        paid=_paid(key),
    )


def _load_member(stage, index, model_id, *, reconstruct=True):
    key = member_key(stage.num, index, model_id)
    record = _read("member/" + key)
    # Death after file/row but before the member acknowledgement: the immutable
    # intent is sufficient proof to finish registration, never a new paid call.
    artifact = _read("artifact/" + key)
    if record is None and artifact is not None and reconstruct:
        if artifact["paid"] != _paid(key):
            raise ExecutionConflict("member original paid sequence changed")
        row = _reconcile_artifact(artifact, reconstruct=True)
        record = _save(
            "member/" + key,
            disposition="success",
            artifact_id=row.id,
            artifact_record_hash=_file_hash(_path("artifact/" + key)),
            paid=_paid(key),
        )
    if record is None:
        return False, None
    if record["paid"] != _paid(key):
        raise ExecutionConflict("member paid receipt binding changed")
    if record["disposition"] == "failed":
        if artifact is not None:
            raise ExecutionConflict("failed member has an accepted artifact")
        return True, None
    if record["disposition"] != "success" or artifact is None:
        raise ExecutionConflict("member disposition is invalid")
    if _file_hash(_path("artifact/" + key)) != record["artifact_record_hash"]:
        raise ExecutionConflict("member artifact receipt changed")
    row = _reconcile_artifact(artifact, reconstruct=reconstruct)
    if record["artifact_id"] != row.id:
        raise ExecutionConflict("member artifact identity changed")
    return True, (model_id, Path(row.content_path).read_text(encoding="utf-8"))


def durable_member(stage):
    def decorate(function):
        @wraps(function)
        async def wrapped(index, model_id):
            ctx = current_execution()
            if ctx is None:
                return await function(index, model_id)
            found, result = _load_member(stage, index, model_id)
            if found:
                return result
            result = await function(index, model_id)
            key = member_key(stage.num, index, model_id)
            if result is None:
                _save("member/" + key, disposition="failed", paid=_paid(key))
            else:
                finish_member(stage, index, model_id, result[1])
            return result

        return wrapped

    return decorate


def _inventory(stage_num):
    ctx = current_execution()
    with ctx.owner.transaction() as session:
        rows = list(
            session.scalars(
                select(storage.Artifact).where(
                    storage.Artifact.run_id == ctx.owner.run_id,
                    storage.Artifact.stage_num < stage_num,
                )
            )
        )
    rows.sort(key=artifact_order)
    return [{**_artifact_metadata(a), "hash": _file_hash(a.content_path)} for a in rows]


def _plan(stage_num):
    ctx = current_execution()
    artifacts = _inventory(stage_num)
    council = json.loads(ctx.manifest["admission"]["council_model_ids_json"])
    available = {
        a["model_id"]
        for a in artifacts
        if a["stage_num"] == 1 and a["kind"] == "analysis"
    }
    models = (
        ctx.roles["writers"]
        if stage_num == 6
        else [ctx.manifest["admission"]["consolidator_model_id"]]
        if stage_num == 4
        else council
    )
    members = [
        {"index": i, "model_id": m}
        for i, m in enumerate(models)
        if stage_num not in (2, 3) or m in available
    ]
    skip = stage_num == 2 and len(members) < 2
    return dict(
        policy_version=POLICY_VERSION,
        stage_num=stage_num,
        inputs=artifacts,
        members=members,
        requested_count=1 if stage_num in (4, 6) else len(members),
        skipped=skip,
        reason="Not enough Stage 1 analyses for peer review" if skip else None,
    )


def _sources(stage_num):
    ctx = current_execution()
    result = []
    for path in sorted((ctx.manifest_path.parent / "sources").glob("*.json")):
        try:
            record = json.loads(path.read_bytes())
        except (OSError, ValueError) as error:
            raise ExecutionConflict("source binding is unreadable") from error
        if record.get("key", "").startswith(f"s{stage_num}/"):
            result.append({"path": str(path), "hash": _file_hash(path)})
    return result


def _outcomes(stage_num, plan, *, reconstruct):
    results = []
    success = False
    for member in plan["members"]:
        index, model = member["index"], member["model_id"]
        if plan["skipped"] or (stage_num == 6 and success):
            results.append({**member, "disposition": "not_requested"})
            continue
        found, result = _load_member(
            STAGES[stage_num - 1], index, model, reconstruct=reconstruct
        )
        if not found:
            raise ExecutionConflict("stage contains an unfinished member")
        key = member_key(stage_num, index, model)
        record = _read("member/" + key)
        success = result is not None
        results.append(
            {
                **member,
                "disposition": record["disposition"],
                "receipt_path": str(_path("member/" + key)),
                "receipt_hash": _file_hash(_path("member/" + key)),
            }
        )
    count = sum(r["disposition"] == "success" for r in results)
    if stage_num in (1, 3, 4, 5, 6) and not count:
        raise ExecutionConflict("stage success policy is not satisfied")
    if stage_num in (4, 6) and count != 1:
        raise ExecutionConflict("single-writer stage success policy changed")
    return results, count


def _stage_record(stage_num, plan, *, reconstruct):
    outcomes, count = _outcomes(stage_num, plan, reconstruct=reconstruct)
    ctx = current_execution()
    with ctx.owner.transaction() as session:
        rows = list(
            session.scalars(
                select(storage.Artifact).where(
                    storage.Artifact.run_id == ctx.owner.run_id,
                    storage.Artifact.stage_num == stage_num,
                    storage.Artifact.operation_key.is_not(None),
                )
            )
        )
    artifacts = []
    for artifact in rows:
        key = "artifact/" + artifact.operation_key
        intent = _read(key)
        if intent is None or intent["paid"] != _paid(artifact.operation_key):
            raise ExecutionConflict("stage artifact original receipt binding changed")
        _reconcile_artifact(intent, reconstruct=False)
        if intent["artifact"] != _artifact_metadata(artifact):
            raise ExecutionConflict("stage artifact identity changed")
        artifacts.append(
            {
                **_artifact_metadata(artifact),
                "hash": _file_hash(artifact.content_path),
                "receipt_path": str(_path(key)),
                "receipt_hash": _file_hash(_path(key)),
            }
        )
    semantic_results = []
    for path in sorted((ctx.manifest_path.parent / "completion").glob("*.json")):
        try:
            key = json.loads(path.read_bytes()).get("binding", {}).get("key", "")
        except (OSError, ValueError) as error:
            raise ExecutionConflict("completion receipt cannot be read") from error
        if key.startswith(f"semantic/s{stage_num}/"):
            record = _read(key)
            if record is None or record["paid"] != _paid(key.removeprefix("semantic/")):
                raise ExecutionConflict(
                    "semantic result original receipt binding changed"
                )
            semantic_results.append({"path": str(path), "hash": _file_hash(path)})

    artifacts.sort(key=lambda a: a["operation_key"])
    with ctx.owner.transaction() as session:
        label_map = (
            json.loads(
                session.get(storage.Run, ctx.owner.run_id).label_map_json or "{}"
            )
            if stage_num == 2
            else None
        )
    return dict(
        plan=plan,
        outcomes=outcomes,
        artifacts=artifacts,
        label_map=label_map,
        sources=_sources(stage_num),
        semantic_results=semantic_results,
        plan_hash=_file_hash(_path(f"plan/{stage_num}")),
        success_count=count,
        requested_count=plan["requested_count"],
        skipped=plan["skipped"],
        reason=plan["reason"],
    )


def _commit_stage(stage_num, plan):
    ctx = current_execution()
    key = f"stage/{stage_num}"
    from .execution import raise_if_execution_blocked

    raise_if_execution_blocked()
    fields = _stage_record(stage_num, plan, reconstruct=False)
    _save(key, **fields)
    values = dict(
        receipt_path=str(_path(key)),
        receipt_hash=_file_hash(_path(key)),
        execution_manifest_hash=ctx.manifest_hash,
        input_fingerprint=ctx.manifest["admission"]["analysis_input_fingerprint"],
        policy_version=POLICY_VERSION,
    )
    with ctx.owner.transaction() as session:
        row = session.get(storage.StageReceipt, (ctx.owner.run_id, stage_num))
        if row is None:
            session.add(
                storage.StageReceipt(
                    run_id=ctx.owner.run_id,
                    stage_num=stage_num,
                    owner_token=ctx.owner.token,
                    owner_generation=ctx.owner.generation,
                    **values,
                )
            )
        elif any(getattr(row, k) != v for k, v in values.items()):
            raise ExecutionConflict("stage receipt registration changed")
    return fields


def _verified_stage(stage_num):
    ctx = current_execution()
    key = f"stage/{stage_num}"
    with ctx.owner.transaction() as session:
        row = session.get(storage.StageReceipt, (ctx.owner.run_id, stage_num))
    record = _read(key)
    if row is None and record is None:
        return None
    if record is None:
        raise ExecutionConflict("stage receipt file missing")
    plan = _plan(stage_num)
    if record != {
        "binding": _binding(key),
        **_stage_record(stage_num, plan, reconstruct=False),
    }:
        raise ExecutionConflict("stage completion bindings changed")
    # Reconcile the proved receipt file after a crash before its short DB commit.
    if row is None:
        return _commit_stage(stage_num, plan)
    expected = dict(
        receipt_path=str(_path(key)),
        receipt_hash=_file_hash(_path(key)),
        execution_manifest_hash=ctx.manifest_hash,
        input_fingerprint=ctx.manifest["admission"]["analysis_input_fingerprint"],
        policy_version=POLICY_VERSION,
    )
    if any(getattr(row, k) != v for k, v in expected.items()):
        raise ExecutionConflict("stage receipt registration changed")
    return record


def stage_progress(stage_num, record):
    return dict(
        run_id=current_execution().owner.run_id,
        stage_num=stage_num,
        stage_name=STAGES[stage_num - 1].name,
        status="complete",
        success_count=record["success_count"],
        requested_count=record["requested_count"],
        **({"skipped": True, "reason": record["reason"]} if record["skipped"] else {}),
        **(
            {"label_map": record["label_map"]}
            if record.get("label_map") is not None
            else {}
        ),
        **(
            {"partial_success": record["success_count"] < record["requested_count"]}
            if stage_num == 3
            else {}
        ),
    )


def verified_stage_prefix():
    """The first missing receipt bounds resume; later artifact rows never do."""
    ctx = current_execution()
    ctx.verify()
    completed = 0
    for stage_num in range(1, 7):
        record = _verified_stage(stage_num)
        if record is None:
            break
        completed = stage_num
    return completed


def durable_stage(stage_num):
    def decorate(function):
        signature = inspect.signature(function)

        @wraps(function)
        async def wrapped(*args, **kwargs):
            ctx = current_execution()
            if ctx is None:
                return await function(*args, **kwargs)
            bound = signature.bind(*args, **kwargs)
            from .execution import validate_stage_admission, canonical_execution_report

            validate_stage_admission(
                bound.arguments["run_id"], bound.arguments.get("council_model_ids")
            )
            if stage_num == 1:
                canonical_execution_report(
                    bound.arguments["run_id"], bound.arguments["report"]
                )
            record = _verified_stage(stage_num)
            original_emit = bound.arguments["emit"]
            if record is not None:
                await original_emit(stage_progress(stage_num, record))
                return
            plan = _plan(stage_num)
            _save(f"plan/{stage_num}", plan=plan)
            if stage_num == 4:
                failure = _read("member/s4/consolidation")
                if failure is not None and failure["disposition"] == "failed":
                    if failure["paid"] != _paid("s4/consolidation"):
                        raise ExecutionConflict("consolidator failure receipt changed")
                    if failure.get("error_type") == "needs_auth":
                        from .workflow.exceptions import _NeedsAuth

                        raise _NeedsAuth(failure["error"])
                    raise RuntimeError(failure["error"])
            committed = False

            async def emit(payload):
                nonlocal committed
                if payload.get("status") == "complete" and not payload.get("task"):
                    receipt = _commit_stage(stage_num, plan)
                    committed = True
                    payload = {**payload, **stage_progress(stage_num, receipt)}
                await original_emit(payload)

            bound.arguments["emit"] = emit
            try:
                result = await function(*bound.args, **bound.kwargs)
            except Exception as error:
                from .execution import raise_if_execution_blocked
                from .workflow.exceptions import _NeedsAuth

                raise_if_execution_blocked(error)
                if stage_num == 4 and _read("member/s4/consolidation") is None:
                    _save(
                        "member/s4/consolidation",
                        disposition="failed",
                        error=str(error),
                        error_type="needs_auth"
                        if isinstance(error, _NeedsAuth)
                        else "failed",
                        paid=_paid("s4/consolidation"),
                    )
                raise
            if not committed:
                raise ExecutionConflict("stage returned without explicit completion")
            return result

        return wrapped

    return decorate
