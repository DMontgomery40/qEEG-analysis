"""Opt-in execution manifest and semantic units. The caller retains RunOwner.

prepare_execution snapshots configuration and admission once. execution_context
is task-local and never acquires/releases the supplied owner. E4/E6 own durable
stage completion and admission/consumer activation; no lifecycle is started here.
"""

from __future__ import annotations

import asyncio
from contextlib import contextmanager, asynccontextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from importlib.metadata import version
import hashlib
import json
import os
import re
from pathlib import Path
import tempfile
from types import MappingProxyType
from sqlalchemy import select
import httpx

from .. import storage
from ..execution_settings import settings
from ..paid_transport import paid_scope, raise_if_paid_blocked, PaidOutcomeUnknown
from ..run_execution import ExecutionConflict

_CURRENT = ContextVar("qeeg_council_execution", default=None)
_CURSOR = ContextVar("qeeg_council_cursor", default=None)

ADMISSION_FIELDS = (
    "patient_id",
    "report_id",
    "council_model_ids_json",
    "consolidator_model_id",
    "requested_model_ids_json",
    "resolved_model_ids_json",
    "creating_instance_id",
    "model_catalogue_fingerprint",
    "source_report_ids_json",
    "source_manifest_json",
    "special_instructions",
    "analysis_input_fingerprint",
)
PROMPTS = (
    "stage1_analysis.md",
    "stage2_peer_review.md",
    "stage3_revision.md",
    "stage4_consolidation.md",
    "stage5_final_review.md",
    "stage6_final_draft.md",
)


def _json(value):
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode()


def _hash(data):
    return hashlib.sha256(data).hexdigest()


def _recipe():
    root = Path(__file__).resolve().parents[1]
    files = sorted((root / "council").rglob("*.py")) + [
        root / "llm_client.py",
        root / "execution_settings.py",
        root / "analysis_inputs.py",
        root / "config.py",
        root / "model_selection.py",
    ]
    return {
        "files": {str(p.relative_to(root)): _hash(p.read_bytes()) for p in files},
        "sdk": {
            name: version(name)
            for name in ("pydantic-ai-slim", "openai", "httpx", "pydantic")
        },
    }


def _publish(owner, path, data):
    with owner.file_guard():
        missing = []
        parent = path.parent
        while not parent.exists():
            missing.append(parent)
            parent = parent.parent
        for directory in reversed(missing):
            directory.mkdir(mode=0o700, exist_ok=True)
            _fsync_directory(directory.parent)
        if path.exists():
            if path.read_bytes() != data:
                raise ExecutionConflict("immutable execution binding changed")
            return
        fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".pending-")
        try:
            with os.fdopen(fd, "wb") as out:
                out.write(data)
                out.flush()
                os.fsync(out.fileno())
            try:
                os.link(tmp, path)
            except FileExistsError:
                if path.read_bytes() != data:
                    raise ExecutionConflict("immutable execution binding changed")
            directory = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
        finally:
            os.unlink(tmp)


@dataclass(frozen=True)
class CouncilExecution:
    owner: object
    manifest_path: Path
    manifest_bytes: bytes
    llm_client: object = field(default=None, compare=False, repr=False)

    async def aclose(self):
        if self.llm_client is not None:
            await self.llm_client.aclose()

    @property
    def manifest_hash(self):
        return _hash(self.manifest_bytes)

    @property
    def manifest(self):
        return json.loads(self.manifest_bytes)

    @property
    def roles(self):
        return self.manifest["roles"]

    def verify(self):
        with self.owner.file_guard():
            if _read_binding(self.manifest_path) != self.manifest_bytes:
                raise ExecutionConflict("execution manifest bytes changed")
        saved = self.manifest
        with self.owner.transaction() as session:
            run = session.get(storage.Run, self.owner.run_id)
            if (run.execution_manifest_path, run.execution_manifest_hash) != (
                str(self.manifest_path),
                self.manifest_hash,
            ):
                raise ExecutionConflict("execution manifest binding changed")
            if {name: getattr(run, name) for name in ADMISSION_FIELDS} != saved[
                "admission"
            ]:
                raise ExecutionConflict("original admission changed")
        _verify_admitted_sources(saved["admission"])
        if saved["recipe"] != _recipe():
            raise ExecutionConflict("incompatible execution recipe or SDK")


def prepare_execution(owner, *, llm_client=None):
    """Create/recover original settings/roles/prompts before any owned paid use.

    Original Run admission remains byte-for-byte unchanged. Actual repaired input
    bytes and outcome-selected inputs are immutable bind_source sidecars, with
    the original manifest hash as their parent. Never adopt a legacy manifest.
    """
    from .workflow import stages
    from ..config import CLIPROXY_BASE_URL
    from ..model_selection import resolve_model_preference

    with owner.transaction() as session:
        run = session.get(storage.Run, owner.run_id)
        admission = {name: getattr(run, name) for name in ADMISSION_FIELDS}
        saved_path = run.execution_manifest_path
        saved_hash = run.execution_manifest_hash
    transport = {
        "base_url": getattr(llm_client, "_base_url", CLIPROXY_BASE_URL),
        "timeout_s": getattr(llm_client, "_timeout_s", 120.0),
    }
    if saved_path:
        path = Path(saved_path)
        data = _read_binding(path)
        if _hash(data) != saved_hash:
            raise ExecutionConflict("execution manifest hash mismatch")
        ctx = CouncilExecution(owner, path, data, _execution_client(llm_client))
        ctx.verify()
        if ctx.manifest["transport"] != transport:
            raise ExecutionConflict("execution transport settings changed")
        return ctx
    if not admission["analysis_input_fingerprint"]:
        raise ExecutionConflict("original admission fingerprint required")
    _verify_admitted_sources(admission)
    env = _settings_snapshot()
    council = json.loads(admission["council_model_ids_json"])
    discovered = sorted(stages.DISCOVERED_MODEL_IDS)
    checker_pref = env.get(
        "QEEG_VISION_CHECKER_MODEL", stages.MODEL_ROLE_DEFAULTS.stage1_vision
    )
    checker = resolve_model_preference(checker_pref, discovered)
    if checker and not stages.is_vision_capable(checker):
        checker = None
    extractors = [m for m in council if stages.is_vision_capable(m)]
    if checker:
        extractors = [checker] + [m for m in extractors if m != checker]
    writer = (
        env.get(
            "QEEG_STAGE6_FINAL_DRAFT_MODEL",
            stages.MODEL_ROLE_DEFAULTS.stage6_final_draft,
        )
        or stages.MODEL_ROLE_DEFAULTS.stage6_final_draft
    ).strip()
    fallback = (
        env.get("QEEG_STAGE6_FINAL_DRAFT_FALLBACK_MODEL", "z-ai/glm-5.2")
        or "z-ai/glm-5.2"
    ).strip()
    writers = []
    if discovered:
        for preference in (writer, fallback):
            resolved = resolve_model_preference(preference, discovered)
            if resolved and resolved not in writers:
                writers.append(resolved)
    else:
        writers.append(writer)
    roles = {
        "checker": checker,
        "checker_preference": checker_pref,
        "extractors": extractors,
        "writers": writers,
        "writer_preference": writer,
        "vision": {
            m: stages.is_vision_capable(m) for m in set(council + extractors + writers)
        },
    }
    root = Path(__file__).resolve().parents[1]
    manifest = {
        "schema_version": 1,
        "admission": admission,
        "settings": env,
        "roles": roles,
        "transport": transport,
        "prompts": {
            name: (root / "prompts" / name).read_bytes().decode("utf-8")
            for name in PROMPTS
        },
        "recipe": _recipe(),
    }
    db = Path(owner.store.engine.url.database).resolve()
    path = (
        db.parent
        / (db.name + ".council")
        / _hash(owner.run_id.encode())
        / "manifest.json"
    )
    data = _json(manifest)
    # An orphan is authority too: never replace it with today's settings.
    if path.exists():
        data = path.read_bytes()
        previous = json.loads(data)
        if (
            previous["admission"] != admission
            or previous["recipe"] != manifest["recipe"]
            or previous["transport"] != transport
        ):
            raise ExecutionConflict("orphan execution manifest incompatible")
    else:
        _publish(owner, path, data)
    owner.bind_manifest(str(path), _hash(data))
    ctx = CouncilExecution(owner, path, data, _execution_client(llm_client))
    ctx.verify()
    return ctx


@contextmanager
def execution_context(context):
    context.verify()
    token = _CURRENT.set(context)
    cursor_token = _CURSOR.set(None)
    setting_token = settings.set(MappingProxyType(context.manifest["settings"]))
    try:
        yield context
    finally:
        settings.reset(setting_token)
        _CURRENT.reset(token)
        _CURSOR.reset(cursor_token)


def current_execution():
    return _CURRENT.get()


def bind_source(key, value, *, consumers=None):
    """Verify exact actual input/selection bytes before their first consuming unit."""
    ctx = _CURRENT.get()
    if ctx is None:
        return
    data = _json({"manifest_hash": ctx.manifest_hash, "key": key, "value": value})
    path = ctx.manifest_path.parent / "sources" / (_hash(key.encode()) + ".json")
    if not path.exists() and consumers is not None:
        with ctx.owner.transaction() as session:
            consumed = session.scalar(
                select(storage.PaidRequest)
                .where(
                    storage.PaidRequest.run_id == ctx.owner.run_id,
                    storage.PaidRequest.scope_key.startswith(consumers),
                )
                .limit(1)
            )
            if consumed is not None:
                raise ExecutionConflict("consumed source binding is missing")
    _publish(ctx.owner, path, data)


def raise_if_execution_blocked(error=None):
    cursor = _CURSOR.get()
    if cursor is not None:
        cursor.raise_if_blocked()
    seen = set()
    while error is not None and id(error) not in seen:
        seen.add(id(error))
        if isinstance(error, (PaidOutcomeUnknown, ExecutionConflict)):
            raise error
        error = error.__cause__ or error.__context__
    ctx = _CURRENT.get()
    if ctx is not None:
        raise_if_paid_blocked(ctx.owner)


async def execute_unit(key, awaitable):
    """Enter inside the actual requesting task, including heartbeat children."""
    ctx = _CURRENT.get()
    if ctx is None:
        return await awaitable
    try:
        ctx.verify()
        with paid_scope(
            ctx.owner,
            key,
            ctx.manifest_hash,
            ctx.manifest["admission"]["analysis_input_fingerprint"],
        ) as cursor:
            token = _CURSOR.set(cursor)
            try:
                result = await awaitable
                cursor.raise_if_blocked()
                return result
            except BaseException as error:
                raise_if_execution_blocked(error)
                raise
            finally:
                _CURSOR.reset(token)
    finally:
        # Close a never-awaited coroutine when verification blocked before entry.
        if hasattr(awaitable, "close"):
            awaitable.close()


async def drain_task(task, *, cancel=False):
    if cancel and not task.done():
        task.cancel()
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            continue
        except BaseException:
            break
    return task.result()


async def gather_units(*awaitables):
    """Drain every started member and retain successes when one is unresolved."""
    tasks = [asyncio.ensure_future(a) for a in awaitables]
    gathering = asyncio.gather(*tasks, return_exceptions=True)
    try:
        results = await asyncio.shield(gathering)
    except BaseException as error:
        for task in tasks:
            task.cancel()
        try:
            settled = await drain_task(gathering)
        except BaseException as child_error:
            raise_if_execution_blocked(child_error)
        else:
            for result in settled:
                if isinstance(result, BaseException):
                    raise_if_execution_blocked(result)
        raise_if_execution_blocked(error)
        raise
    failures = [r for r in results if isinstance(r, BaseException)]
    if failures:
        error = next(
            (
                r
                for r in failures
                if isinstance(r, (PaidOutcomeUnknown, ExecutionConflict))
            ),
            failures[0],
        )
        error.completed_results = [
            r for r in results if r is not None and not isinstance(r, BaseException)
        ]
        raise error
    return results


def current_semantic_key():
    cursor = _CURSOR.get()
    return cursor.semantic_key if cursor is not None else None


def require_semantic_scope():
    if _CURRENT.get() is not None and _CURSOR.get() is None:
        raise ExecutionConflict("owned generation requires an explicit semantic unit")


def validate_stage_admission(run_id, council_model_ids=None):
    ctx = _CURRENT.get()
    if ctx is None:
        return
    ctx.verify()
    if run_id != ctx.owner.run_id:
        raise ExecutionConflict("stage run differs from retained owner")
    if council_model_ids is not None and council_model_ids != json.loads(
        ctx.manifest["admission"]["council_model_ids_json"]
    ):
        raise ExecutionConflict("stage council differs from original admission")


def canonical_execution_report(run_id, report, *, report_text=None, page_images=None):
    """Resolve owned input from admission before accepting caller source material."""
    ctx = _CURRENT.get()
    if ctx is None:
        return report
    validate_stage_admission(run_id)
    admission = ctx.manifest["admission"]
    with ctx.owner.transaction() as session:
        canonical = session.get(storage.Report, admission["report_id"])
        if canonical is None or canonical.patient_id != admission["patient_id"]:
            raise ExecutionConflict("original admitted report identity is unavailable")
        for field in (
            "id",
            "patient_id",
            "filename",
            "mime_type",
            "stored_path",
            "extracted_text_path",
        ):
            if getattr(report, field, None) != getattr(canonical, field):
                raise ExecutionConflict(
                    "supplied report differs from original admission"
                )
    if report_text is not None or page_images is not None:
        from ..analysis_inputs import repair_combined_report
        from .report_assets import (
            _derive_report_dir,
            _load_best_report_text,
            _load_page_images,
        )

        try:
            repair_combined_report(canonical, run_id=run_id)
            directory = _derive_report_dir(canonical)
            if report_text is not None and report_text != _load_best_report_text(
                canonical, directory
            ):
                raise ExecutionConflict(
                    "supplied text differs from admitted report source"
                )
            if page_images is not None and page_images != _load_page_images(
                canonical, directory
            ):
                raise ExecutionConflict(
                    "supplied images differ from admitted report source"
                )
        except ExecutionConflict:
            raise
        except Exception as exc:
            raise ExecutionConflict(
                "admitted report source cannot be verified"
            ) from exc
    return canonical


def bind_artifact_sources(key, artifacts):
    ctx = _CURRENT.get()
    if ctx is None:
        return
    bind_source(
        key,
        [
            {
                "id": a.id,
                "model": a.model_id,
                "path": a.content_path,
                "bytes_hex": Path(a.content_path).read_bytes().hex(),
            }
            for a in artifacts
        ],
        consumers=key.split("/")[0] + "/",
    )


class BorrowedAsyncTransport(httpx.AsyncBaseTransport):
    """A context owns its pool; the original caller retains a supplied transport."""

    def __init__(self, inner):
        self.inner = inner

    async def handle_async_request(self, request):
        return await self.inner.handle_async_request(request)


def _execution_client(original):
    if original is None:
        return None
    from ..llm_client import AsyncOpenAICompatClient

    return AsyncOpenAICompatClient(
        base_url=original._base_url,
        api_key=original._api_key,
        timeout_s=original._timeout_s,
        transport=BorrowedAsyncTransport(original._transport)
        if original._transport is not None
        else None,
    )


def execution_llm(original):
    ctx = _CURRENT.get()
    return (
        ctx.llm_client if ctx is not None and ctx.llm_client is not None else original
    )


@asynccontextmanager
async def owned_execution(owner, *, llm_client=None):
    """Retain the external owner; close this context's pools after its work drains."""
    context = prepare_execution(owner, llm_client=llm_client)
    try:
        with execution_context(context):
            yield context
    finally:
        await context.aclose()


def _fsync_directory(path):
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _read_binding(path):
    try:
        return path.read_bytes()
    except OSError as error:
        raise ExecutionConflict("execution manifest is unavailable") from error


def _verify_admitted_sources(admission):
    from ..analysis_inputs import _source_snapshot

    try:
        manifest = json.loads(admission["source_manifest_json"])
        if manifest.get("legacy", True):
            return
        for saved in manifest["sources"]:
            with storage.session_scope() as session:
                report = storage.get_report(session, saved["report_id"])
            if report is None or report.patient_id != admission["patient_id"]:
                raise ExecutionConflict("original admitted source is unavailable")
            current, _ = _source_snapshot(report)
            if any(
                current[k] != saved[k]
                for k in ("report_id", "original_sha256", "asset_digests")
            ):
                raise ExecutionConflict("original admitted source changed")
    except ExecutionConflict:
        raise
    except Exception as error:
        raise ExecutionConflict(
            "original admitted source cannot be verified"
        ) from error


def _settings_snapshot():
    root = Path(__file__).resolve().parents[1]
    consumers = list((root / "council").rglob("*.py")) + [root / "llm_client.py"]
    names = set()
    for path in consumers:
        names.update(re.findall(r"\bQEEG_[A-Z0-9_]+\b", path.read_text()))
    names.update(
        (
            "OPENAI_REASONING_EFFORT",
            "OPENROUTER_BASE_URL",
            "OPENROUTER_HTTP_REFERER",
            "OPENROUTER_APP_TITLE",
        )
    )
    dynamic = (
        "QEEG_OPENROUTER_REASONING_EFFORT_",
        "QEEG_OPENROUTER_REASONING_EXCLUDE_",
    )
    return {k: v for k, v in os.environ.items() if k in names or k.startswith(dynamic)}
