"""Inactive durable patient report continuation. E6 retains and releases RunOwner.

Admission is explicit for historical complete runs. No scheduler, council stage,
Cathode queue, delivery verifier or notification is started here. Sync receipts
report local handoff only; they never assert deployed delivery.
"""

from __future__ import annotations

import asyncio
import json
import tempfile
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from types import MappingProxyType

from sqlalchemy import select, update

from . import config, storage, patient_facing_pdf
from .council.execution import _hash, _json, _publish, _execution_client, drain_task
from .execution_settings import settings
from .paid_transport import PaidOutcomeUnknown, post_paid_scope
from .run_execution import ExecutionConflict, ExecutionStore
from .portal_files import normalize_portal_patient_id
from . import portal_sync
from scripts import generate_patient_facing_writeups as writer


def _recipe():
    root = Path(__file__).resolve().parents[1]
    return {
        "files": {
            name: _hash((root / name).read_bytes())
            for name in (
                "scripts/generate_patient_facing_writeups.py",
                "backend/patient_facing_pdf.py",
                "backend/llm_client.py",
                "backend/patient_postprocessing.py",
            )
        },
        "packages": {
            name: version(name) for name in ("httpx", "markdown", "weasyprint")
        },
    }


def snapshot_post_config(env, discovered, *, base_url, timeout_s):
    """Call once at council admission, or at explicit post-only admission."""
    preferred = (
        env.get("QEEG_PATIENT_FACING_MODEL")
        or config.MODEL_ROLE_DEFAULTS.patient_facing_rewrite
    ).strip()
    resolved = config.resolve_role_model(preferred, "z-ai/glm-5.2", discovered)
    return {
        "enabled": env.get("QEEG_AUTO_PATIENT_FACING", "1").lower().strip()
        not in ("", "0", "false", "no", "off", "n"),
        "retired_cathode_flag": env.get("QEEG_AUTO_CATHODE_VIDEO", ""),
        "model_preference": preferred,
        "model_id": resolved,
        "version_prefix": (
            env.get("QEEG_PATIENT_FACING_AUTO_VERSION_PREFIX") or "auto"
        ).strip()
        or "auto",
        "base_url": base_url,
        "timeout_s": timeout_s,
        "settings": dict(env),
        "recipe": _recipe(),
        "logo_uri": patient_facing_pdf._get_logo_base64(),
        "example_hex": (
            Path(writer.__file__).resolve().parents[1]
            / "examples/final-patient-facing-writeup-example.md"
        )
        .read_bytes()
        .hex(),
        "portal_dir": str(portal_sync.portal_patients_dir().resolve()),
        "sync_enabled": env.get("QEEG_PORTAL_NETLIFY_SYNC_ON_PUBLISH", "")
        .lower()
        .strip()
        in ("1", "true", "yes", "on", "y"),
        "sync_repo": str(portal_sync.portal_sync_repo().resolve()),
    }


def _root(owner):
    return (
        owner.store.db_path.parent
        / (owner.store.db_path.name + ".post")
        / _hash(owner.run_id.encode())
    )


def _load(path, digest=None):
    try:
        raw = Path(path).read_bytes()
        if digest is not None and _hash(raw) != digest:
            raise ExecutionConflict("postprocessing receipt hash changed")
        return json.loads(raw)
    except (OSError, ValueError) as error:
        raise ExecutionConflict("postprocessing receipt unavailable") from error


class PostAdmissionUnavailable(ExecutionConflict):
    """Clinical completion stands; patient-output prerequisites are unavailable."""


def _source_bytes(artifact):
    try:
        return Path(artifact.content_path).read_bytes().hex()
    except OSError as error:
        raise PostAdmissionUnavailable(
            "originating council source file unavailable"
        ) from error


def _manifest(owner, cfg, *, explicit=False):
    path = _root(owner) / "patient_facing.json"
    with owner.transaction() as session:
        previous = session.get(storage.PostObligation, (owner.run_id, "patient_facing"))
        run = session.get(storage.Run, owner.run_id)
        patient = session.get(storage.Patient, run.patient_id)
        artifacts = list(
            session.scalars(
                select(storage.Artifact).where(storage.Artifact.run_id == run.id)
            )
        )
    if previous is not None:
        return Path(previous.manifest_path), _load(
            previous.manifest_path, previous.manifest_hash
        )
    if path.exists():
        saved = _load(path)
        if saved.get("run_id") != owner.run_id or saved.get("kind") != "patient_facing":
            raise ExecutionConflict("orphan post manifest identity changed")
        return path, saved
    enabled = explicit or cfg["enabled"]
    data = {
        "schema_version": 1,
        "run_id": run.id,
        "kind": "patient_facing",
        "explicit": explicit,
        "enabled": enabled,
        "config": cfg,
    }
    if enabled:
        chosen = writer.select_source_artifacts(run, artifacts)
        problem = (
            "original canonical patient identity unavailable"
            if patient is None
            or normalize_portal_patient_id(patient.label) != patient.label
            else "originating run has no council report sources"
            if not chosen
            or not any(a.stage_num == 6 and a.kind == "final_draft" for a in chosen)
            else None
        )
        if problem:
            raise PostAdmissionUnavailable(problem)
        sources = [
            {
                "artifact_id": a.id,
                "stage_num": a.stage_num,
                "kind": a.kind,
                "model_id": a.model_id,
                "content_path": a.content_path,
                "name": f"stage-{a.stage_num}:{a.kind}:{a.model_id}",
                "bytes_hex": _source_bytes(a),
            }
            for a in chosen
        ]
        for source in sources:
            source["sha256"] = _hash(bytes.fromhex(source["bytes_hex"]))
        now = datetime.now(timezone.utc)
        version = f"{cfg['version_prefix']}-{run.id.split('-')[0]}"
        stem = writer._output_stem(
            patient_label=patient.label,
            version=version,
            date_str=now.date().isoformat(),
        )
        destinations = {
            kind: str(Path(cfg["portal_dir"]) / patient.label / (stem + ext))
            for kind, ext in (("md", ".md"), ("pdf", ".pdf"), ("meta", "__meta.json"))
        }
        if Path(stem).name != stem:
            raise PostAdmissionUnavailable(
                "output version must be a single filename component"
            )
        if any(Path(p).exists() for p in destinations.values()):
            raise PostAdmissionUnavailable(
                "historic output already occupies authorized destination"
            )
        prompt = writer._build_prompt(
            patient_label=patient.label,
            example_text=bytes.fromhex(cfg["example_hex"]).decode(
                "utf-8", errors="replace"
            ),
            source_reports=[
                (
                    s["name"],
                    bytes.fromhex(s["bytes_hex"]).decode("utf-8", errors="replace"),
                )
                for s in sources
            ],
            target_body_words=writer._count_words(
                bytes.fromhex(cfg["example_hex"]).decode("utf-8", errors="replace")
            ),
        )
        data.update(
            patient_id=patient.id,
            patient_label=patient.label,
            selected_artifact_id=run.selected_artifact_id,
            sources=sources,
            source_fingerprint=_hash(_json(sources)),
            prompt=prompt,
            version=version,
            date=now.date().isoformat(),
            generated_at=now.isoformat(),
            destinations=destinations,
            max_tokens=12000,
            temperature=0.2,
        )
    _publish(owner, path, _json(data))
    return path, data


def prepare_completion_posts(owner, cfg):
    """Prepare immutable files BEFORE council-complete's fenced DB transaction."""
    try:
        path, data = _manifest(owner, cfg)
    except PostAdmissionUnavailable as error:
        path = _root(owner) / "patient_facing.json"
        data = {
            "schema_version": 1,
            "run_id": owner.run_id,
            "kind": "patient_facing",
            "explicit": False,
            "enabled": cfg["enabled"],
            "config": cfg,
            "blocked_reason": str(error),
        }
        _publish(owner, path, _json(data))
    cathode = {
        "schema_version": 1,
        "run_id": owner.run_id,
        "kind": "cathode",
        "reason": "manual_fallback",
        "diagnostic": "automatic_cathode_routing_retired"
        if cfg["retired_cathode_flag"]
        else None,
    }
    cp = _root(owner) / "cathode.json"
    _publish(owner, cp, _json(cathode))
    return [(path, data), (cp, cathode)]


def register_completion_posts(session, owner, prepared):
    """No file IO/await: caller commits this with the clinical complete projection."""
    for path, data in prepared:
        key = (owner.run_id, data["kind"])
        digest = _hash(_json(data))
        row = session.get(storage.PostObligation, key)
        if row is not None:
            if (row.manifest_path, row.manifest_hash) != (str(path), digest):
                raise ExecutionConflict("post obligation binding changed")
            continue
        skip = data["kind"] == "cathode" or not data["enabled"]
        session.add(
            storage.PostObligation(
                run_id=owner.run_id,
                kind=data["kind"],
                manifest_path=str(path),
                manifest_hash=digest,
                state="skipped"
                if skip
                else "blocked"
                if data.get("blocked_reason")
                else "pending",
                blocked_reason=(
                    "manual_fallback" if data["kind"] == "cathode" else "disabled"
                )
                if skip
                else data.get("blocked_reason"),
                owner_token=owner.token,
                owner_generation=owner.generation,
            )
        )


def admit_patient_facing(store: ExecutionStore, run_id: str, *, config_snapshot):
    """Explicit authorized historical action. Repeated/concurrent calls rejoin.

    Returns projection; never runs council or modifies its input/model attestation.
    Busy ownership returns an admitting projection for E6 to retry the same call.
    """
    from .orchestration import run_downstream_delivery_gaps, summarize_run_progress
    from sqlalchemy.orm import Session

    with Session(store.engine) as session:
        run = session.get(storage.Run, run_id)
        if run is None:
            raise KeyError(run_id)
        prior = session.get(storage.PostObligation, (run_id, "patient_facing"))
        if prior is not None:
            return project_patient_facing(store, run_id)
        artifacts = list(
            session.scalars(
                select(storage.Artifact).where(storage.Artifact.run_id == run_id)
            )
        )
        if run_downstream_delivery_gaps(
            run,
            progress=summarize_run_progress(run),
            artifacts=artifacts,
            require_final_draft=True,
        ):
            raise ExecutionConflict(
                "explicit post-only request requires delivery-ready complete run"
            )
    store.request_run_start(run_id)
    # Explicitly add post-only work to a previously finished execution. This CAS
    # cannot reopen any prior obligation or take a live/blocked owner's work.
    with Session(store.engine) as session:
        existing_post = select(storage.PostObligation.run_id).where(
            storage.PostObligation.run_id == run_id,
            storage.PostObligation.kind == "patient_facing",
        )
        unresolved_paid = select(storage.PaidRequest.run_id).where(
            storage.PaidRequest.run_id == run_id,
            storage.PaidRequest.state.in_(["prepared", "dispatched", "unknown"]),
        )
        session.execute(
            update(storage.Run)
            .where(
                storage.Run.id == run_id,
                storage.Run.status == "complete",
                storage.Run.execution_state == "done",
                ~existing_post.exists(),
                ~unresolved_paid.exists(),
            )
            .values(execution_state="pending", next_check_at=None)
        )
        session.commit()
    owner = store.claim_run_owner(run_id)
    if owner is None:
        return {"run_id": run_id, "state": "admitting", "verified": False}
    try:
        path, data = _manifest(owner, config_snapshot, explicit=True)
        owner.ensure_post_obligation("patient_facing", str(path), _hash(_json(data)))
    finally:
        owner.release()
    return project_patient_facing(store, run_id)


def _verify_manifest(owner, data):
    if data["run_id"] != owner.run_id or data["config"]["recipe"] != _recipe():
        raise ExecutionConflict("incompatible patient output recipe")
    with owner.transaction() as session:
        run = session.get(storage.Run, owner.run_id)
        patient = session.get(storage.Patient, data["patient_id"])
        if (
            run.patient_id != data["patient_id"]
            or patient is None
            or patient.label != data["patient_label"]
        ):
            raise ExecutionConflict("original patient identity changed")
        rows = {
            s["artifact_id"]: session.get(storage.Artifact, s["artifact_id"])
            for s in data["sources"]
        }
    for source in data["sources"]:
        row = rows[source["artifact_id"]]
        if (
            row is None
            or row.run_id != owner.run_id
            or any(
                getattr(row, k) != source[k]
                for k in ("stage_num", "kind", "model_id", "content_path")
            )
        ):
            raise ExecutionConflict("original source identity changed")
        try:
            content = Path(source["content_path"]).read_bytes()
        except OSError as error:
            raise ExecutionConflict("original source unavailable") from error
        if content.hex() != source["bytes_hex"] or _hash(content) != source["sha256"]:
            raise ExecutionConflict("original source bytes changed")
    if _hash(_json(data["sources"])) != data["source_fingerprint"]:
        raise ExecutionConflict("original source fingerprint changed")


def _accepted_output(owner, root, kind, destination, produce):
    intent = root / (kind + ".json")
    if intent.exists():
        binding = _load(intent)
        content = bytes.fromhex(binding["bytes_hex"])
        if (
            binding["path"] != destination
            or _hash(content) != binding["sha256"]
            or len(content) != binding["size"]
        ):
            raise ExecutionConflict("output intent changed")
    else:
        content = produce()
        binding = dict(
            path=destination,
            size=len(content),
            sha256=_hash(content),
            bytes_hex=content.hex(),
        )
        _publish(owner, intent, _json(binding))
    # Never overwrite differing historic bytes. Missing bytes can be reproduced exactly.
    _publish(owner, Path(destination), content)
    return {k: binding[k] for k in ("path", "size", "sha256")}


def _publish_outputs(owner, data, md, meta):
    root = _root(owner) / "outputs"
    result = {}
    result["md"] = _accepted_output(
        owner, root, "md", data["destinations"]["md"], lambda: (md + "\n").encode()
    )

    def pdf():
        with owner.file_guard(), tempfile.TemporaryDirectory(dir=root) as directory:
            path = Path(directory) / "report.pdf"
            with patient_facing_pdf.patient_pdf_assets(data["config"]["logo_uri"]):
                writer.render_patient_facing_markdown_to_pdf(
                    md, path, patient_label=data["patient_label"]
                )
            return path.read_bytes()

    result["pdf"] = _accepted_output(
        owner, root, "pdf", data["destinations"]["pdf"], pdf
    )
    result["meta"] = _accepted_output(
        owner,
        root,
        "meta",
        data["destinations"]["meta"],
        lambda: (json.dumps(meta, indent=2, sort_keys=True) + "\n").encode(),
    )
    _publish(owner, root / "local.json", _json(result))
    return result


def _verify_outputs(outputs):
    for binding in outputs.values():
        try:
            raw = Path(binding["path"]).read_bytes()
        except OSError as error:
            raise ExecutionConflict("required output unavailable") from error
        if len(raw) != binding["size"] or _hash(raw) != binding["sha256"]:
            raise ExecutionConflict("required output binding changed")


async def continue_patient_facing(owner, *, llm_client, sync=None):
    """One bounded original generation→MD→PDF→meta→sync continuation.

    Caller retains RunOwner; this function owns/closes its borrowed transport
    client before returning/raising. Local failures remain retryable. Unknown or
    incompatible authority blocks the post; caller releases its run accordingly.
    """
    with owner.transaction() as session:
        row = session.get(storage.PostObligation, (owner.run_id, "patient_facing"))
    if row is None:
        raise ExecutionConflict("patient output was not admitted")
    if row.state in ("done", "skipped", "blocked"):
        return project_patient_facing(owner.store, owner.run_id)
    original_state = row.state
    owner.transition_post_obligation(
        "patient_facing", expected_state=original_state, state="owned"
    )
    client = None
    setting_token = None
    try:
        data = _load(row.manifest_path, row.manifest_hash)
        _verify_manifest(owner, data)
        cfg = data["config"]
        client = _execution_client(llm_client)
        client._base_url, client._timeout_s = cfg["base_url"], cfg["timeout_s"]
        setting_token = settings.set(MappingProxyType(cfg["settings"]))
        with owner.transaction() as session:
            prior = session.scalar(
                select(storage.PaidRequest)
                .where(
                    storage.PaidRequest.run_id == owner.run_id,
                    storage.PaidRequest.scope_key == "post/patient_facing/generation",
                )
                .order_by(storage.PaidRequest.dispatch_ordinal.desc())
                .limit(1)
            )
        if prior is None or prior.state in ("prepared", "rejected"):
            if cfg["model_id"] not in await client.list_models():
                raise ExecutionConflict("pinned patient model unavailable")
        with post_paid_scope(
            owner,
            "patient_facing",
            row.manifest_path,
            row.manifest_hash,
            data["source_fingerprint"],
        ) as scope:
            meta = {
                "patient_label": data["patient_label"],
                "patient_id": data["patient_id"],
                "run_id": owner.run_id,
                "llm_model_id": cfg["model_id"],
                "generated_at": data["generated_at"],
                "source_artifacts": [
                    {
                        k: s[k]
                        for k in (
                            "artifact_id",
                            "stage_num",
                            "kind",
                            "model_id",
                            "content_path",
                        )
                    }
                    for s in data["sources"]
                ],
            }
            outputs = await writer.generate_writeup(
                client,
                model_id=cfg["model_id"],
                prompt=data["prompt"],
                temperature=data["temperature"],
                max_tokens=data["max_tokens"],
                label=data["patient_label"],
                meta=meta,
                **{
                    kind + "_path": Path(data["destinations"][kind])
                    for kind in ("md", "pdf", "meta")
                },
                publisher=lambda md, meta: _publish_outputs(owner, data, md, meta),
            )
            scope.raise_if_blocked()
        _verify_outputs(outputs)
        sync_path = _root(owner) / "sync.json"
        if not sync_path.exists():
            if cfg["sync_enabled"]:
                if sync is None:
                    # Existing helper uses process env; verify its original routing, never mutate global env.
                    if (
                        str(portal_sync.portal_patients_dir().resolve()),
                        str(portal_sync.portal_sync_repo().resolve()),
                    ) != (cfg["portal_dir"], cfg["sync_repo"]):
                        raise ExecutionConflict("original sync routing changed")
                    sync = writer.sync_patient_to_thrylen
                if not sync(data["patient_label"]):
                    raise OSError("patient output sync remains pending")
            _publish(
                owner,
                sync_path,
                _json(
                    {
                        "outputs": outputs,
                        "status": "handed_off" if cfg["sync_enabled"] else "disabled",
                        "delivery_verified": False,
                    }
                ),
            )
        sync_receipt = _load(sync_path)
        if sync_receipt["outputs"] != outputs:
            raise ExecutionConflict("sync output bindings changed")
        receipt = {
            "run_id": owner.run_id,
            "manifest_hash": row.manifest_hash,
            "outputs": outputs,
            "paid": _paid_bindings(owner.store, owner.run_id),
            "sync": sync_receipt,
            "delivery_verified": False,
        }
        receipt_path = _root(owner) / "complete.json"
        _publish(owner, receipt_path, _json(receipt))
        _verify_outputs(outputs)
        owner.transition_post_obligation(
            "patient_facing",
            expected_state="owned",
            state="done",
            receipt_path=str(receipt_path),
            receipt_hash=_hash(_json(receipt)),
        )
    except writer.UpstreamError as error:
        with owner.transaction() as session:
            acknowledged = session.scalar(
                select(storage.PaidRequest)
                .where(
                    storage.PaidRequest.run_id == owner.run_id,
                    storage.PaidRequest.scope_key == "post/patient_facing/generation",
                    storage.PaidRequest.state.in_(["response_saved", "rejected"]),
                )
                .limit(1)
            )
        if acknowledged is not None:
            owner.transition_post_obligation(
                "patient_facing",
                expected_state="owned",
                state="blocked",
                blocked_reason=str(error),
            )
        raise
    except (PaidOutcomeUnknown, ExecutionConflict, ValueError) as error:
        owner.transition_post_obligation(
            "patient_facing",
            expected_state="owned",
            state="blocked",
            blocked_reason=str(error),
        )
        raise
    finally:
        try:
            if client is not None:
                await drain_task(asyncio.create_task(client.aclose()))
        finally:
            if setting_token is not None:
                settings.reset(setting_token)
    return project_patient_facing(owner.store, owner.run_id)


def project_patient_facing(store, run_id):
    """Read-only exact manifest for future GET-run/catalogue integration."""
    from sqlalchemy.orm import Session

    with Session(store.engine) as session:
        row = session.get(storage.PostObligation, (run_id, "patient_facing"))
        if row is None:
            return {"run_id": run_id, "state": "absent", "verified": False}
        result = {
            "run_id": run_id,
            "state": row.state,
            "verified": False,
            "blocked_reason": row.blocked_reason,
            "manifest_path": row.manifest_path,
            "manifest_hash": row.manifest_hash,
            "delivery_verified": False,
        }
        try:
            manifest = _load(row.manifest_path, row.manifest_hash)
            local_path = Path(row.manifest_path).parent / "outputs" / "local.json"
            result["local_complete"] = False
            if local_path.exists():
                local = _load(local_path)
                if {
                    kind: binding["path"] for kind, binding in local.items()
                } != manifest["destinations"]:
                    raise ExecutionConflict("local output destinations changed")
                _verify_outputs(local)
                result.update(local_complete=True, outputs=local)
            if row.state == "done":
                receipt = _load(row.receipt_path, row.receipt_hash)
                if (
                    receipt["run_id"] != run_id
                    or receipt["manifest_hash"] != row.manifest_hash
                ):
                    raise ExecutionConflict("completion binding changed")
                if receipt["paid"] != _paid_bindings(store, run_id):
                    raise ExecutionConflict("original generation binding changed")
                _verify_outputs(receipt["outputs"])
                result.update(
                    verified=True, outputs=receipt["outputs"], sync=receipt["sync"]
                )
        except (ExecutionConflict, KeyError, TypeError) as error:
            result.update(
                verified=False, local_complete=False, integrity_error=str(error)
            )
        return result


def _paid_bindings(store, run_id):
    """Verify complete original request/response files even after local completion."""
    from sqlalchemy.orm import Session

    fields = (
        "scope_key",
        "dispatch_ordinal",
        "request_path",
        "request_hash",
        "response_path",
        "response_hash",
        "route_json",
        "state",
        "execution_manifest_hash",
        "input_fingerprint",
        "http_status",
    )
    with Session(store.engine) as session:
        rows = list(
            session.scalars(
                select(storage.PaidRequest)
                .where(
                    storage.PaidRequest.run_id == run_id,
                    storage.PaidRequest.scope_key == "post/patient_facing/generation",
                )
                .order_by(storage.PaidRequest.dispatch_ordinal)
            )
        )
        bindings = [{key: getattr(row, key) for key in fields} for row in rows]
    if not bindings:
        raise ExecutionConflict("original generation receipt missing")
    for binding in bindings:
        if binding["state"] not in ("response_saved", "rejected"):
            raise ExecutionConflict("original generation is unresolved")
        for kind in ("request", "response"):
            try:
                raw = Path(binding[kind + "_path"]).read_bytes()
            except (OSError, TypeError) as error:
                raise ExecutionConflict(
                    "original generation receipt unavailable"
                ) from error
            if _hash(raw) != binding[kind + "_hash"]:
                raise ExecutionConflict("original generation receipt changed")
    return bindings
