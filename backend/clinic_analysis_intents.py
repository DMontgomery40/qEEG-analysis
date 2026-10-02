"""Private original-upload policy binding. Existing E6 remains the only executor."""

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
from sqlalchemy import select, and_, or_, case, func
from . import config, storage
from .council import execution
from .clinic_models import CatalogueConflict, CatalogueNotFound, CatalogueUnavailable
from .clinic_records import ClinicUpload, ClinicUploadItem
from .clinic_catalogue_reads import _json


def _snapshot_prompts():
    root = Path(execution.__file__).resolve().parents[1] / "prompts"
    return {
        name: (root / name).read_text(encoding="utf-8") for name in execution.PROMPTS
    }


def current_policy_snapshot():
    """One current snapshot for preview and original-upload confirmation."""
    from .clinic_naming import POLICY
    from .patient_postprocessing import snapshot_post_config

    settings = {
        k: v
        for k, v in execution._settings_snapshot().items()
        if not k.endswith(("_KEY", "_TOKEN", "_PASSWORD", "_SECRET", "_CREDENTIALS"))
    }
    post = snapshot_post_config(
        settings, config.DISCOVERED_MODEL_IDS, base_url="", timeout_s=600.0
    )
    council = [m.id for m in config.COUNCIL_MODELS]
    offered = [*council, config.DEFAULT_CONSOLIDATOR]
    discovered = set(config.DISCOVERED_MODEL_IDS)
    missing = [m for m in offered if m not in discovered]
    if discovered and missing:
        # The policy the hub shows is the policy a confirmed upload runs on,
        # so only offer models this engine has. On 2026-09-29 the hub offered
        # a council pinned to openai/gpt-5.6-terra, an id the engine never
        # discovered; the confirmed intent could never be admitted and the
        # clinic's "analyze" box did nothing.
        from . import main
        from .clinic_models import AnalysisPolicyUnavailable

        main.LOGGER.warning("analysis_policy_unavailable", missing_models=missing)
        raise AnalysisPolicyUnavailable(
            "Analysis is not available right now. This is on David's end.",
            models=missing,
        )
    return dict(
        publicPolicy=dict(
            councilModelIds=council,
            finalDraftModelIds=council,
            consolidatorModelId=config.DEFAULT_CONSOLIDATOR,
            modelRoles=asdict(config.MODEL_ROLE_DEFAULTS),
        ),
        presentationPolicy={
            **POLICY,
            "render": dict(
                automaticPatientDocument=post["enabled"],
                patientDocumentModel=post["model_id"],
                formats=["markdown", "pdf"],
            ),
            "video": dict(automatic=False, requiresSeparateConfirmation=True),
        },
        settings=settings,
        prompts=_snapshot_prompts(),
        recipe=execution._recipe(),
    )


def public_current_policy():
    snapshot = current_policy_snapshot()
    return dict(
        policy={**snapshot["presentationPolicy"], "analysis": snapshot["publicPolicy"]},
        analysisPolicyFingerprint=hashlib.sha256(_json(snapshot).encode()).hexdigest(),
    )


def prepare_policy_binding(root, *, expected_fingerprint=None):
    from .clinic_intake import _immutable

    path = root / "analysis-policy.json"
    data = (
        path.read_bytes()
        if path.exists()
        else _json(current_policy_snapshot()).encode()
    )
    fingerprint = hashlib.sha256(data).hexdigest()
    if expected_fingerprint is not None and expected_fingerprint != fingerprint:
        raise CatalogueConflict(
            "Analysis policy changed; review and confirm the current policy"
        )
    _immutable(path, data)
    snapshot = json.loads(data)
    return snapshot["publicPolicy"], dict(path=str(path), sha256=fingerprint)


def read_policy_binding(binding):
    try:
        data = Path(binding["path"]).read_bytes()
        if hashlib.sha256(data).hexdigest() != binding["sha256"]:
            raise CatalogueConflict("Original upload policy binding changed")
        snapshot = json.loads(data)
        compatible = snapshot["recipe"] == execution._recipe()
        return snapshot, compatible
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise CatalogueUnavailable("Original upload policy is unavailable") from error


def confirmed_analysis_binding(upload_id):
    """Internal Task3 admission inputs. Does not call admission or start a consumer."""
    with storage.session_scope() as s:
        u = s.get(ClinicUpload, upload_id)
        if not u or not u.analysis_json:
            raise CatalogueNotFound("Confirmed analysis intent not found")
        analysis = json.loads(u.analysis_json)
        snapshot, compatible = read_policy_binding(analysis["policyBinding"])
        items = list(
            s.scalars(
                select(ClinicUploadItem)
                .where(ClinicUploadItem.upload_id == u.id)
                .order_by(ClinicUploadItem.position)
            )
        )
        selected = [items[i] for i in analysis["reportItemIndexes"]]
        return dict(
            uploadId=u.id,
            operationId=analysis["operationId"],
            patientUuid=u.patient_uuid,
            sourceReportIds=[i.source_id for i in selected],
            ready=bool(u.patient_uuid)
            and all(i.status == "registered" for i in selected)
            and compatible,
            compatible=compatible,
            specialInstructions=analysis["specialInstructions"],
            policySnapshot=snapshot,
            policyHash=analysis["policyBinding"]["sha256"],
        )


def run_policy_binding(run):
    """Original confirmed upload policy, verified against the admitted operation."""
    from .run_execution import ExecutionConflict

    with storage.session_scope() as s:
        uploads = list(
            s.scalars(
                select(ClinicUpload).where(ClinicUpload.analysis_json.is_not(None))
            )
        )
        matching = [
            u
            for u in uploads
            if json.loads(u.analysis_json)["operationId"] == run.operation_id
        ]
        if not matching:
            return None
        if len(matching) != 1:
            raise ExecutionConflict("Confirmed operation ownership is ambiguous")
        binding = confirmed_analysis_binding(matching[0].id)
        if not binding["ready"] or not binding["compatible"]:
            raise ExecutionConflict(
                "Original confirmed policy is not ready or compatible"
            )
        operation = s.get(storage.AnalysisInputReservation, run.operation_id)
        material = (
            json.loads(operation.immutable_request_json or "{}") if operation else {}
        )
        if (
            material.get("clinicPolicyHash") != binding["policyHash"]
            or run.patient_id != binding["patientUuid"]
            or json.loads(run.source_report_ids_json or "[]")
            != binding["sourceReportIds"]
        ):
            raise ExecutionConflict("Original confirmed operation binding differs")
        return binding["policySnapshot"]


def admit_confirmed_upload(upload_id):
    """Free original E6 Run admission, only for the already confirmed manifest."""
    from .analysis_inputs import admit_run
    from . import runtime_identity
    from .run_execution import ExecutionConflict
    from .clinic_catalogue import _read_local
    from .clinic_models import ClinicArtifact

    binding = confirmed_analysis_binding(upload_id)
    if not binding["ready"]:
        return None
    snapshot = binding["policySnapshot"]
    policy = snapshot["publicPolicy"]
    requested = [*policy["councilModelIds"], policy["consolidatorModelId"]]
    # Validate each original upload item's byte binding before original admission.
    with storage.session_scope() as s:
        for rid in binding["sourceReportIds"]:
            source = s.get(storage.Report, rid)
            artifact = s.scalar(
                select(ClinicArtifact).where(
                    ClinicArtifact.source_kind == "report",
                    ClinicArtifact.source_id == rid,
                )
            )
            if (
                not source
                or not artifact
                or source.patient_id != binding["patientUuid"]
            ):
                raise ExecutionConflict("Original confirmed source missing")
            _, digest, size, _ = _read_local(source.stored_path)
            if (digest, size) != (artifact.sha256, artifact.size):
                raise ExecutionConflict("Original confirmed source bytes differ")

    def models():
        discovered = sorted(config.DISCOVERED_MODEL_IDS)
        if any(m not in discovered for m in requested):
            raise CatalogueUnavailable("Original confirmed models are unavailable")
        runtime = runtime_identity.current_runtime_identity(discovered)
        return dict(
            council_model_ids=policy["councilModelIds"],
            consolidator_model_id=policy["consolidatorModelId"],
            requested_model_ids=requested,
            resolved_model_ids=requested,
            creating_instance_id=str(runtime["instance_id"]),
            model_catalogue_fingerprint=str(runtime["model_catalogue_fingerprint"]),
        )

    material = dict(
        patient_id=binding["patientUuid"],
        source_ids=binding["sourceReportIds"],
        special_instructions=binding["specialInstructions"],
        source_session_aliases={},
        requested_model_ids=requested,
        allowed_model_fallbacks={},
        clinicPolicyHash=binding["policyHash"],
    )
    run = admit_run(
        patient_id=binding["patientUuid"],
        source_ids=binding["sourceReportIds"],
        special_instructions=binding["specialInstructions"],
        source_session_aliases={},
        operation_id=binding["operationId"],
        model_fields=models,
        immutable_request=material,
    )
    run_policy_binding(run)
    return run


# A confirmed upload whose admission raised, with when to try again. Before
# this, the scan retried SC_12-20-1975's hub upload once a second for 72
# minutes on 2026-09-29 (4,128 full tracebacks: "Original confirmed models
# are unavailable") and the hub showed the clinic nothing.
_ADMISSION_BACKOFF: dict[str, dict] = {}
_BACKOFF_FIRST_S = 30.0
_BACKOFF_MAX_S = 1800.0


def admission_block(upload_id) -> str | None:
    """The plain reason a confirmed upload is not running yet, if it is blocked."""
    entry = _ADMISSION_BACKOFF.get(upload_id)
    return entry["reason"] if entry else None


def _note_admission_failure(upload_id, error, now) -> float:
    entry = _ADMISSION_BACKOFF.get(upload_id) or {"attempts": 0}
    attempts = int(entry["attempts"]) + 1
    # One immediate retry covers a transient fault (a lock, a timeout); a
    # second failure is a condition nobody fixes in a second, so back off.
    delay = 0.0 if attempts == 1 else min(_BACKOFF_FIRST_S * (2 ** (attempts - 2)), _BACKOFF_MAX_S)
    _ADMISSION_BACKOFF[upload_id] = {
        "attempts": attempts,
        "next_attempt": now + delay,
        "reason": str(error)[:200] or type(error).__name__,
    }
    return delay


async def activate_confirmed_uploads(runtime):
    """Existing E6 consumer scans its original confirmed upload references."""
    import asyncio
    from . import main

    if runtime.store.engine is not storage.engine:
        return
    with storage.session_scope() as s:
        operation_id = case(
            (
                func.json_valid(ClinicUpload.analysis_json),
                func.json_extract(ClinicUpload.analysis_json, "$.operationId"),
            ),
            else_=None,
        )
        pending = list(
            s.execute(
                select(ClinicUpload.id, storage.Run)
                .outerjoin(
                    storage.Run,
                    and_(
                        storage.Run.operation_id == operation_id,
                        storage.Run.patient_id == ClinicUpload.patient_uuid,
                    ),
                )
                .where(ClinicUpload.analysis_json.is_not(None))
                .where(
                    or_(
                        storage.Run.id.is_(None),
                        and_(
                            storage.Run.start_requested_at.is_(None),
                            storage.Run.status != "complete",
                        ),
                    )
                )
                .order_by(ClinicUpload.id)
            )
        )
    now = asyncio.get_running_loop().time()
    for upload_id, run in pending:
        waiting = _ADMISSION_BACKOFF.get(upload_id)
        if waiting and waiting["next_attempt"] > now:
            continue
        try:
            if run is None:
                run = await runtime.admission(admit_confirmed_upload, upload_id)
            else:
                # Admission is already durable. Preserve original upload ownership
                # checks while recovering only its missing start intent.
                await runtime.admission(run_policy_binding, run)
            # A stage that succeeded clears the count; a later failure is new.
            _ADMISSION_BACKOFF.pop(upload_id, None)
            if run is not None and run.start_requested_at is None:
                await runtime.admission(main._new_start_intent, runtime.store, run.id)
                _ADMISSION_BACKOFF.pop(upload_id, None)
        except asyncio.CancelledError:
            raise
        except Exception as error:
            delay = _note_admission_failure(upload_id, error, now)
            main.LOGGER.warning(
                "confirmed_upload_admission_blocked",
                upload_id=upload_id,
                reason=str(error)[:200],
                error_type=type(error).__name__,
                retry_in_s=delay,
                attempts=_ADMISSION_BACKOFF[upload_id]["attempts"],
            )
