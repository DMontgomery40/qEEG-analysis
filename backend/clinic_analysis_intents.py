"""Private original-upload policy binding. Existing E6 remains the only executor."""

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
from sqlalchemy import select
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


def prepare_policy_binding(root):
    from .clinic_intake import _immutable

    path = root / "analysis-policy.json"
    if not path.exists():
        settings = {
            k: v
            for k, v in execution._settings_snapshot().items()
            if not k.endswith(
                ("_KEY", "_TOKEN", "_PASSWORD", "_SECRET", "_CREDENTIALS")
            )
        }
        snapshot = dict(
            publicPolicy=dict(
                councilModelIds=[m.id for m in config.COUNCIL_MODELS],
                consolidatorModelId=config.DEFAULT_CONSOLIDATOR,
                modelRoles=asdict(config.MODEL_ROLE_DEFAULTS),
            ),
            settings=settings,
            prompts=_snapshot_prompts(),
            recipe=execution._recipe(),
        )
        _immutable(path, _json(snapshot).encode())
    data = path.read_bytes()
    snapshot = json.loads(data)
    return snapshot["publicPolicy"], dict(
        path=str(path), sha256=hashlib.sha256(data).hexdigest()
    )


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
