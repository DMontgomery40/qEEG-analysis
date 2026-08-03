"""Move one patient from an old clinic ID to a new one, everywhere it is filed.

This is the unit the cutover migrator repeats per patient, and it is also what a
runtime relabel needs: before this module existed, changing a patient's ID
renamed the database label and left the portal folder sitting under the old one.

Everything here is a rename. Report bytes, video bytes, PDF bytes, and the
``messages`` array of every conversation are read, hashed, and written back
byte-identical or not touched at all. Nothing is regenerated.

Each step is idempotent: a run that already moved the folder, already renamed a
file, or already repointed a conversation sees the finished state and reports it
as such. That is what makes a resumed run safe — it replays nothing, so no file
is duplicated and no deliverable version is inflated.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence


# Historical content the brief requires to survive byte-identical: what the hub
# accepts, plus the stills and alternative video containers that sit in patient
# folders. Cheaper to hash something that did not need it than to carry a file
# across a filesystem boundary unchecked.
DELIVERABLE_SUFFIXES = frozenset(
    {
        ".pdf", ".mp4", ".md", ".docx", ".rtf", ".zip", ".txt",
        ".mov", ".webm", ".m4v", ".png", ".jpg", ".jpeg", ".gif", ".webp",
        ".wav", ".mp3", ".m4a", ".csv", ".edf",
    }
)


class PatientRekeyError(RuntimeError):
    """A rekey that cannot proceed without someone deciding something."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_messages(messages: Any) -> str:
    """Hash a conversation's ``messages`` value exactly as it sits on disk.

    Serialized with sorted keys and no whitespace so the hash tracks content,
    not formatting. Two runs over the same untouched array agree; one changed
    character anywhere does not.
    """
    payload = json.dumps(messages, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def rename_in_name(name: str, old_id: str, new_id: str) -> str:
    """Swap every whole-token occurrence of the old ID inside one filename.

    Portal filenames carry the ID as a ``__``-delimited token and sometimes
    carry it twice (``<id>__<id>__v63__2026-08-03.pdf`` — the remote key shape
    echoed back by sync). Both occurrences move. A name that merely contains the
    digits somewhere else is left alone, and so is a name that never mentions
    the ID at all, such as an original source PDF.
    """
    if not old_id or old_id == new_id or old_id not in name:
        return name

    out: list[str] = []
    index = 0
    span = len(old_id)
    while True:
        found = name.find(old_id, index)
        if found < 0:
            out.append(name[index:])
            break
        before = name[found - 1] if found > 0 else ""
        after_at = found + span
        after = name[after_at] if after_at < len(name) else ""
        # A token boundary is the start/end of the name or a separator. Refusing
        # to match mid-token keeps `12-11-1963-0` out of `12-11-1963-01`.
        boundary = (before in ("", "_", ".", "-", " ", "/")
                    and after in ("", "_", ".", "-", " ", "/"))
        out.append(name[index:found])
        out.append(new_id if boundary else old_id)
        index = after_at
    return "".join(out)


@dataclass
class FileRename:
    old_path: Path
    new_path: Path
    size: int
    sha256: str


@dataclass
class RekeyPlan:
    """Every rename this patient needs, computed before anything is written."""

    old_id: str
    new_id: str
    portal_root: Path | None = None
    conversations_dir: Path | None = None
    sync_state_paths: tuple[Path, ...] = ()
    folder_move: tuple[Path, Path] | None = None
    file_renames: list[FileRename] = field(default_factory=list)
    carried_files: list[FileRename] = field(default_factory=list)
    conversation_files: list[Path] = field(default_factory=list)
    pipeline_job_files: list[Path] = field(default_factory=list)
    remote_rekeys: list[dict[str, Any]] = field(default_factory=list)
    total_bytes: int = 0


@dataclass
class RekeyResult:
    old_id: str
    new_id: str
    folder_moved: bool = False
    files_renamed: int = 0
    files_already_renamed: int = 0
    files_carried_verified: int = 0
    conversations_repointed: int = 0
    sync_entries_repointed: int = 0
    pipeline_jobs_repointed: int = 0
    db_label_changed: bool = False
    message_hashes: dict[str, str] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)


def _portal_dirs(portal_root: Path, old_id: str, new_id: str) -> tuple[Path, Path]:
    return portal_root / old_id, portal_root / new_id


def plan_patient_rekey(
    old_id: str,
    new_id: str,
    *,
    portal_root: Path | None = None,
    conversations_dir: Path | None = None,
    sync_state_paths: Sequence[Path] = (),
    pipeline_jobs_dir: Path | None = None,
    hash_files: bool = True,
) -> RekeyPlan:
    """Work out every rename for this patient without touching anything."""
    if not old_id or not new_id:
        raise PatientRekeyError("A rekey needs both the old and the new clinic ID.")

    plan = RekeyPlan(
        old_id=old_id,
        new_id=new_id,
        portal_root=portal_root,
        conversations_dir=conversations_dir,
        sync_state_paths=tuple(sync_state_paths),
    )

    if portal_root is not None:
        old_dir, new_dir = _portal_dirs(portal_root, old_id, new_id)
        source = old_dir if old_dir.is_dir() else (new_dir if new_dir.is_dir() else None)
        if source is not None:
            if old_dir.is_dir() and old_id != new_id:
                if new_dir.exists():
                    raise PatientRekeyError(
                        f"{new_dir} already exists, so {old_dir} cannot move onto it. "
                        f"Resolve the collision before rekeying {old_id}."
                    )
                plan.folder_move = (old_dir, new_dir)
            # Renames are computed against wherever the files live right now, so
            # a resumed run that already moved the folder still finds them.
            for path in sorted(source.rglob("*")):
                if not path.is_file():
                    continue
                renamed = rename_in_name(path.name, old_id, new_id)
                size = path.stat().st_size
                plan.total_bytes += size
                digest = (
                    _sha256_file(path)
                    if hash_files and path.suffix.lower() in DELIVERABLE_SUFFIXES
                    else ""
                )
                if renamed == path.name:
                    # A file whose name never carried the ID still crosses the
                    # migration — inside the folder move, which is a copy when
                    # the destination is on another filesystem. Original source
                    # PDFs are exactly these files, and they are the ones a
                    # byte check actually earns its keep on: `os.replace` on a
                    # renamed file cannot change bytes, but a cross-device copy
                    # can.
                    if digest:
                        plan.carried_files.append(
                            FileRename(
                                old_path=path,
                                new_path=path,
                                size=size,
                                sha256=digest,
                            )
                        )
                    continue
                plan.file_renames.append(
                    FileRename(
                        old_path=path,
                        new_path=path.with_name(renamed),
                        size=size,
                        sha256=digest,
                    )
                )

    if conversations_dir is not None and conversations_dir.is_dir():
        for path in sorted(conversations_dir.glob("*.json")):
            try:
                record = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            if not isinstance(record, dict):
                continue
            filed = record.get("patient_id") or record.get("patient_label")
            if filed in (old_id, new_id):
                plan.conversation_files.append(path)

    # The pipeline worker's per-patient status file is named for the patient and
    # names them again inside. It is live routing state, not history: left
    # behind, the worker keeps reporting on an ID nothing else uses.
    if pipeline_jobs_dir is not None and pipeline_jobs_dir.is_dir():
        old_job = pipeline_jobs_dir / f"{old_id}.json"
        if old_job.is_file():
            plan.pipeline_job_files.append(old_job)

    return plan


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json_atomic(path: Path, payload: Any) -> None:
    tmp = path.with_name(path.name + ".rekey-tmp")
    tmp.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    os.replace(tmp, path)


def _canonical_ordinal(new_id: str) -> int:
    from .patient_identity import parse_canonical_patient_id

    parsed = parse_canonical_patient_id(new_id)
    return parsed.ordinal if parsed else 1


def _rekey_folder_meta(folder: Path, old_id: str, new_id: str) -> bool:
    """Point ``$meta.json`` at the new ID and the 1-based canonical ordinal."""
    meta_path = folder / "$meta.json"
    if not meta_path.is_file():
        return False
    try:
        meta = _load_json(meta_path)
    except ValueError:
        return False
    if not isinstance(meta, dict):
        return False
    meta["patientId"] = new_id
    meta["index"] = _canonical_ordinal(new_id)
    _write_json_atomic(meta_path, meta)
    return True


def _rekey_conversation(path: Path, old_id: str, new_id: str) -> tuple[str, str]:
    """Repoint one conversation's filing fields, leaving ``messages`` alone.

    Returns the ``messages`` hash before and after. The array is lifted out of
    the record untouched and put back by identity, so the two agree unless
    something else on this machine rewrote the file underneath us.
    """
    record = _load_json(path)
    before = sha256_messages(record.get("messages"))

    record.pop("patient_label", None)
    # A second display form is exactly what the cutover removes: the clinic ID
    # is what staff read, so there is nothing left for this field to say.
    record.pop("patient_display_label", None)
    record["patient_id"] = new_id

    staged = record.get("staged_reports")
    if isinstance(staged, list):
        for entry in staged:
            if not isinstance(entry, dict):
                continue
            # In legacy files this field holds the engine's UUID. Left named
            # `patient_id` it would surface a UUID as a clinic ID on re-drop.
            if "patient_id" in entry and "patient_uuid" not in entry:
                entry["patient_uuid"] = entry.pop("patient_id")
            if "patient_label" in entry:
                entry.pop("patient_label")
            entry["patient_id"] = new_id

    artifacts = record.get("artifacts")
    if isinstance(artifacts, list):
        for entry in artifacts:
            if isinstance(entry, dict) and isinstance(entry.get("path"), str):
                entry["path"] = _rekey_path_string(entry["path"], old_id, new_id)

    after = sha256_messages(record.get("messages"))
    if before != after:
        raise PatientRekeyError(
            f"{path.name}: the messages array changed during rekey. "
            f"Historical conversation content is immutable — nothing was saved."
        )

    _write_json_atomic(path, record)
    return before, after


def _rekey_path_string(value: str, old_id: str, new_id: str) -> str:
    """Rewrite an ID that appears as a path segment or inside a filename."""
    parts = value.split("/")
    return "/".join(
        new_id if part == old_id else rename_in_name(part, old_id, new_id)
        for part in parts
    )


def _rekey_sync_state(path: Path, old_id: str, new_id: str) -> int:
    """Repoint one sync-state file's patient entry, file keys, and remote keys."""
    if not path.is_file():
        return 0
    try:
        state = _load_json(path)
    except ValueError:
        return 0
    if not isinstance(state, dict):
        return 0

    touched = 0
    patients = state.get("patients")
    if isinstance(patients, dict) and old_id in patients:
        patients[new_id] = patients.pop(old_id)
        entry = patients[new_id]
        if isinstance(entry, dict) and isinstance(entry.get("patientId"), str):
            entry["patientId"] = new_id
        touched += 1

    files = state.get("files")
    if isinstance(files, dict):
        for key in [k for k in files if k.split("/", 1)[0] == old_id]:
            entry = files.pop(key)
            tail = key.split("/", 1)[1] if "/" in key else ""
            new_key = f"{new_id}/{rename_in_name(tail, old_id, new_id)}" if tail else new_id
            if isinstance(entry, dict):
                for field_name in ("remoteFileKey", "logicalName"):
                    if isinstance(entry.get(field_name), str):
                        entry[field_name] = rename_in_name(
                            entry[field_name], old_id, new_id
                        )
            files[new_key] = entry
            touched += 1

    if touched:
        _write_json_atomic(path, state)
    return touched


def apply_patient_rekey(
    plan: RekeyPlan,
    *,
    session: Any = None,
    patient_uuid: str | None = None,
) -> RekeyResult:
    """Carry out one patient's rekey. Safe to call again on a finished patient."""
    result = RekeyResult(old_id=plan.old_id, new_id=plan.new_id)

    if plan.folder_move is not None:
        old_dir, new_dir = plan.folder_move
        if old_dir.is_dir():
            if new_dir.exists():
                raise PatientRekeyError(
                    f"{new_dir} appeared after planning; refusing to merge folders."
                )
            shutil.move(str(old_dir), str(new_dir))
            result.folder_moved = True
        else:
            result.notes.append("portal folder was already at the new ID")

    for rename in plan.file_renames:
        old_path, new_path = rename.old_path, rename.new_path
        # A resumed run finds the folder already moved, so re-anchor both ends.
        if plan.folder_move is not None:
            old_dir, new_dir = plan.folder_move
            old_path = _reanchor(old_path, old_dir, new_dir)
            new_path = _reanchor(new_path, old_dir, new_dir)
        if not old_path.exists() and new_path.exists():
            result.files_already_renamed += 1
            continue
        if not old_path.exists():
            raise PatientRekeyError(
                f"{old_path} is gone and {new_path} was never written. "
                f"Stopping rather than guessing what happened to it."
            )
        if new_path.exists():
            raise PatientRekeyError(
                f"{new_path} already exists; renaming {old_path.name} onto it "
                f"would destroy a file."
            )
        os.replace(old_path, new_path)
        if rename.sha256:
            after = _sha256_file(new_path)
            if after != rename.sha256:
                raise PatientRekeyError(
                    f"{new_path.name} changed bytes during rename "
                    f"({rename.sha256[:12]} -> {after[:12]})."
                )
        result.files_renamed += 1

    # Files that kept their names travelled inside the folder move. Prove they
    # arrived intact — a cross-filesystem move is a copy, and a copy can fail
    # halfway without raising.
    for carried in plan.carried_files:
        path = carried.old_path
        if plan.folder_move is not None:
            old_dir, new_dir = plan.folder_move
            path = _reanchor(path, old_dir, new_dir)
        if not path.is_file():
            raise PatientRekeyError(
                f"{path} did not survive the folder move for {plan.old_id}."
            )
        after = _sha256_file(path)
        if after != carried.sha256:
            raise PatientRekeyError(
                f"{path.name} changed bytes crossing the folder move "
                f"({carried.sha256[:12]} -> {after[:12]})."
            )
        result.files_carried_verified += 1

    if plan.portal_root is not None:
        folder = plan.portal_root / plan.new_id
        if folder.is_dir():
            _rekey_folder_meta(folder, plan.old_id, plan.new_id)

    for path in plan.conversation_files:
        if not path.is_file():
            continue
        before, _ = _rekey_conversation(path, plan.old_id, plan.new_id)
        result.message_hashes[path.name] = before
        result.conversations_repointed += 1

    for path in plan.sync_state_paths:
        result.sync_entries_repointed += _rekey_sync_state(
            path, plan.old_id, plan.new_id
        )

    for path in plan.pipeline_job_files:
        target = path.with_name(f"{plan.new_id}.json")
        if not path.is_file():
            continue
        if target.exists():
            raise PatientRekeyError(
                f"{target} already exists; the pipeline job file for "
                f"{plan.old_id} cannot move onto it."
            )
        try:
            job = _load_json(path)
        except ValueError:
            job = None
        if isinstance(job, dict):
            if isinstance(job.get("patient_id"), str):
                job["patient_id"] = plan.new_id
            if isinstance(job.get("note"), str):
                job["note"] = rename_in_name(job["note"], plan.old_id, plan.new_id)
            _write_json_atomic(path, job)
        os.replace(path, target)
        result.pipeline_jobs_repointed += 1

    if session is not None and patient_uuid:
        from .storage import Patient

        patient = session.get(Patient, patient_uuid)
        if patient is not None and patient.label != plan.new_id:
            patient.label = plan.new_id
            session.commit()
            result.db_label_changed = True

    return result


def _reanchor(path: Path, old_dir: Path, new_dir: Path) -> Path:
    try:
        return new_dir / path.relative_to(old_dir)
    except ValueError:
        return path


PORTAL_README = """Local clinician-portal share folder (gitignored)

One folder per patient, named with the clinic patient ID:
  XX_MM-DD-YYYY[_N]   two initials, date of birth, collision ordinal from 2
Example:
  ZZ_01-01-1900       and ZZ_01-01-1900_2 for a second patient with the same
                      initials and date of birth

This is the only patient identifier the engine, the folders, the filenames, the
sync keys, and the hub all use. The engine's internal UUID never appears here.

Put any deliverables the clinic should be able to download inside that folder:
  - PDF, MD, MP4, DOCX, RTF, TXT, ZIP
  - Revisions are fine (keep multiple versions)

How this maps to the Netlify portal:
- The hub (https://thrylen.com/qeeg/) does not read this folder directly; the
  sync pushes from here.

Filename convention:
  <PATIENT_ID>__<name>__v<version>__YYYY-MM-DD.<ext>
Example:
  ZZ_01-01-1900__analysis__v2__2026-01-16.md
"""


def rewrite_portal_readme(portal_root: Path) -> bool:
    """Stop the share folder's own README teaching the retired ID format.

    It is documentation, not a clinical record, and it is the first thing anyone
    reads before dropping a file in here.
    """
    readme = portal_root / "_README.txt"
    if not readme.is_file():
        return False
    if readme.read_text(encoding="utf-8") == PORTAL_README:
        return False
    tmp = readme.with_name("_README.txt.rekey-tmp")
    tmp.write_text(PORTAL_README, encoding="utf-8")
    os.replace(tmp, readme)
    return True


def remote_rekey_worklist(
    sync_state: dict[str, Any], old_id: str, new_id: str
) -> list[dict[str, Any]]:
    """Every remote blob this patient needs copied to a new key, then verified.

    Built from the local sync state rather than a remote listing: the state
    already records the remote key, byte size, logical name, version, and upload
    date of every file that was ever pushed, so the plan is complete offline and
    the remote is only touched during the window.
    """
    files = sync_state.get("files") if isinstance(sync_state, dict) else None
    if not isinstance(files, dict):
        return []

    work: list[dict[str, Any]] = []
    for key in sorted(files):
        if key.split("/", 1)[0] != old_id:
            continue
        entry = files[key] if isinstance(files[key], dict) else {}
        old_key = entry.get("remoteFileKey")
        if not isinstance(old_key, str) or not old_key:
            continue
        work.append(
            {
                "patientIdOld": old_id,
                "patientIdNew": new_id,
                "oldFileKey": old_key,
                "newFileKey": rename_in_name(old_key, old_id, new_id),
                "logicalName": rename_in_name(
                    str(entry.get("logicalName") or ""), old_id, new_id
                ),
                "version": entry.get("version"),
                "uploadedAt": entry.get("uploadedAt"),
                "size": entry.get("size"),
            }
        )
    return work
