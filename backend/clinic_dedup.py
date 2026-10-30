"""Operator-invoked exact-content consolidation; no generation or background work."""

from pathlib import Path
import hashlib
import os
import shutil
import uuid
from collections import defaultdict
from sqlalchemy import select
from . import storage
from .clinic_catalogue import _write, _bump, _hash_chunks, _now
from .clinic_catalogue_reads import _patient, _allowed_path, _fingerprint
from .clinic_models import (
    ClinicArtifact,
    ClinicLocation,
    ClinicPublication,
    CatalogueConflict,
)


def consolidate_remote(
    patient_id, canonical_key, duplicate_keys, sha256, size, readback
):
    """Bind old keys before the operator deletes their separately verified bodies.

    The caller verifies each removed body. This function independently reads the
    surviving body and rejects recorded ownership/content conflicts atomically.
    Inactive locations retain exact old download-key resolution.
    """
    with storage.session_scope() as s:
        if _patient(s, patient_id).label != patient_id:
            raise ValueError("Use the canonical patient ID for remote cleanup")
    prefix = f"patients/{patient_id}/files/"
    keys = set(duplicate_keys)
    if canonical_key in keys or any(
        not k.startswith(prefix) or "/" in k[len(prefix) :] or not k[len(prefix) :]
        for k in keys | {canonical_key}
    ):
        raise ValueError("Exact patient file keys are required")
    if _hash_chunks(readback()) != (sha256, size):
        raise CatalogueConflict("Surviving remote bytes do not match")
    with _write() as s:
        patient = _patient(s, patient_id)
        if patient.label != patient_id:
            raise CatalogueConflict("Patient changed during remote readback")
        artifacts = list(
            s.scalars(
                select(ClinicArtifact).where(
                    ClinicArtifact.patient_uuid == patient.id,
                    ClinicArtifact.sha256 == sha256,
                    ClinicArtifact.size == size,
                )
            )
        )
        if not artifacts:
            raise CatalogueConflict("Content is not bound to this patient")
        ids = {a.id for a in artifacts}
        existing = list(
            s.scalars(
                select(ClinicLocation).where(
                    ClinicLocation.kind == "netlify",
                    ClinicLocation.key.in_(keys | {canonical_key}),
                )
            )
        )
        if any(row.artifact_id not in ids for row in existing):
            raise CatalogueConflict(
                "A key has a different recorded patient/content binding"
            )
        keep = next(
            (a for a in artifacts if a.file_key == canonical_key[len(prefix) :]), None
        )
        if keep is None:
            owner = next(
                (row.artifact_id for row in existing if row.key == canonical_key), None
            )
            keep = next((a for a in artifacts if a.id == owner), None)
        if keep is None:
            raise CatalogueConflict(
                "Surviving key needs its original catalogue binding"
            )
        if len(artifacts) > 1 and keep.file_key != canonical_key[len(prefix) :]:
            raise CatalogueConflict(
                "Shared content requires its database-issued canonical file key"
            )
        changed = set()
        owner_binding = s.get(ClinicPublication, keep.id)
        if owner_binding is None:
            s.add(ClinicPublication(artifact_id=keep.id, remote_key=canonical_key))
            changed.add(patient.id)
        elif owner_binding.remote_key != canonical_key:
            owner_binding.remote_key = canonical_key
            changed.add(patient.id)
        canonical_locations = {
            row.artifact_id: row for row in existing if row.key == canonical_key
        }
        for artifact in artifacts:
            row = canonical_locations.get(artifact.id)
            if row is None:
                s.add(
                    ClinicLocation(
                        id=str(uuid.uuid4()),
                        artifact_id=artifact.id,
                        kind="netlify",
                        key=canonical_key,
                        patient_alias=patient_id,
                        active=True,
                        verified=True,
                        verified_at=_now(),
                    )
                )
                changed.add(patient.id)
            elif not row.active or not row.verified:
                row.active = row.verified = True
                row.verified_at = _now()
                changed.add(patient.id)
        present = {row.key for row in existing}
        for key in sorted(keys - present):
            owner = next(
                (a for a in artifacts if a.file_key == key[len(prefix) :]), keep
            )
            s.add(
                ClinicLocation(
                    id=str(uuid.uuid4()),
                    artifact_id=owner.id,
                    kind="netlify",
                    key=key,
                    patient_alias=patient_id,
                    active=False,
                    verified=False,
                )
            )
            changed.add(patient.id)
        for row in existing:
            if row.key in keys and (row.active or row.verified):
                row.active = row.verified = False
                row.verified_at = None
                changed.add(patient.id)
        if changed:
            _bump(s, changed)
        return {
            "canonicalKey": canonical_key,
            "aliases": len(keys),
            "fileIds": sorted(ids),
        }


def link_survives_move(source, target):
    """Whether a file at ``source`` still reads the same bytes once moved to ``target``.

    Local consolidation leaves named files as relative links to one ``.content``
    body. Moving such a link to a different depth, or away from the portal
    anchor it points at, would leave it dangling, so a chart tool asks here
    before it plans the move. Ordinary files always survive a move.
    """
    source, target = Path(source), Path(target)
    if not source.is_symlink():
        return True
    # The target folder may not exist yet, so read "../" in the link against
    # the planned path rather than asking the filesystem to walk it.
    planned = os.path.normpath(target.parent.absolute() / os.readlink(source))
    try:
        return Path(planned).resolve(strict=True) == source.resolve(strict=True)
    except (OSError, RuntimeError):
        return False


def _digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest(), Path(path).stat().st_size


def consolidate_local(
    patient_id, receipt=lambda record: None, *, producers_quiescent=False
):
    """Keep original paths as symlinks to one read-only body per patient/hash.

    Replacement is atomic and only follows a fresh byte comparison. Originals
    remain readable at every step. Receipts contain paths/hashes, never copies.
    Re-running after interruption adopts the same content path. The operator must
    stop producers for the entire operation; atomic replacement is not a lock
    against a concurrent producer installing a new output at the same path.

    Two consequences the operator owns:
    - Replacing names with links changes each consolidated patient folder, so
      the next hub watch pass re-reads those folders once. Run this in the same
      maintenance window, with the publisher stopped.
    - Deleting a file later removes its name, not the shared body: the bytes
      under ``.content`` stay until someone releases bodies no name links to.
      Nothing releases them automatically yet.
    """
    if not producers_quiescent:
        raise CatalogueConflict("Stop local producers before consolidating bodies")
    with storage.session_scope() as s:
        patient = _patient(s, patient_id)
        patient_id = patient.label
        rows = list(
            s.execute(
                select(ClinicLocation.key, ClinicArtifact.sha256, ClinicArtifact.size)
                .join(ClinicArtifact, ClinicArtifact.id == ClinicLocation.artifact_id)
                .where(
                    ClinicArtifact.patient_uuid == patient.id,
                    ClinicLocation.kind == "local",
                    ClinicLocation.active.is_(True),
                )
            )
        )
    groups = defaultdict(set)
    failures = []
    for key, digest, size in rows:
        try:
            path = _allowed_path(key)
            if _digest(path) == (digest, size):
                groups[(digest, size)].add(str(Path(key).absolute()))
        except (OSError, ValueError, RuntimeError):
            continue
    from .portal_sync import portal_patients_dir

    data_root = Path(storage.DATA_DIR).resolve()
    portal_root = portal_patients_dir().resolve()
    # The sole inode stays under an immutable internal patient UUID. A hardlink
    # inside the portal tree lets its named aliases remain bounded by that tree;
    # rekeying the visible folder leaves external producer aliases readable.
    content_root = data_root / ".content" / patient.id
    # Old imports may have left extra named copies in the patient's folder that
    # never received catalogue rows. Match their actual bytes to this patient's
    # known content, preserving unrecognized files and all original paths.
    patient_root = portal_root / patient_id
    if portal_root.is_relative_to(data_root) and patient_root.is_dir():
        for path in patient_root.rglob("*"):
            if (
                path.relative_to(patient_root).parts[0] == ".content"
                or path.is_symlink()
                or not path.is_file()
            ):
                continue
            try:
                content = _digest(_allowed_path(path))
                if content in groups:
                    groups[content].add(str(path.absolute()))
            except (OSError, ValueError, RuntimeError):
                continue
    removed = saved = 0
    for (digest, size), names in groups.items():
        canonical = content_root / digest / "original"
        try:
            inodes = {
                (Path(name).stat().st_dev, Path(name).stat().st_ino) for name in names
            }
            if len(inodes) < 2 and not canonical.exists():
                continue
            canonical.parent.mkdir(parents=True, exist_ok=True)
            if not canonical.exists():
                first = _allowed_path(sorted(names)[0])
                in_group = sum(1 for name in names if Path(name).samefile(first))
                if first.stat().st_nlink > in_group:
                    # A name outside this group (an export hard-linked into
                    # place, say) shares the inode. The body becomes read-only,
                    # so it gets its own inode and that name keeps its mode.
                    pending = canonical.with_name(f".pending-{uuid.uuid4().hex}")
                    try:
                        shutil.copyfile(first, pending)
                        os.replace(pending, canonical)
                    finally:
                        pending.unlink(missing_ok=True)
                    removed -= 1
                    saved -= size
                else:
                    os.link(first, canonical)
            if _digest(canonical) != (digest, size):
                raise CatalogueConflict("Local content path contains different bytes")
            portal_anchor = None
            if portal_root.is_relative_to(data_root):
                portal_anchor = patient_root / ".content" / digest / "original"
                portal_anchor.parent.mkdir(parents=True, exist_ok=True)
                if not portal_anchor.exists():
                    os.link(canonical, portal_anchor)
                if not portal_anchor.samefile(canonical):
                    raise CatalogueConflict(
                        "Portal content anchor is not the retained body"
                    )
            # Aliases are read paths. Regeneration uses a new output or atomic
            # replace; an accidental in-place write must not alter every
            # historical reference.
            if canonical.stat().st_mode & 0o777 != 0o444:
                canonical.chmod(0o444)
        except (OSError, ValueError) as error:
            # One unlinkable group (a vanished file, a cross-device link) is
            # reported with the rest; the other groups still consolidate.
            failures.append({"path": sorted(names)[0], "error": str(error)})
            continue
        for name in sorted(names):
            path = Path(name)
            try:
                target = (
                    portal_anchor
                    if portal_anchor is not None and path.is_relative_to(patient_root)
                    else canonical
                )
                if path.absolute() in (canonical.absolute(), target.absolute()) or (
                    path.is_symlink() and path.resolve() == target.resolve()
                ):
                    pass
                else:
                    if _digest(path) != (digest, size):
                        raise CatalogueConflict(
                            "Local candidate changed before replacement"
                        )
                    stat = path.stat()
                    was_copy = (stat.st_dev, stat.st_ino) != (
                        canonical.stat().st_dev,
                        canonical.stat().st_ino,
                    )
                    temporary = path.with_name(path.name + ".dedup-" + uuid.uuid4().hex)
                    try:
                        temporary.symlink_to(os.path.relpath(target, path.parent))
                        os.replace(temporary, path)
                    finally:
                        temporary.unlink(missing_ok=True)
                    if was_copy and stat.st_nlink == 1:
                        removed += 1
                        saved += size
                with _write() as s:
                    changed = False
                    for row in s.scalars(
                        select(ClinicLocation).where(
                            ClinicLocation.kind == "local",
                            ClinicLocation.key.in_(names),
                        )
                    ):
                        if Path(row.key).samefile(canonical):
                            fingerprint = _fingerprint(row.key)
                            if row.fingerprint != fingerprint or not row.verified:
                                row.fingerprint = fingerprint
                                row.verified = True
                                changed = True
                    if changed:
                        _bump(s, patient.id)
                receipt(
                    {
                        "path": name,
                        "canonical": str(canonical),
                        "sha256": digest,
                        "size": size,
                    }
                )
            except (OSError, ValueError) as error:
                failures.append({"path": name, "error": str(error)})
    if failures:
        raise CatalogueConflict(f"Local consolidation incomplete: {failures}")
    return {"removedBodies": removed, "savedBytes": saved}
