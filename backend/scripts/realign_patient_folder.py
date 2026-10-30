"""Move one chart's patient-folder files into the folder named for its clinic ID.

    uv run python -m backend.scripts.realign_patient_folder --chart BT_12-11-1963          # plan only
    uv run python -m backend.scripts.realign_patient_folder --chart BT_12-11-1963 \\
        --apply --yes-realign BT_12-11-1963 --audit realign.json

A relabel or merge changes the chart's ID in the catalogue, and so on the hub,
but has left its files in a folder under the old ID (``folder_label`` in the
chart consistency check). This finishes that one move and nothing else.

- Only files the catalogue lists for this chart move, each to the same relative
  path under ``portal_patients/<chart id>/``, bytes checked by sha256 on arrival.
  Their catalogue locations move with them, so the hub and the clinic keep
  reading them.
- Nothing is written over. A name already present with the same bytes is reused;
  with different bytes the plan refuses.
- The plan refuses when the old folder holds a file of another chart, or any file
  the catalogue does not list, so two people's work is never folded together.
- Finder and folder bookkeeping (``.DS_Store``, ``$meta.json``) is set aside
  beside the audit file, never deleted. Empty folders left behind are removed.
- File moves and the database step are one unit: if the database step fails,
  every moved file goes back to its old path, checked by hash.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sqlalchemy import text

from backend import storage
from backend.patient_rekey import _sha256_file, _write_json_atomic
from backend.portal_sync import portal_patients_dir
from backend.scripts.merge_patient_charts import (
    SKIPPED_NAMES,
    MergeRefused,
    _chart,
    _distinct_target,
)


class RealignRefused(MergeRefused):
    """Nothing was written; the message says what has to be decided first."""


def plan_realign(label: str) -> dict[str, Any]:
    portal = portal_patients_dir().resolve()
    target_dir = portal / label
    with storage.session_scope() as session:
        chart = _chart(session, label)
        if not chart or not chart["uuid"]:
            raise RealignRefused(f"No chart is labelled {label}.")
        rows = session.execute(
            text(
                "SELECT l.id, l.key, a.id, a.sha256, a.size, a.patient_uuid "
                "FROM clinic_locations l JOIN clinic_artifacts a ON a.id = l.artifact_id "
                "WHERE l.kind = 'local' AND l.active = 1 AND substr(l.key, 1, :n) = :prefix"
            ),
            {"n": len(str(portal)) + 1, "prefix": str(portal) + os.sep},
        ).all()
    ours: dict[str, dict[str, Any]] = {}
    others: dict[str, str] = {}
    for location_id, key, artifact_id, sha, size, owner in rows:
        if owner != chart["uuid"]:
            others[key] = owner
            continue
        folder = Path(key).relative_to(portal).parts[0]
        if folder == label:
            continue
        entry = ours.setdefault(key, {"from": key, "folder": folder, "sha256": sha, "size": size,
                                      "artifact_ids": [], "location_ids": []})
        if entry["sha256"] != sha:
            raise RealignRefused(f"{key} is listed with two different byte hashes; resolve that first.")
        entry["artifact_ids"].append(artifact_id)
        entry["location_ids"].append(location_id)
    folders = sorted({e["folder"] for e in ours.values()})
    moves, repoints, bookkeeping = [], [], []
    for entry in sorted(ours.values(), key=lambda e: e["from"]):
        source = Path(entry["from"])
        target = target_dir / source.relative_to(portal / entry["folder"])
        entry["to"] = str(target)
        if target.exists():
            if _sha256_file(target) != entry["sha256"]:
                raise RealignRefused(f"{target} already holds different bytes; nothing is written over it.")
            repoints.append(entry)
        elif source.exists():
            if _sha256_file(source) != entry["sha256"]:
                raise RealignRefused(f"{source} no longer has the bytes the catalogue lists.")
            moves.append(entry)
        else:
            raise RealignRefused(f"{source} is missing and {target} does not exist; the catalogue copy is gone.")
    listed = set(ours)
    for folder in folders:
        root = portal / folder
        for path in sorted(p for p in root.rglob("*") if p.is_file()):
            if str(path) in listed:
                continue
            if path.name in SKIPPED_NAMES:
                bookkeeping.append({"path": str(path)})
            elif str(path) in others:
                raise RealignRefused(f"{path} belongs to another chart; {folder} holds more than one person's files.")
            else:
                raise RealignRefused(f"{path} is not listed in the catalogue; decide where it belongs first.")
    return {"chart": chart, "target_dir": str(target_dir), "folders": folders, "moves": moves,
            "repoints": repoints, "bookkeeping": bookkeeping,
            "nothing_to_do": not (moves or repoints or bookkeeping)}


def apply_realign(plan: dict[str, Any], audit_path: Path) -> dict[str, Any]:
    from backend.clinic_catalogue import _bump
    from backend.clinic_catalogue_reads import _fingerprint

    audit = {"started_at": datetime.now(timezone.utc).isoformat(), "chart": plan["chart"]["label"],
             "chart_uuid": plan["chart"]["uuid"], "moves": [], "repoints": [], "bookkeeping": []}
    _write_json_atomic(audit_path, audit)
    done: list[dict[str, Any]] = []
    committed = False
    try:
        for move in plan["moves"]:
            source, target = Path(move["from"]), Path(move["to"])
            if target.exists():
                raise RealignRefused(f"{target} appeared after planning; nothing overwritten.")
            target.parent.mkdir(parents=True, exist_ok=True)
            os.replace(source, target)
            done.append(move)
            if _sha256_file(target) != move["sha256"]:
                raise RealignRefused(f"{target.name} changed bytes moving.")
            audit["moves"].append({k: move[k] for k in ("from", "to", "sha256", "artifact_ids", "location_ids")})
            _write_json_atomic(audit_path, audit)
        now = int(datetime.now(timezone.utc).timestamp() * 1000)
        with storage.session_scope() as session:
            session.execute(text("BEGIN IMMEDIATE"))
            for entry in plan["moves"] + plan["repoints"]:
                for location_id in entry["location_ids"]:
                    updated = session.execute(
                        text("UPDATE clinic_locations SET key = :to, fingerprint = :fp, verified_at = :now "
                             "WHERE id = :id AND kind = 'local' AND key = :from"),
                        {"to": entry["to"], "fp": _fingerprint(entry["to"]), "now": now,
                         "id": location_id, "from": entry["from"]},
                    ).rowcount
                    if updated != 1:
                        raise RealignRefused(f"location {location_id} did not move ({updated} rows); nothing committed.")
            audit["catalog_revision"] = _bump(session, plan["chart"]["uuid"])
            session.commit()
        committed = True
    except BaseException as error:
        if not committed:
            for move in reversed(done):
                source, target = Path(move["from"]), Path(move["to"])
                source.parent.mkdir(parents=True, exist_ok=True)
                os.replace(target, source)
                if _sha256_file(source) != move["sha256"]:
                    audit.setdefault("rollback_failures", []).append(str(source))
            audit["rollback"] = {"at": datetime.now(timezone.utc).isoformat(), "error": f"{type(error).__name__}: {error}"}
            _write_json_atomic(audit_path, audit)
        raise
    audit["repoints"] = [{k: e[k] for k in ("from", "to", "sha256", "location_ids")} for e in plan["repoints"]]
    _write_json_atomic(audit_path, audit)
    aside = audit_path.with_name(audit_path.stem + "__bookkeeping")
    portal = portal_patients_dir().resolve()
    for entry in plan["bookkeeping"]:
        source = Path(entry["path"])
        if not source.is_file():
            continue
        sha = _sha256_file(source)
        target = _distinct_target(aside / source.relative_to(portal), set())
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(source), str(target))
        audit["bookkeeping"].append({"from": str(source), "to": str(target), "sha256": sha})
        _write_json_atomic(audit_path, audit)
    for repoint in plan["repoints"]:
        # The same bytes already sit at the new path; the old copy is set aside, not deleted.
        source = Path(repoint["from"])
        if source.is_file() and _sha256_file(source) == repoint["sha256"]:
            target = _distinct_target(aside / source.relative_to(portal), set())
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(source), str(target))
            audit["bookkeeping"].append({"from": str(source), "to": str(target), "sha256": repoint["sha256"]})
    for folder in plan["folders"]:
        root = portal / folder
        if root.is_dir():
            for directory in sorted((d for d in root.rglob("*") if d.is_dir()), reverse=True):
                if not any(directory.iterdir()):
                    directory.rmdir()
            if not any(root.iterdir()):
                root.rmdir()
                audit.setdefault("folders_removed", []).append(folder)
    audit["finished_at"] = datetime.now(timezone.utc).isoformat()
    _write_json_atomic(audit_path, audit)
    return audit


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--chart", required=True)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--yes-realign", default="")
    parser.add_argument("--audit", type=Path)
    args = parser.parse_args(argv)
    try:
        plan = plan_realign(args.chart)
    except MergeRefused as refused:
        print(json.dumps({"ok": False, "refused": str(refused)}, indent=2))
        return 2
    summary = {"chart": args.chart, "from_folders": plan["folders"], "moves": len(plan["moves"]),
               "repoints": len(plan["repoints"]), "bookkeeping": len(plan["bookkeeping"]),
               "nothing_to_do": plan["nothing_to_do"]}
    if not args.apply or plan["nothing_to_do"]:
        print(json.dumps({"ok": True, "plan": summary}, indent=2))
        return 0
    if args.yes_realign != args.chart or not args.audit:
        print(json.dumps({"ok": False, "refused": "--apply needs --yes-realign <chart> and --audit <path>"}))
        return 2
    try:
        audit = apply_realign(plan, args.audit)
    except MergeRefused as refused:
        print(json.dumps({"ok": False, "refused": str(refused), "audit": str(args.audit)}, indent=2))
        return 2
    print(json.dumps({"ok": True, "plan": summary, "audit": str(args.audit),
                      "folders_removed": audit.get("folders_removed", [])}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
