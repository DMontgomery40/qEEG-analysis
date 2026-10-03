"""Merge one duplicate patient chart into the chart that survives it.

    uv run python -m backend.scripts.merge_patient_charts \\
        --duplicate XX_01-01-2013 --survivor CV_02-12-2012            # plan only
    uv run python -m backend.scripts.merge_patient_charts \\
        --duplicate XX_01-01-2013 --survivor CV_02-12-2012 \\
        --apply --yes-merge XX_01-01-2013:CV_02-12-2012 --audit merge.json

The clinic has already ruled that the two charts are one person; this carries
the ruling out and nothing else. Never run it on a guess.

- Every row keyed to the duplicate's UUID (reports, runs, patient files,
  catalogue artifacts, uploads, producer operations, projections) moves to the
  survivor. The duplicate's patient row is then retired the way the cutover
  retired merged rows: deleted. Its ID reservation stays forever, so the ID is
  never issued again.
- A portal file moves only when the catalogue files it under the duplicate or
  the survivor *and* a run ID or byte hash independently agrees. It lands under
  the survivor's prefix with its bytes untouched; a name already taken gets a
  distinct name, never an overwrite. Its catalogue location moves with it, so
  the clinic and the hub keep reading it. Anything else stays where it is and
  the plan says why.
- The old ID becomes a historical alias of the survivor, so hub keys published
  under it stay bound — unless the old folder also holds another chart's files,
  in which case the ID named more than one person and resolves to none.

The hub reads this catalogue; nothing on the hub needs to run. The legacy
portal sync state and workbench conversations are reported, never rewritten.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sqlalchemy import text

from backend import storage
from backend.patient_rekey import _sha256_file, _write_json_atomic, rename_in_name
from backend.portal_sync import portal_patients_dir

# Bookkeeping the catalogue keeps per chart rather than per row. Aliases are
# repointed and the revision row is retired with the chart.
PER_CHART_TABLES = {"clinic_patient_aliases", "clinic_patient_catalog_state"}
SKIPPED_NAMES = {".DS_Store", "$meta.json"}
RUN_TOKEN = re.compile(r"auto-([0-9a-f]{8})(?![0-9a-f])")
FULL_UUID = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}")


class MergeRefused(RuntimeError):
    """Nothing was written; the message says what has to be decided first."""


def _foreign_keys(session) -> list[tuple[str, str]]:
    """Every column that points at a patient, read from the live schema."""
    rows = session.execute(
        text(
            "SELECT m.name, f.\"from\" FROM sqlite_master m, "
            "pragma_foreign_key_list(m.name) f "
            "WHERE m.type = 'table' AND f.\"table\" = 'patients'"
        )
    ).all()
    return sorted((table, column) for table, column in rows)


def _chart(session, label: str) -> dict[str, Any] | None:
    rows = session.execute(
        text("SELECT id FROM patients WHERE label = :label"), {"label": label}
    ).scalars().all()
    if len(rows) > 1:
        raise MergeRefused(f"{label} is worn by {len(rows)} patient rows; resolve that first.")
    reserved = session.execute(
        text("SELECT 1 FROM patient_id_reservations WHERE patient_id = :label"),
        {"label": label},
    ).first()
    return {"label": label, "uuid": rows[0] if rows else None, "reserved": bool(reserved)}


def _run_owners(session, path: Path, relative: str) -> dict[str, str]:
    """Run IDs this file names, each mapped to the chart that owns the run."""
    tokens = set(FULL_UUID.findall(relative)) | set(RUN_TOKEN.findall(path.name))
    if path.suffix == ".json":
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(record, dict) and isinstance(record.get("run_id"), str):
                tokens.add(record["run_id"])
        except (OSError, ValueError):
            pass
    owners = {}
    for token in sorted(tokens):
        found = session.execute(
            text("SELECT id, patient_id FROM runs WHERE id = :t OR id LIKE :p"),
            {"t": token, "p": token + "%" if len(token) == 8 else token},
        ).all()
        if len(found) == 1:
            owners[found[0][0]] = found[0][1]
    return owners


def _known_hashes(session, data_dir: Path, uuids: set[str]) -> dict[str, str]:
    """sha256 -> what it is, for the merged person's report originals and exports."""
    known = {}
    params = {f"u{i}": u for i, u in enumerate(sorted(uuids))}
    marks = ", ".join(f":{k}" for k in params)
    for report_id, stored in session.execute(
        text(f"SELECT id, stored_path FROM reports WHERE patient_id IN ({marks})"), params
    ):
        # Stored paths are relative to the checkout that owns DATA_DIR.
        path = Path(stored) if Path(stored).is_absolute() else data_dir.parent / stored
        if path.is_file():
            known[_sha256_file(path)] = f"report {report_id} original"
    for (run_id,) in session.execute(
        text(f"SELECT id FROM runs WHERE patient_id IN ({marks})"), params
    ):
        for name in ("final.md", "final.pdf"):
            path = data_dir / "exports" / run_id / name
            if path.is_file():
                known[_sha256_file(path)] = f"run {run_id} {name}"
    return known


def _distinct_target(target: Path, taken: set[Path]) -> Path:
    if not target.exists() and target not in taken:
        return target
    n = 2
    while True:
        candidate = target.with_name(f"{target.stem}__merged-{n}{target.suffix}")
        if not candidate.exists() and candidate not in taken:
            return candidate
        n += 1


def plan_merge(
    duplicate: str,
    survivor: str,
    *,
    holds: tuple[str, ...] = (),
    conversations_dir: Path | None = None,
) -> dict[str, Any]:
    """Everything the merge would do, computed without writing anything."""
    if duplicate == survivor:
        raise MergeRefused("A chart cannot be merged into itself.")
    data_dir = Path(storage.DATA_DIR).resolve()
    portal = portal_patients_dir().resolve()
    if not portal.is_relative_to(data_dir):
        raise MergeRefused(f"{portal} is outside DATA_DIR {data_dir}; refusing to touch it.")

    with storage.session_scope() as session:
        dup = _chart(session, duplicate)
        keep = _chart(session, survivor)
        if keep["uuid"] is None:
            raise MergeRefused(f"The survivor {survivor} has no patient row.")
        if not dup["reserved"]:
            raise MergeRefused(f"{duplicate} has no ID reservation; this is not a clinic ID.")
        person = {keep["uuid"]} | ({dup["uuid"]} if dup["uuid"] else set())

        rows: dict[str, list[str]] = {}
        if dup["uuid"]:
            for table, column in _foreign_keys(session):
                if table in PER_CHART_TABLES:
                    continue
                rows[f"{table}.{column}"] = session.execute(
                    text(f"SELECT id FROM {table} WHERE {column} = :id"),
                    {"id": dup["uuid"]},
                ).scalars().all()
        aliases = (
            session.execute(
                text("SELECT alias FROM clinic_patient_aliases WHERE patient_uuid = :id"),
                {"id": dup["uuid"]},
            ).scalars().all()
            if dup["uuid"]
            else []
        )
        existing_alias = session.execute(
            text("SELECT patient_uuid, ambiguous FROM clinic_patient_aliases WHERE alias = :a"),
            {"a": duplicate},
        ).first()

        folder = portal / duplicate
        prefix = str(folder) + os.sep
        located: dict[str, list[tuple]] = {}
        for row in session.execute(
            text(
                "SELECT l.id, l.key, a.patient_uuid, a.id, a.sha256, p.label "
                "FROM clinic_locations l JOIN clinic_artifacts a ON a.id = l.artifact_id "
                "LEFT JOIN patients p ON p.id = a.patient_uuid "
                "WHERE l.kind = 'local' AND l.active = 1 AND substr(l.key, 1, :n) = :prefix"
            ),
            {"n": len(prefix), "prefix": prefix},
        ):
            located.setdefault(row[1], []).append(row)

        known = _known_hashes(session, data_dir, person)
        moves, stays, others = [], [], set()
        taken: set[Path] = set()
        paths = {Path(k) for k in located} | (
            {p for p in folder.rglob("*") if p.is_file()} if folder.is_dir() else set()
        )
        for path in sorted(paths):
            relative = path.relative_to(folder).as_posix()
            entries = located.get(str(path), [])
            owners = {e[2] for e in entries}
            labels = sorted({e[5] or e[2] for e in entries})
            if path.name in SKIPPED_NAMES:
                stays.append({"path": relative, "reason": "folder bookkeeping"})
                continue
            if not entries:
                stays.append({"path": relative, "reason": "not in the catalogue"})
                continue
            if len(owners) != 1:
                stays.append({"path": relative, "reason": f"catalogued to {labels}"})
                continue
            owner = next(iter(owners))
            if owner not in person:
                others.add(labels[0])
                stays.append({"path": relative, "reason": f"catalogued to {labels[0]}"})
                continue
            if relative in holds:
                stays.append({"path": relative, "reason": "held by the operator"})
                continue
            exists = path.is_file()
            target_dir = portal / survivor / Path(relative).parent
            target = target_dir / rename_in_name(path.name, duplicate, survivor)
            sha = _sha256_file(path) if exists else entries[0][4]
            if not exists:
                # A resumed run: the file already crossed and only its catalogue
                # location still names the old path.
                landed = [
                    p for p in target_dir.glob(f"{target.stem}*{target.suffix}")
                    if p.is_file() and _sha256_file(p) == sha
                ]
                if len(landed) != 1:
                    stays.append({"path": relative, "reason": "catalogued here but the file is gone"})
                    continue
                target = landed[0]
            else:
                target = _distinct_target(target, taken)
            run_owners = _run_owners(session, target if not exists else path, relative)
            evidence = [f"run {run} belongs to {'survivor' if u == keep['uuid'] else 'duplicate' if u == dup['uuid'] else u}"
                        for run, u in run_owners.items()]
            if sha in known:
                evidence.append(f"bytes equal {known[sha]}")
            strangers = {u for u in run_owners.values() if u not in person}
            if strangers:
                stays.append({"path": relative, "reason": f"catalogue says this chart but run evidence says {sorted(strangers)}"})
                continue
            if not evidence:
                stays.append({"path": relative, "reason": "catalogued to this person but no run or hash confirms it"})
                continue
            taken.add(target)
            moves.append(
                {
                    "from": str(path),
                    "to": str(target),
                    "sha256": sha,
                    "size": path.stat().st_size if exists else None,
                    "already_moved": not exists,
                    "location_ids": [e[0] for e in entries],
                    "artifact_ids": sorted({e[3] for e in entries}),
                    "evidence": evidence,
                }
            )

    conversations = []
    if conversations_dir is not None and conversations_dir.is_dir():
        for path in sorted(conversations_dir.glob("*.json")):
            try:
                record = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            if isinstance(record, dict) and duplicate in (
                record.get("patient_id"), record.get("patient_label")
            ):
                conversations.append(path.name)

    sync_state = portal / ".qeeg_portal_sync_state.json"
    legacy_entries = 0
    if sync_state.is_file():
        try:
            files = json.loads(sync_state.read_text(encoding="utf-8")).get("files") or {}
            legacy_entries = sum(1 for k in files if k.split("/", 1)[0] == duplicate)
        except (OSError, ValueError, AttributeError):
            pass

    holder = existing_alias[0] if existing_alias else None
    if holder not in (None, keep["uuid"], dup["uuid"]):
        raise MergeRefused(f"{duplicate} is already an alias of another chart.")
    if others:
        # The old ID named more than one person, so it must resolve to none. An
        # alias row the duplicate already had moves with its rows and is marked
        # ambiguous, which is the catalogue's own word for exactly this.
        alias = {
            "register": False,
            "ambiguous": holder is not None,
            "reason": f"{duplicate} also filed {sorted(others)}; it names more than one person",
        }
    else:
        alias = {
            "register": holder is None,
            "ambiguous": False,
            "reason": "historical ID of the survivor",
        }

    return {
        "duplicate": dup,
        "survivor": keep,
        "data_dir": str(data_dir),
        "rows": rows,
        "aliases_repointed": aliases,
        "alias": alias,
        "moves": moves,
        "stays": stays,
        "conversations": conversations,
        "legacy_sync_entries": legacy_entries,
        "nothing_to_do": not dup["uuid"] and not moves and not alias["register"]
        and not (alias["ambiguous"] and holder == dup["uuid"]),
    }


def apply_merge(plan: dict[str, Any], audit_path: Path) -> dict[str, Any]:
    """Carry the plan out, journalling every step to ``audit_path`` as it goes."""
    from backend.clinic_catalogue import _bump
    from backend.clinic_catalogue_reads import _fingerprint

    audit = {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "duplicate": plan["duplicate"],
        "survivor": plan["survivor"],
        "moves": [],
        "rows": {},
        "stays": plan["stays"],
    }
    _write_json_atomic(audit_path, audit)

    # Files first: a crash here leaves locations naming the old path, which the
    # next plan finds and finishes as an already-moved file.
    for move in plan["moves"]:
        source, target = Path(move["from"]), Path(move["to"])
        if source.exists():
            if target.exists():
                raise MergeRefused(f"{target} appeared after planning; nothing overwritten.")
            target.parent.mkdir(parents=True, exist_ok=True)
            os.replace(source, target)
        after = _sha256_file(target)
        if after != move["sha256"]:
            raise MergeRefused(f"{target.name} changed bytes moving ({move['sha256'][:12]} -> {after[:12]}).")
        audit["moves"].append({k: move[k] for k in ("from", "to", "sha256", "size", "artifact_ids", "evidence")})
        _write_json_atomic(audit_path, audit)

    dup, keep = plan["duplicate"], plan["survivor"]
    with storage.session_scope() as session:
        session.execute(text("BEGIN IMMEDIATE"))
        touched = False
        if dup["uuid"]:
            for table, column in _foreign_keys(session):
                if table == "clinic_patient_catalog_state":
                    continue
                moved = session.execute(
                    text(f"UPDATE {table} SET {column} = :keep WHERE {column} = :dup"),
                    {"keep": keep["uuid"], "dup": dup["uuid"]},
                ).rowcount
                audit["rows"][f"{table}.{column}"] = moved
            for table, column in _foreign_keys(session):
                if table == "clinic_patient_catalog_state":
                    continue
                left = session.execute(
                    text(f"SELECT count(*) FROM {table} WHERE {column} = :dup"),
                    {"dup": dup["uuid"]},
                ).scalar()
                if left:
                    raise MergeRefused(f"{table}.{column} still names the duplicate.")
            session.execute(
                text("DELETE FROM clinic_patient_catalog_state WHERE patient_uuid = :dup"),
                {"dup": dup["uuid"]},
            )
            session.execute(text("DELETE FROM patients WHERE id = :dup"), {"dup": dup["uuid"]})
            touched = True
        for move in plan["moves"]:
            for location_id in move["location_ids"]:
                session.execute(
                    text(
                        "UPDATE clinic_locations SET key = :to, fingerprint = :fp, "
                        "verified_at = :now WHERE id = :id AND kind = 'local'"
                    ),
                    {
                        "to": move["to"],
                        "fp": _fingerprint(move["to"]),
                        "now": int(datetime.now(timezone.utc).timestamp() * 1000),
                        "id": location_id,
                    },
                )
            touched = True
        if plan["alias"]["register"]:
            session.execute(
                text(
                    "INSERT INTO clinic_patient_aliases (alias, ambiguous, patient_uuid) "
                    "VALUES (:alias, 0, :keep)"
                ),
                {"alias": dup["label"], "keep": keep["uuid"]},
            )
            touched = True
        if plan["alias"]["ambiguous"]:
            touched |= bool(session.execute(
                text("UPDATE clinic_patient_aliases SET ambiguous = 1 "
                     "WHERE alias = :alias AND ambiguous = 0"),
                {"alias": dup["label"]},
            ).rowcount)
        if not session.execute(
            text("SELECT 1 FROM patient_id_reservations WHERE patient_id = :label"),
            {"label": dup["label"]},
        ).first():
            raise MergeRefused(f"The reservation for {dup['label']} is missing; nothing committed.")
        if touched:
            audit["catalog_revision"] = _bump(session, {keep["uuid"]})
        session.commit()

    folder = portal_patients_dir().resolve() / dup["label"]
    if folder.is_dir():
        for directory in sorted((d for d in folder.rglob("*") if d.is_dir()), reverse=True):
            if not any(directory.iterdir()):
                directory.rmdir()
        if not any(folder.iterdir()):
            folder.rmdir()
            audit["duplicate_folder_removed"] = True
    audit["finished_at"] = datetime.now(timezone.utc).isoformat()
    _write_json_atomic(audit_path, audit)
    return audit


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--duplicate", required=True, help="clinic ID that goes away")
    parser.add_argument("--survivor", required=True, help="clinic ID that keeps everything")
    parser.add_argument("--apply", action="store_true", help="carry the plan out")
    parser.add_argument("--yes-merge", help="DUPLICATE:SURVIVOR, required with --apply")
    parser.add_argument("--hold", action="append", default=[], help="portal path (relative to the duplicate's folder) to leave in place")
    parser.add_argument("--plan-out", type=Path, help="also write the plan JSON here")
    parser.add_argument("--audit", type=Path, help="audit JSON path for --apply")
    parser.add_argument("--conversations-dir", type=Path, help="workbench conversations, counted read-only")
    args = parser.parse_args(argv)

    try:
        plan = plan_merge(
            args.duplicate,
            args.survivor,
            holds=tuple(args.hold),
            conversations_dir=args.conversations_dir,
        )
    except MergeRefused as error:
        print(f"refused: {error}", file=sys.stderr)
        return 2
    rendered = json.dumps(plan, indent=2, ensure_ascii=False)
    if args.plan_out:
        args.plan_out.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    if not args.apply:
        return 0
    if args.yes_merge != f"{args.duplicate}:{args.survivor}":
        print(f"refused: --apply needs --yes-merge {args.duplicate}:{args.survivor}", file=sys.stderr)
        return 2
    if plan["conversations"]:
        print(f"refused: {len(plan['conversations'])} workbench conversation(s) are filed "
              f"under {args.duplicate}; repoint them first.", file=sys.stderr)
        return 2
    audit_path = args.audit or Path(plan["data_dir"]) / "merge-audits" / (
        f"{args.duplicate}__into__{args.survivor}__"
        f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
    )
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        audit = apply_merge(plan, audit_path)
    except MergeRefused as error:
        print(f"stopped: {error} (audit so far: {audit_path})", file=sys.stderr)
        return 1
    print(f"merged {args.duplicate} into {args.survivor}: {len(audit['moves'])} file(s) moved, "
          f"rows {audit['rows']}; audit {audit_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
