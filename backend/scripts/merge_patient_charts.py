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
- Where both charts hold the same catalogue version of a family, the
  duplicate's versions in that family follow the survivor's highest, in their
  own order. Row ids, file keys, bytes and provenance stay as they are. Two
  charts sharing a file key is a question for the operator: the plan refuses.
- File moves and the database step are one unit. If anything fails before the
  commit, every file this run moved goes back to its old path, checked by hash.
- Finder and folder bookkeeping files (``.DS_Store``, ``$meta.json``) never hold
  a chart open. On retirement they are set aside beside the audit file.

The hub reads this catalogue; nothing on the hub needs to run. The legacy
portal sync state and workbench conversations are reported, never rewritten.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sqlalchemy import text

from backend import storage
from backend.patient_identity import parse_canonical_patient_id
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


def _named_owners(session, path: Path, relative: str, provenance: list[str]) -> dict[str, str]:
    """Runs and reports this file names, each mapped to the chart that owns it.

    Read from its name and path, a meta file's ``run_id``, and the catalogue's own
    provenance record for it.
    """
    runs = set(FULL_UUID.findall(relative)) | set(RUN_TOKEN.findall(path.name))
    full = set(FULL_UUID.findall(relative))
    for blob in provenance:
        full |= set(FULL_UUID.findall(blob or ""))
    if path.suffix == ".json" and path.is_file():
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(record, dict) and isinstance(record.get("run_id"), str):
                runs.add(record["run_id"])
        except (OSError, ValueError):
            pass
    owners = {}
    for token in sorted(runs | full):
        for table in ("runs", "reports"):
            if table == "reports" and token not in full:
                continue
            found = session.execute(
                text(f"SELECT id, patient_id FROM {table} WHERE id = :t OR id LIKE :p"),
                {"t": token, "p": token + "%" if len(token) == 8 else token},
            ).all()
            if len(found) == 1:
                owners[f"{table[:-1]} {found[0][0]}"] = found[0][1]
    return owners


def _known_hashes(session, data_dir: Path, uuids: set[str]) -> dict[str, tuple[str, str]]:
    """sha256 -> (what it is, whose), for these charts' report originals and exports."""
    known = {}
    params = {f"u{i}": u for i, u in enumerate(sorted(uuids))}
    marks = ", ".join(f":{k}" for k in params) or "NULL"
    for report_id, stored, owner in session.execute(
        text(f"SELECT id, stored_path, patient_id FROM reports WHERE patient_id IN ({marks})"),
        params,
    ):
        # Stored paths are relative to the checkout that owns DATA_DIR.
        path = Path(stored) if Path(stored).is_absolute() else data_dir.parent / stored
        if path.is_file():
            known[_sha256_file(path)] = (f"report {report_id} original", owner)
    for run_id, owner in session.execute(
        text(f"SELECT id, patient_id FROM runs WHERE patient_id IN ({marks})"), params
    ):
        for name in ("final.md", "final.pdf"):
            path = data_dir / "exports" / run_id / name
            if path.is_file():
                known[_sha256_file(path)] = (f"run {run_id} {name}", owner)
    return known


def _artifact_versions(session, dup_uuid: str, keep_uuid: str) -> list[dict[str, Any]]:
    """The version changes the duplicate's catalogue rows need to join the survivor.

    In a family where any of the duplicate's versions is already the survivor's,
    every duplicate version in that family follows the survivor's highest, in its
    original order. Shifting the whole family keeps the order and keeps a shifted
    version off one of the duplicate's own. A file key both charts use names a
    file the catalogue cannot hold twice, so the merge stops there.
    """
    def rows(uuid):
        return session.execute(
            text(
                "SELECT id, logical_family, version, file_key FROM clinic_artifacts "
                "WHERE patient_uuid = :p"
            ),
            {"p": uuid},
        ).all()

    dup_rows, keep_rows = rows(dup_uuid), rows(keep_uuid)
    shared = sorted({r[3] for r in dup_rows} & {r[3] for r in keep_rows})
    if shared:
        named = ", ".join(shared[:5]) + (", ..." if len(shared) > 5 else "")
        raise MergeRefused(
            f"Both charts have a catalogue file under the same file key ({named}). "
            "One chart cannot hold two files with one key; decide which file keeps it first."
        )
    taken = {(family, version) for _, family, version, _ in keep_rows}
    highest: dict[str, int] = {}
    for family, version in taken:
        highest[family] = max(highest.get(family, 0), version)
    families: dict[str, list[tuple[int, str]]] = {}
    for artifact_id, family, version, _ in dup_rows:
        families.setdefault(family, []).append((version, artifact_id))
    renumbered = []
    for family, versions in sorted(families.items()):
        if not any((family, version) in taken for version, _ in versions):
            continue
        top = highest[family]
        for rank, (version, artifact_id) in enumerate(sorted(versions), 1):
            if version != top + rank:
                renumbered.append(
                    {
                        "artifact_id": artifact_id,
                        "logical_family": family,
                        "from": version,
                        "to": top + rank,
                    }
                )
    return renumbered


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
    to_census_owners: bool = False,
    no_alias: bool = False,
    confirmed: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Everything the merge would do, computed without writing anything.

    With ``to_census_owners`` a file the catalogue files under a third chart goes
    to that chart's folder instead of staying, on the same two-source proof.
    """
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
                "SELECT l.id, l.key, a.patient_uuid, a.id, a.sha256, p.label, a.provenance_json "
                "FROM clinic_locations l JOIN clinic_artifacts a ON a.id = l.artifact_id "
                "LEFT JOIN patients p ON p.id = a.patient_uuid "
                "WHERE l.kind = 'local' AND l.active = 1 AND substr(l.key, 1, :n) = :prefix"
            ),
            {"n": len(prefix), "prefix": prefix},
        ):
            located.setdefault(row[1], []).append(row)

        renumbered = _artifact_versions(session, dup["uuid"], keep["uuid"]) if dup["uuid"] else []
        census = {e[2] for found in located.values() for e in found}
        known = _known_hashes(session, data_dir, person | census)
        moves, stays, others, bookkeeping = [], [], set(), []
        taken: set[Path] = set()
        paths = {Path(k) for k in located} | (
            {p for p in folder.rglob("*") if p.is_file()} if folder.is_dir() else set()
        )

        def same(uuid):
            return keep["uuid"] if uuid in person else uuid

        for path in sorted(paths):
            relative = path.relative_to(folder).as_posix()
            entries = located.get(str(path), [])
            owners = {e[2] for e in entries}
            labels = sorted({e[5] or e[2] for e in entries})
            if path.name in SKIPPED_NAMES:
                # Never holds the chart open. A catalogue row for it moves with
                # the duplicate's other rows; the file is set aside on retirement.
                bookkeeping.append({"path": relative, "catalogued": bool(entries)})
                continue
            if not entries:
                stays.append({"path": relative, "reason": "not in the catalogue"})
                continue
            if len(owners) != 1:
                stays.append({"path": relative, "reason": f"catalogued to {labels}"})
                continue
            owner = same(next(iter(owners)))
            home = survivor if owner == keep["uuid"] else labels[0]
            if owner != keep["uuid"]:
                others.add(home)
                if not to_census_owners or not parse_canonical_patient_id(home):
                    stays.append({"path": relative, "reason": f"catalogued to {home}"})
                    continue
            if relative in holds:
                stays.append({"path": relative, "reason": "held by the operator"})
                continue
            exists = path.is_file()
            target_dir = portal / home / Path(relative).parent
            target = target_dir / rename_in_name(path.name, duplicate, home)
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
            named = _named_owners(
                session, path if exists else target, relative, [e[6] for e in entries]
            )
            evidence = [f"{what} belongs to {home if same(u) == owner else u}"
                        for what, u in named.items()]
            proven = {same(u) for u in named.values()}
            if sha in known:
                evidence.append(f"bytes equal {known[sha][0]}")
                proven.add(same(known[sha][1]))
            ruling = (confirmed or {}).get(relative)
            if ruling is not None and ruling != home:
                stays.append({"path": relative, "reason": f"operator named {ruling}; catalogue says {home}"})
                continue
            if ruling == home:
                # The operator settled a conflict in the catalogue's favour after
                # reading the evidence; it can confirm the census, never invent.
                evidence.append(f"operator confirmed {home} over {sorted(proven - {owner})}")
                proven = {owner}
            if proven - {owner}:
                stays.append({
                    "path": relative,
                    "reason": f"catalogued to {home} but evidence names {sorted(proven - {owner})}",
                })
                continue
            if not proven:
                stays.append({"path": relative, "reason": f"catalogued to {home}; no run, report or hash confirms it"})
                continue
            taken.add(target)
            moves.append(
                {
                    "from": str(path),
                    "to": str(target),
                    "owner": home,
                    "owner_uuid": owner,
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
    elsewhere = holder not in (None, keep["uuid"], dup["uuid"])
    if elsewhere and not existing_alias[1]:
        raise MergeRefused(f"{duplicate} is already an alias of another chart.")
    if elsewhere:
        # Already the catalogue's answer for an ID more than one person wore.
        alias = {
            "register": False,
            "ambiguous": False,
            "reason": f"{duplicate} is already an ambiguous historical alias; it stays that way",
        }
    elif others or no_alias:
        # The old ID named more than one person, so it must resolve to none. An
        # alias row the duplicate already had moves with its rows and is marked
        # ambiguous, which is the catalogue's own word for exactly this.
        alias = {
            "register": False,
            "ambiguous": holder is not None,
            "reason": f"{duplicate} also filed {sorted(others) or 'other charts'}; it names more than one person",
        }
    else:
        alias = {
            "register": holder is None,
            "ambiguous": False,
            "reason": "historical ID of the survivor",
        }

    # The chart goes only once nothing of anyone's is left under its name.
    retire = bool(dup["uuid"]) and not stays
    return {
        "duplicate": dup,
        "survivor": keep,
        "data_dir": str(data_dir),
        "rows": rows,
        "renumbered": renumbered,
        "aliases_repointed": aliases,
        "alias": alias,
        "moves": moves,
        "stays": stays,
        "bookkeeping": bookkeeping,
        "conversations": conversations,
        "legacy_sync_entries": legacy_entries,
        "retire": retire,
        "nothing_to_do": not moves and not retire and not any(rows.values()),
    }


def apply_merge(plan: dict[str, Any], audit_path: Path) -> dict[str, Any]:
    """Carry the plan out, journalling every step to ``audit_path`` as it goes."""
    audit = {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "duplicate": plan["duplicate"],
        "survivor": plan["survivor"],
        "moves": [],
        "rows": {},
        # Exactly what a reversal needs: which rows moved, which alias changed.
        "row_ids": plan["rows"],
        "aliases_repointed": plan["aliases_repointed"],
        "alias": plan["alias"],
        "stays": plan["stays"],
        "renumbered": [],
        "bookkeeping": [],
    }
    _write_json_atomic(audit_path, audit)

    # Files first, then the database, as one unit: a failure before the commit
    # puts back every file this run moved. Only a hard crash can leave
    # locations naming the old path, which the next plan finds and finishes as
    # an already-moved file.
    done: list[dict[str, Any]] = []
    created: list[Path] = []
    committed = False
    try:
        for move in plan["moves"]:
            source, target = Path(move["from"]), Path(move["to"])
            if source.exists():
                if target.exists():
                    raise MergeRefused(f"{target} appeared after planning; nothing overwritten.")
                parent = target.parent
                while not parent.exists():
                    created.append(parent)
                    parent = parent.parent
                target.parent.mkdir(parents=True, exist_ok=True)
                os.replace(source, target)
                done.append(move)
            after = _sha256_file(target)
            if after != move["sha256"]:
                raise MergeRefused(f"{target.name} changed bytes moving ({move['sha256'][:12]} -> {after[:12]}).")
            audit["moves"].append({k: move[k] for k in ("from", "to", "owner", "sha256", "size", "artifact_ids", "location_ids", "evidence")})
            _write_json_atomic(audit_path, audit)
        _commit_rows(plan, audit)
        committed = True
    except BaseException as error:
        if not committed:
            _reverse_moves(done, created, error, audit, audit_path)
        raise

    dup = plan["duplicate"]
    folder = portal_patients_dir().resolve() / dup["label"]
    if plan["retire"]:
        # Finder and folder bookkeeping kept nothing of anyone's; set it aside
        # beside the audit rather than delete it, so the empty folder can go.
        aside = audit_path.with_name(audit_path.stem + "__bookkeeping")
        for entry in plan["bookkeeping"]:
            source = folder / entry["path"]
            if not source.is_file():
                continue
            sha = _sha256_file(source)
            target = _distinct_target(aside / entry["path"], set())
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(source, target)
            if _sha256_file(target) != sha:
                raise MergeRefused(f"{target.name} changed bytes setting it aside.")
            audit["bookkeeping"].append({"from": str(source), "to": str(target), "sha256": sha})
            _write_json_atomic(audit_path, audit)
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


def _reverse_moves(
    done: list[dict[str, Any]],
    created: list[Path],
    error: BaseException,
    audit: dict[str, Any],
    audit_path: Path,
) -> None:
    """Put every file this run moved back at its old path, checked by hash."""
    reversed_moves, failed = [], []
    for move in reversed(done):
        source, target = Path(move["from"]), Path(move["to"])
        try:
            if source.exists():
                raise FileExistsError(f"{source} is occupied again")
            source.parent.mkdir(parents=True, exist_ok=True)
            os.replace(target, source)
            after = _sha256_file(source)
            if after != move["sha256"]:
                raise ValueError(f"bytes changed ({move['sha256'][:12]} -> {after[:12]})")
        except Exception as problem:  # keep going; the original error still propagates
            failed.append({"from": move["to"], "to": move["from"], "error": f"{type(problem).__name__}: {problem}"})
            continue
        reversed_moves.append({"from": move["to"], "to": move["from"], "sha256": after})
    # Folders this run created for the moves go too, once nothing is in them.
    for directory in sorted(created, key=lambda d: len(d.parts), reverse=True):
        try:
            directory.rmdir()
        except OSError:
            pass
    audit["rollback"] = {
        "at": datetime.now(timezone.utc).isoformat(),
        "error": f"{type(error).__name__}: {error}",
        "database": "nothing committed",
        "reversed": reversed_moves,
        "failed": failed,
    }
    _write_json_atomic(audit_path, audit)


def _commit_rows(plan: dict[str, Any], audit: dict[str, Any]) -> None:
    """The database half of the merge, in one write transaction."""
    from backend.clinic_catalogue import _bump
    from backend.clinic_catalogue_reads import _fingerprint

    dup, keep = plan["duplicate"], plan["survivor"]
    with storage.session_scope() as session:
        session.execute(text("BEGIN IMMEDIATE"))
        touched = False
        retire = plan["retire"]
        if dup["uuid"]:
            # Read again under the write lock; the plan was computed without it.
            renumbered = _artifact_versions(session, dup["uuid"], keep["uuid"])
            for sign in (-1, 1):
                # Through negative stand-ins first, so no step lands on a
                # version one of the duplicate's own rows still holds.
                for entry in renumbered:
                    session.execute(
                        text("UPDATE clinic_artifacts SET version = :v WHERE id = :id"),
                        {"v": sign * entry["to"], "id": entry["artifact_id"]},
                    )
            audit["renumbered"] = renumbered
            touched |= bool(renumbered)
            for table, column in _foreign_keys(session):
                if table == "clinic_patient_catalog_state":
                    continue
                if table == "clinic_patient_aliases" and not retire:
                    continue
                moved = session.execute(
                    text(f"UPDATE {table} SET {column} = :keep WHERE {column} = :dup"),
                    {"keep": keep["uuid"], "dup": dup["uuid"]},
                ).rowcount
                audit["rows"][f"{table}.{column}"] = moved
                touched |= bool(moved)
        if retire:
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
        if retire and plan["alias"]["register"]:
            session.execute(
                text(
                    "INSERT INTO clinic_patient_aliases (alias, ambiguous, patient_uuid) "
                    "VALUES (:alias, 0, :keep)"
                ),
                {"alias": dup["label"], "keep": keep["uuid"]},
            )
            touched = True
        if retire and plan["alias"]["ambiguous"]:
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
            charts = {keep["uuid"]} | {m["owner_uuid"] for m in plan["moves"]}
            if dup["uuid"] and not retire:
                charts.add(dup["uuid"])
            audit["catalog_revision"] = _bump(session, charts)
            audit["retired"] = retire
        session.commit()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--duplicate", required=True, help="clinic ID that goes away")
    parser.add_argument("--survivor", required=True, help="clinic ID that keeps everything")
    parser.add_argument("--apply", action="store_true", help="carry the plan out")
    parser.add_argument("--yes-merge", help="DUPLICATE:SURVIVOR, required with --apply")
    parser.add_argument("--hold", action="append", default=[], help="portal path (relative to the duplicate's folder) to leave in place")
    parser.add_argument("--plan-out", type=Path, help="also write the plan JSON here")
    parser.add_argument("--audit", type=Path, help="audit JSON path for --apply")
    parser.add_argument("--to-census-owners", action="store_true", help="move files the catalogue files under a third chart to that chart")
    parser.add_argument("--no-alias", action="store_true", help="the old ID named more than one person: never resolve it to the survivor")
    parser.add_argument("--confirm-owners", type=Path, help="JSON {path relative to the duplicate folder: clinic ID} confirming the catalogue owner where evidence conflicts")
    parser.add_argument("--conversations-dir", type=Path, help="workbench conversations, counted read-only")
    args = parser.parse_args(argv)

    try:
        plan = plan_merge(
            args.duplicate,
            args.survivor,
            holds=tuple(args.hold),
            conversations_dir=args.conversations_dir,
            to_census_owners=args.to_census_owners,
            no_alias=args.no_alias,
            confirmed=json.loads(args.confirm_owners.read_text(encoding="utf-8"))
            if args.confirm_owners
            else None,
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
          f"rows {audit['rows']}, {len(audit['renumbered'])} version(s) renumbered, "
          f"{len(audit['bookkeeping'])} bookkeeping file(s) set aside, "
          f"chart retired: {audit.get('retired', False)}; audit {audit_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
