"""Move one misfiled source report to the chart it belongs on.

    uv run python -m backend.scripts.reassign_report --report d16db596 --to MF_09-05-1954      # plan only
    uv run python -m backend.scripts.reassign_report --report d16db596 --to MF_09-05-1954 \\
        --apply --yes-reassign d16db596:MF_09-05-1954 --audit reassign.json

The clinic decides who is who; this carries one ruling out and nothing else.
The chart consistency check (backend/scripts/check_chart_consistency.py) is
what usually finds the report.

- The report's row, its catalogue artifact (renumbered after the new chart's
  highest version of the same family when that version is taken) and its
  local catalogue location move to the new chart. Its folder under
  ``data/reports/`` moves whole, every file checked by sha256 on arrival.
- A failed run built only on this report moves with it, so the run and its
  report never name different charts.
- Refused, with the reason, when the report fed a completed run on its current
  chart (analyses built on it belong to that chart's history), when a run on it
  is still in flight, when a failed run also used other reports, or when a
  catalogue artifact was made from such a run. Nothing is moved on a guess.
- The hub publication key stays as it was published, like a chart merge leaves
  it; the catalogue lists the file under the new chart. Upload records and any
  other copies of the same bytes are reported, never touched.
- File move and database step are one unit: if the database step fails, the
  folder goes back to its old path, checked by hash.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sqlalchemy import text

from backend import config, reports, storage
from backend.patient_rekey import _sha256_file, _write_json_atomic


class ReassignRefused(RuntimeError):
    pass


def _base() -> Path:
    """What an older, relative ``data/reports/...`` path is relative to: the engine
    root, which is the data folder's parent."""
    return Path(config.DATA_DIR).resolve().parent


def _resolve(raw: str) -> Path:
    path = Path(raw)
    return path if path.is_absolute() else _base() / path


def _swap_dir(raw: str, old_dir: Path, new_dir: Path) -> str:
    """The same path string, relative or absolute as it was, under the new folder."""
    path = Path(raw)
    if path.is_absolute():
        return str(new_dir / path.resolve().relative_to(old_dir))
    inside = (_base() / path).resolve().relative_to(old_dir)
    return str(new_dir.relative_to(_base()) / inside)


def _files(folder: Path) -> list[dict[str, Any]]:
    return [
        {"path": str(p.relative_to(folder)), "sha256": _sha256_file(p), "size": p.stat().st_size}
        for p in sorted(folder.rglob("*")) if p.is_file()
    ]


def plan_reassign(report_ref: str, target_label: str) -> dict[str, Any]:
    with storage.session_scope() as s:
        found = s.execute(
            text("SELECT r.id, r.patient_id, r.filename, r.stored_path, r.extracted_text_path, p.label "
                 "FROM reports r JOIN patients p ON p.id = r.patient_id WHERE r.id = :ref OR r.id LIKE :prefix"),
            {"ref": report_ref, "prefix": report_ref + "%"},
        ).fetchall()
        if len(found) != 1:
            raise ReassignRefused(f"{len(found)} reports match {report_ref!r}; give the full report id.")
        report_id, source_uuid, filename, stored, extracted, source_label = found[0]
        target = s.execute(text("SELECT id, label FROM patients WHERE label = :l"), {"l": target_label}).fetchone()
        if target is None:
            raise ReassignRefused(f"No chart is labelled {target_label}.")
        target_uuid = target[0]
        if target_uuid == source_uuid:
            raise ReassignRefused(f"{filename} is already on {target_label}.")

        moved_runs, refusals = [], []
        for run_id, status, primary, sources in s.execute(
            text("SELECT id, status, report_id, source_report_ids_json FROM runs "
                 "WHERE report_id = :r OR source_report_ids_json LIKE :like"),
            {"r": report_id, "like": f'%"{report_id}"%'},
        ):
            used = set(json.loads(sources or "[]")) | {primary}
            if report_id not in used:
                continue
            if status == "complete":
                refusals.append(f"run {run_id[:8]} on {source_label} completed on this report; analyses "
                                f"built on it belong to {source_label}'s history, so the report stays")
            elif status != "failed":
                refusals.append(f"run {run_id[:8]} on it is {status}; wait for it to finish")
            elif used != {report_id}:
                refusals.append(f"failed run {run_id[:8]} also used other reports of {source_label}")
            elif s.execute(
                text("SELECT 1 FROM clinic_artifacts WHERE source_id = :run OR provenance_json LIKE :like LIMIT 1"),
                {"run": run_id, "like": f"%{run_id}%"},
            ).first():
                refusals.append(f"failed run {run_id[:8]} has catalogue files made from it")
            else:
                moved_runs.append(run_id)
        if refusals:
            raise ReassignRefused("; ".join(refusals) + ".")

        artifacts = []
        for artifact_id, family, version, file_key, sha in s.execute(
            text("SELECT id, logical_family, version, file_key, sha256 FROM clinic_artifacts "
                 "WHERE source_kind = 'report' AND source_id = :r"),
            {"r": report_id},
        ):
            if s.execute(text("SELECT 1 FROM clinic_artifacts WHERE patient_uuid = :t AND file_key = :k"),
                         {"t": target_uuid, "k": file_key}).first():
                raise ReassignRefused(f"{target_label} already has a catalogue file keyed {file_key}.")
            taken = s.execute(
                text("SELECT MAX(version) FROM clinic_artifacts WHERE patient_uuid = :t AND logical_family = :f"),
                {"t": target_uuid, "f": family},
            ).scalar()
            artifacts.append({"id": artifact_id, "family": family, "sha256": sha, "version_from": version,
                              "version_to": version if taken is None else max(version, taken + 1)})
        locations = [
            {"id": loc_id, "kind": kind, "key": key}
            for loc_id, kind, key in s.execute(
                text("SELECT id, kind, key FROM clinic_locations WHERE artifact_id IN "
                     "(SELECT id FROM clinic_artifacts WHERE source_kind = 'report' AND source_id = :r)"),
                {"r": report_id},
            )
        ]
        shas = sorted({a["sha256"] for a in artifacts if a["sha256"]})
        same_bytes = [
            {"artifact_id": a, "source_kind": k, "patient_id": label, "name": name}
            for sha in shas
            for a, k, label, name in s.execute(
                text("SELECT a.id, a.source_kind, p.label, a.original_name FROM clinic_artifacts a "
                     "JOIN patients p ON p.id = a.patient_uuid WHERE a.sha256 = :sha "
                     "AND NOT (a.source_kind = 'report' AND a.source_id = :r)"),
                {"sha": sha, "r": report_id},
            )
        ]
        uploads = [row[0] for row in s.execute(
            text("SELECT upload_id FROM clinic_upload_items WHERE source_id = :r"), {"r": report_id})]

    old_dir = _resolve(stored).parent
    new_dir = reports.report_dir(target_uuid, report_id).resolve()
    if not old_dir.is_dir():
        raise ReassignRefused(f"The report's folder {old_dir} is missing.")
    if new_dir.exists():
        raise ReassignRefused(f"{new_dir} already exists; nothing overwritten.")
    local = [loc for loc in locations if loc["kind"] == "local"]
    for loc in local:
        if not Path(loc["key"]).resolve().is_relative_to(old_dir.resolve()):
            raise ReassignRefused(f"Catalogue location {loc['key']} is outside the report's folder.")
    return {
        "report": {"id": report_id, "file": filename, "from": source_label, "to": target_label},
        # First, because David may prefer retiring this copy to moving it.
        "same_bytes_elsewhere": same_bytes,
        "source": {"uuid": source_uuid, "label": source_label},
        "target": {"uuid": target_uuid, "label": target_label},
        "folder": {"from": str(old_dir), "to": str(new_dir), "files": _files(old_dir)},
        "report_row": {
            "stored_path": [stored, _swap_dir(stored, old_dir.resolve(), new_dir)],
            "extracted_text_path": [extracted, _swap_dir(extracted, old_dir.resolve(), new_dir)],
        },
        "artifacts": artifacts,
        "locations": [
            dict(loc, to=str(new_dir / Path(loc["key"]).resolve().relative_to(old_dir.resolve())))
            for loc in local
        ],
        "runs": moved_runs,
        "stays": (
            [{"what": f"hub key {loc['key']}", "why": "published keys stay as published; the catalogue "
              f"now lists the file under {target_label}"} for loc in locations if loc["kind"] != "local"]
            + [{"what": f"upload {u}", "why": "the filing record keeps what was submitted"} for u in uploads]
        ),
    }


def _move_folder(source: Path, target: Path, files: list[dict[str, Any]]) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    os.replace(source, target)
    for entry in files:
        after = _sha256_file(target / entry["path"])
        if after != entry["sha256"]:
            raise ReassignRefused(f"{entry['path']} changed bytes moving ({entry['sha256'][:12]} -> {after[:12]}).")


def _commit_rows(plan: dict[str, Any]) -> dict[str, Any]:
    from backend.clinic_catalogue import _bump
    from backend.clinic_catalogue_reads import _fingerprint

    report, target = plan["report"], plan["target"]
    done: dict[str, Any] = {}
    with storage.session_scope() as session:
        session.execute(text("BEGIN IMMEDIATE"))
        still = session.execute(text("SELECT patient_id FROM reports WHERE id = :r"), {"r": report["id"]}).scalar()
        if still != plan["source"]["uuid"]:
            raise ReassignRefused("The report moved after planning; nothing committed.")
        done["reports"] = session.execute(
            text("UPDATE reports SET patient_id = :t, stored_path = :s, extracted_text_path = :e WHERE id = :r"),
            {"t": target["uuid"], "s": plan["report_row"]["stored_path"][1],
             "e": plan["report_row"]["extracted_text_path"][1], "r": report["id"]},
        ).rowcount
        for artifact in plan["artifacts"]:
            session.execute(
                text("UPDATE clinic_artifacts SET patient_uuid = :t, version = :v WHERE id = :id"),
                {"t": target["uuid"], "v": artifact["version_to"], "id": artifact["id"]},
            )
        done["artifacts"] = len(plan["artifacts"])
        for loc in plan["locations"]:
            session.execute(
                text("UPDATE clinic_locations SET key = :k, patient_alias = :a, fingerprint = :fp, "
                     "verified_at = :now WHERE id = :id AND kind = 'local'"),
                {"k": loc["to"], "a": target["label"], "fp": _fingerprint(loc["to"]),
                 "now": int(datetime.now(timezone.utc).timestamp() * 1000), "id": loc["id"]},
            )
        done["locations"] = len(plan["locations"])
        for run_id in plan["runs"]:
            session.execute(text("UPDATE runs SET patient_id = :t WHERE id = :id AND status = 'failed'"),
                            {"t": target["uuid"], "id": run_id})
        done["runs"] = len(plan["runs"])
        done["catalog_revision"] = _bump(session, {plan["source"]["uuid"], target["uuid"]})
        session.commit()
    return done


def apply_reassign(plan: dict[str, Any], audit_path: Path) -> dict[str, Any]:
    """Carry the plan out, journalling to ``audit_path`` as it goes."""
    audit = {"started_at": datetime.now(timezone.utc).isoformat(), "plan": plan}
    _write_json_atomic(audit_path, audit)
    source, target = Path(plan["folder"]["from"]), Path(plan["folder"]["to"])
    moved = False
    try:
        _move_folder(source, target, plan["folder"]["files"])
        moved = True
        audit["folder_moved"] = True
        _write_json_atomic(audit_path, audit)
        audit["rows"] = _commit_rows(plan)
    except BaseException as error:
        if moved or target.exists():
            try:
                if source.exists():
                    raise FileExistsError(f"{source} is occupied again")
                os.replace(target, source)
                bad = [f["path"] for f in plan["folder"]["files"] if _sha256_file(source / f["path"]) != f["sha256"]]
                audit["rollback"] = {"error": f"{type(error).__name__}: {error}", "database": "nothing committed",
                                     "folder": "returned", "changed": bad}
            except Exception as problem:
                audit["rollback"] = {"error": f"{type(error).__name__}: {error}", "database": "nothing committed",
                                     "folder": f"NOT returned: {type(problem).__name__}: {problem}"}
            _write_json_atomic(audit_path, audit)
        raise
    audit["finished_at"] = datetime.now(timezone.utc).isoformat()
    _write_json_atomic(audit_path, audit)
    return audit


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--report", required=True, help="report id (or a unique prefix)")
    parser.add_argument("--to", required=True, help="clinic ID of the chart it belongs on")
    parser.add_argument("--apply", action="store_true", help="carry the plan out")
    parser.add_argument("--yes-reassign", help="REPORT:PATIENT exactly as given, required with --apply")
    parser.add_argument("--plan-out", type=Path, help="also write the plan JSON here")
    parser.add_argument("--audit", type=Path, help="audit JSON path for --apply")
    args = parser.parse_args(argv)
    try:
        plan = plan_reassign(args.report, args.to)
    except ReassignRefused as error:
        print(f"refused: {error}", file=sys.stderr)
        return 2
    rendered = json.dumps(plan, indent=2, ensure_ascii=False)
    if args.plan_out:
        args.plan_out.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    if not args.apply:
        return 0
    if args.yes_reassign != f"{args.report}:{args.to}":
        print(f"refused: --apply needs --yes-reassign {args.report}:{args.to}", file=sys.stderr)
        return 2
    audit_path = args.audit or Path(config.DATA_DIR) / "merge-audits" / (
        f"reassign__{plan['report']['id'][:8]}__{plan['report']['from']}__to__{args.to}__"
        f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
    )
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        audit = apply_reassign(plan, audit_path)
    except ReassignRefused as error:
        print(f"stopped: {error} (audit: {audit_path})", file=sys.stderr)
        return 1
    print(f"moved {plan['report']['file']} from {plan['report']['from']} to {args.to}: "
          f"rows {audit['rows']}; audit {audit_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
