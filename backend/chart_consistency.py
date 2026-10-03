"""Charts whose contents disagree with each other, found by code alone.

Reports have been filed under the wrong chart at least six times, and each was
found by accident days to months later: one placeholder chart held 415 files of
four people; a woman's report (printing "Female, 9/5/1954") sat on a man born
in 1982; a test upload became an invented patient whose printed ages
contradicted its own birthday. Nothing checked that a chart's contents agree.

This reads the database and each source report's OCR text, with no model call,
and names every disagreement:

- ``birthday``: a report prints a birthday that is not the chart's;
- ``age``: a report's printed age at a session date does not fit the chart's
  birthday (it catches reports whose birthday was blacked out);
- ``sex``: one chart's reports print different sexes;
- ``duplicate``: the same report bytes (sha256) on two charts;
- ``same_visit``: two charts each hold a session on the same date with the
  same deterministic measured values (at least ``SAME_VISIT_MIN_MATCHES``
  matching, none differing), parsed by the engine's own report parsers.

Each violation names the charts, the reports, one plain sentence of evidence,
and a stable key built from content and chart, never from a label, so a
relabel does not make a known violation look new. The clinic is the authority
on who is who: this only tells David, it never blocks or moves anything.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
import threading
from contextlib import closing
from datetime import date
from pathlib import Path
from typing import Any, Iterable

from backend.logging_utils import get_logger

LOGGER = get_logger(__name__)
PARSER_VERSION = 1
SAME_VISIT_MIN_MATCHES = 8
CACHE_NAME = "chart_consistency_cache.json"

# "fF — Female, 9/5/1954 — ID: N/A — Generated: 2/18/2026": the sex and
# birthday WAVi prints in every page header. Anchored on the sex word so the
# "Generated" date never reads as a birthday.
_HEADER_RE = re.compile(r"\b(Male|Female)\s*,\s*(\d{1,2})/(\d{1,2})/(\d{4})\b")
# "Session 1 (8/11/2025) Followup ... 42 yrs": one session-table row.
_SESSION_AGE_RE = re.compile(
    r"(?m)^[^\n]*?\bSession\s+(\d+)\s*\((\d{1,2})/(\d{1,2})/(\d{4})\)[^\n]*?\b(\d{1,3})\s*yrs\b"
)
_SESSION_DATE_RE = re.compile(r"\bSession\s+(\d+)\s*\((\d{1,2})/(\d{1,2})/(\d{4})\)")

_CACHE_LOCK = threading.Lock()


def _iso(month: str | int, day: str | int, year: str | int) -> str | None:
    try:
        return date(int(year), int(month), int(day)).isoformat()
    except ValueError:
        return None


def _chart_birthdate(label: str, column: str | None) -> str | None:
    """The chart's birthday as an ISO date: from the clinic id, else the column."""
    from backend.patient_identity import parse_canonical_patient_id

    parsed = parse_canonical_patient_id(label)
    raw = parsed.birthdate if parsed else (column or "")
    match = re.fullmatch(r"(\d{1,2})-(\d{1,2})-(\d{4})", raw.strip())
    return _iso(*match.groups()) if match else None


def _is_clinic_id(label: str) -> bool:
    from backend.patient_identity import parse_canonical_patient_id

    return parse_canonical_patient_id(label) is not None


def _placeholder(label: str, birthdate: str | None) -> bool:
    initials = label.split("_", 1)[0]
    return "X" in initials or bool(birthdate and birthdate.endswith("-01-01"))


def _age_on(born: str, on: str) -> int:
    b, d = date.fromisoformat(born), date.fromisoformat(on)
    return d.year - b.year - ((d.month, d.day) < (b.month, b.day))


def _ages_that_fit(born: str, on: str) -> set[int]:
    """The true age, and WAVi's: it counts 365-day years, so its age turns over
    a few days before the birthday (LM_12-02-1985 printed 39 on 11/27/2024,
    AN_04-08-1986 printed 40 on 4/3/2026)."""
    days = (date.fromisoformat(on) - date.fromisoformat(born)).days
    return {_age_on(born, on), days // 365}


def _us(iso: str) -> str:
    d = date.fromisoformat(iso)
    return f"{d.month}/{d.day}/{d.year}"


def _chart_date(iso: str) -> str:
    d = date.fromisoformat(iso)
    return f"{d.month:02d}-{d.day:02d}-{d.year}"


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def parse_report_text(text: str) -> dict[str, Any]:
    """Everything the checks need from one report's OCR text.

    The enhanced text holds several OCR streams of the same pages, so each
    field is the set of every reading; a check fails only when no reading
    agrees with the chart, which keeps one misread digit from raising an alarm.
    """
    from backend.council.report_text import (
        _facts_from_report_text_n100_central_frontal,
        _facts_from_report_text_summary,
    )

    sexes: dict[str, int] = {}
    birthdates: dict[str, int] = {}
    for sex, month, day, year in _HEADER_RE.findall(text):
        sexes[sex.lower()] = sexes.get(sex.lower(), 0) + 1
        iso = _iso(month, day, year)
        if iso:
            birthdates[iso] = birthdates.get(iso, 0) + 1
    ages: dict[str, list[int]] = {}
    for _, month, day, year, age in _SESSION_AGE_RE.findall(text):
        iso = _iso(month, day, year)
        if iso and int(age) not in ages.setdefault(iso, []):
            ages[iso].append(int(age))
    session_dates: dict[int, set[str]] = {}
    for index, month, day, year in _SESSION_DATE_RE.findall(text):
        iso = _iso(month, day, year)
        if iso:
            session_dates.setdefault(int(index), set()).add(iso)
    measurements: dict[str, dict[str, list[str]]] = {}
    indices = sorted(session_dates)
    if indices:
        try:
            facts = _facts_from_report_text_summary(text, expected_sessions=indices)
            facts += _facts_from_report_text_n100_central_frontal(text, expected_sessions=indices)
        except Exception:  # an unparseable layout has no measurements, not a crash
            facts = []
        for fact in facts:
            dates = session_dates.get(fact.get("session_index"))
            if not dates or len(dates) != 1:
                continue  # a session with no date, or conflicting dates, cannot be matched
            key = _canonical([fact.get(k) for k in ("fact_type", "metric", "electrode", "condition", "unit")])
            value = _canonical([fact.get(k) for k in ("value", "sd_plus_minus")])
            values = measurements.setdefault(next(iter(dates)), {}).setdefault(key, [])
            if value not in values:
                values.append(value)
    return {
        "v": PARSER_VERSION,
        "sexes": sexes,
        "birthdates": birthdates,
        "ages": ages,
        "measurements": {d: {k: sorted(v) for k, v in m.items()} for d, m in measurements.items()},
    }


def _load_cache(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _save_cache(path: Path, cache: dict[str, Any]) -> None:
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp.write_text(json.dumps(cache, sort_keys=True, separators=(",", ":")), encoding="utf-8")
        os.replace(tmp, path)
    except OSError:
        tmp.unlink(missing_ok=True)


def _resolve(raw: str, root: Path, data_dir: Path) -> Path:
    path = Path(raw)
    if path.is_absolute():
        return path
    for base in (root, data_dir.parent):
        if (base / path).exists():
            return base / path
    return root / path


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_rows(db_path: Path, *, root: Path, data_dir: Path) -> tuple[dict[str, dict], list[dict]]:
    """Charts and source reports, read-only, with each report's paths resolved."""
    with closing(sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)) as db:
        charts = {
            uuid: {
                "uuid": uuid,
                "label": label,
                "birthdate": _chart_birthdate(label, column),
                "clinic": _is_clinic_id(label),
            }
            for uuid, label, column in db.execute("SELECT id, label, birthdate FROM patients")
        }
        shas = dict(db.execute(
            "SELECT source_id, sha256 FROM clinic_artifacts WHERE source_kind = 'report' AND sha256 != ''"
        ))
        rows = db.execute(
            "SELECT id, patient_id, filename, stored_path, extracted_text_path FROM reports ORDER BY created_at, id"
        ).fetchall()
    reports = []
    for report_id, patient_uuid, filename, stored, extracted in rows:
        original = _resolve(stored, root, data_dir)
        enhanced = original.parent / "extracted_enhanced.txt"
        reports.append({
            "id": report_id,
            "patient_uuid": patient_uuid,
            "filename": filename,
            "original": original,
            "text": enhanced if enhanced.is_file() else _resolve(extracted, root, data_dir),
            "sha256": shas.get(report_id),
        })
    return charts, reports


def read_reports(reports: list[dict], *, cache_path: Path | None) -> list[str]:
    """Attach the parsed facts to every report. Returns the ids that could not be read."""
    with _CACHE_LOCK:
        cache = _load_cache(cache_path)
    fresh: dict[str, Any] = {}
    unreadable = []
    for report in reports:
        try:
            raw = report["text"].read_bytes()
            if not report.get("sha256"):
                report["sha256"] = _sha256_file(report["original"])
        except OSError:
            unreadable.append(report["id"])
            report["facts"] = None
            continue
        text_sha = hashlib.sha256(raw).hexdigest()
        facts = cache.get(text_sha)
        if not isinstance(facts, dict) or facts.get("v") != PARSER_VERSION:
            facts = parse_report_text(raw.decode("utf-8", "replace"))
        fresh[text_sha] = facts
        report["facts"] = facts
    if cache_path is not None and fresh.keys() != cache.keys():
        with _CACHE_LOCK:
            _save_cache(cache_path, fresh)
    return unreadable


def _ref(report: dict, charts: dict[str, dict]) -> dict[str, str]:
    return {
        "id": report["id"],
        "patient_id": charts[report["patient_uuid"]]["label"],
        "file": report["filename"],
    }


def _join(items: list[str]) -> str:
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]


def _files(members: list[dict]) -> str:
    """One file name, and how many more there are, so a chart holding six
    hundred copies still reads as one sentence."""
    names = sorted({m["filename"] for m in members}, key=lambda n: (len(n), n))
    more = len(members) - 1
    return names[0] + (f" (and {more} more cop{'y' if more == 1 else 'ies'})" if more else "")


def find_violations(charts: dict[str, dict], reports: list[dict]) -> list[dict[str, Any]]:
    """Every disagreement across the clinic's charts, sorted by key.

    Only charts with a clinic id (XX_MM-DD-YYYY) are checked. The engine has
    issued nothing else since the cutover; the leftovers (DOB-only legacy
    charts, "diag-..." and pipeline test charts) are counted by the CLI, not
    alarmed on every half hour.
    """
    readable = [
        r for r in reports
        if r.get("facts") and r["patient_uuid"] in charts and charts[r["patient_uuid"]].get("clinic", True)
    ]
    # The same bytes twice on one chart are one report for every check.
    groups: dict[tuple[str, str], list[dict]] = {}
    for report in readable:
        groups.setdefault((report["patient_uuid"], report["sha256"]), []).append(report)
    out: list[dict[str, Any]] = []

    for (uuid, sha), members in groups.items():
        chart, facts, first = charts[uuid], members[0]["facts"], members[0]
        refs = [_ref(m, charts) for m in members]
        born = chart["birthdate"]
        printed = facts["birthdates"]
        maybe = " The chart's birthday may be a placeholder." if _placeholder(chart["label"], born) else ""
        if born and printed and born not in printed:
            shown = max(printed, key=printed.get)
            out.append({
                "kind": "birthday",
                "key": f"birthday:{sha[:12]}@{uuid[:8]}",
                "patients": [chart["label"]],
                "reports": refs,
                "evidence": f"{first['filename']} prints the birthday {_us(shown)}, but it is filed on "
                            f"{chart['label']}, whose birthday is {_chart_date(born)}.{maybe}",
            })
            continue  # its ages follow from that birthday; one sentence says it
        if born:
            wrong = [
                (on, ages) for on, ages in sorted(facts["ages"].items())
                if on >= born and not _ages_that_fit(born, on) & set(ages)
            ]
            if wrong:
                said = _join([f"{'/'.join(map(str, ages))} on {_us(on)}" for on, ages in wrong])
                fits = _join([str(_age_on(born, on)) for on, _ in wrong])
                out.append({
                    "kind": "age",
                    "key": f"age:{sha[:12]}@{uuid[:8]}",
                    "patients": [chart["label"]],
                    "reports": refs,
                    "evidence": f"{first['filename']} prints the age {said}, but {chart['label']} "
                                f"(born {_chart_date(born)}) was {fits} then.{maybe}",
                })

    by_chart: dict[str, dict[str, list[dict]]] = {}
    for (uuid, _), members in groups.items():
        sexes = members[0]["facts"]["sexes"]
        if sexes:
            sex = max(sexes, key=sexes.get)
            by_chart.setdefault(uuid, {}).setdefault(sex, []).extend(members)
    for uuid, split in by_chart.items():
        if len(split) < 2:
            continue
        label = charts[uuid]["label"]
        said = "; ".join(
            f"{sex.capitalize()} in {_join(sorted({m['filename'] for m in members}))}"
            for sex, members in sorted(split.items())
        )
        out.append({
            "kind": "sex",
            "key": f"sex:{uuid[:8]}",
            "patients": [label],
            "reports": [_ref(m, charts) for sex in sorted(split) for m in split[sex]],
            "evidence": f"{label}'s reports print different sexes: {said}.",
        })

    by_sha: dict[str, dict[str, list[dict]]] = {}
    shared_sha: set[tuple[str, str]] = set()
    for (uuid, sha), members in groups.items():
        by_sha.setdefault(sha, {})[uuid] = members
    for sha, holders in by_sha.items():
        if len(holders) < 2:
            continue
        uuids = sorted(holders, key=lambda u: charts[u]["label"])
        labels = [charts[u]["label"] for u in uuids]
        shared_sha.update((a, b) for a in uuids for b in uuids if a != b)
        out.append({
            "kind": "duplicate",
            "key": f"duplicate:{sha[:12]}:{'+'.join(sorted(u[:8] for u in uuids))}",
            "patients": labels,
            "reports": [_ref(m, charts) for u in uuids for m in holders[u]],
            "evidence": "The same report file is filed on more than one chart: "
                        + "; ".join(f"{charts[u]['label']} has {_files(holders[u])}" for u in uuids) + ".",
        })

    by_date: dict[str, list[tuple[str, str, dict, list[dict]]]] = {}
    for (uuid, sha), members in groups.items():
        for on, values in members[0]["facts"]["measurements"].items():
            by_date.setdefault(on, []).append((uuid, sha, values, members))
    same_visit: dict[tuple[str, str, str], dict[str, Any]] = {}
    for on, entries in by_date.items():
        for n, (uuid_a, sha_a, left, members_a) in enumerate(entries):
            for uuid_b, sha_b, right, members_b in entries[:n]:
                if uuid_a == uuid_b or sha_a == sha_b or (uuid_a, uuid_b) in shared_sha:
                    # One chart, or two charts already named for sharing the
                    # same bytes: that line says it, and one misfiling should
                    # not read as three.
                    continue
                shared = left.keys() & right.keys()
                matching = sum(1 for k in shared if left[k] == right[k])
                if matching < SAME_VISIT_MIN_MATCHES or matching != len(shared):
                    continue
                pair = tuple(sorted((uuid_a, uuid_b), key=lambda u: charts[u]["label"]))
                entry = same_visit.setdefault(pair, {"dates": {}, "members": {}})
                entry["dates"][on] = max(entry["dates"].get(on, 0), matching)
                for m in members_a + members_b:
                    entry["members"][m["id"]] = m
    # One line per pair of charts, keyed by the dates, so a further shared
    # visit is a new line and a known one stays quiet.
    for (uuid_a, uuid_b), entry in same_visit.items():
        labels = [charts[uuid_a]["label"], charts[uuid_b]["label"]]
        members = sorted(entry["members"].values(), key=lambda m: (charts[m["patient_uuid"]]["label"], m["filename"]))
        dates = sorted(entry["dates"])
        holders = "; ".join(
            f"{charts[u]['label']} has {_files([m for m in members if m['patient_uuid'] == u])}"
            for u in (uuid_a, uuid_b)
        )
        out.append({
            "kind": "same_visit",
            "key": f"same_visit:{'+'.join(sorted(u[:8] for u in (uuid_a, uuid_b)))}:{','.join(dates)}",
            "patients": labels,
            "reports": [_ref(m, charts) for m in members],
            "evidence": f"{_join(labels)} hold the same visit on {_join([_us(d) for d in dates])}: "
                        f"at least {min(entry['dates'].values())} measured values are identical and none "
                        f"differ ({holders}).",
        })
    return sorted(out, key=lambda v: v["key"])


def check(
    db_path: Path,
    *,
    root: Path,
    data_dir: Path,
    cache_path: Path | None,
    exclude_reports: Iterable[str] = (),
) -> dict[str, Any]:
    charts, reports = load_rows(db_path, root=root, data_dir=data_dir)
    unreadable = read_reports(reports, cache_path=cache_path)
    skip = set(exclude_reports)
    violations = find_violations(charts, [r for r in reports if r["id"] not in skip])
    return {
        "charts": len(charts),
        "reports": len(reports),
        "charts_not_checked": sorted(c["label"] for c in charts.values() if not c["clinic"]),
        "unreadable_reports": unreadable,
        "violations": violations,
        "_charts": charts,
        "_reports": reports,
    }


def note_new_filing(patient_uuid: str, report_id: str, *, logger: Any = None) -> list[dict[str, Any]]:
    """After a report is filed, name each disagreement it brought to its chart.

    "New" is the chart's violation keys with this report minus its keys
    without it, so a clean report added to an already mixed chart says
    nothing. Each new one is a ``chart_consistency_violation`` engine event,
    which the workbench health check emails to David. Never raises, never
    blocks the filing.
    """
    from backend import config, storage

    log = logger or LOGGER
    try:
        db_path = Path(storage.engine.url.database)
        data_dir = Path(config.DATA_DIR)
        charts, reports = load_rows(db_path, root=config.REPO_ROOT, data_dir=data_dir)
        read_reports(reports, cache_path=data_dir / CACHE_NAME)

        label = charts.get(patient_uuid, {}).get("label", "")

        def keys(rows: list[dict]) -> dict[str, dict]:
            return {v["key"]: v for v in find_violations(charts, rows) if label in v["patients"]}

        after = keys(reports)
        before = keys([r for r in reports if r["id"] != report_id])
        new = [after[k] for k in sorted(after.keys() - before.keys())]
        for violation in new:
            log.warning(
                "chart_consistency_violation",
                patient_id=label,
                report_id=report_id,
                kind=violation["kind"],
                key=violation["key"],
                evidence=violation["evidence"],
            )
        return new
    except Exception as error:  # a detector must never become a blocker
        try:
            log.warning("chart_consistency_check_failed", patient_uuid=patient_uuid,
                        report_id=report_id, error=f"{type(error).__name__}: {error}")
        except Exception:
            pass
        return []
