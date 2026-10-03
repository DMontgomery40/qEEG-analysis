"""Whose chart a newly added report belongs to, from evidence already on file.

Two answers, both computed here so a model is handed results, not raw dates:

* ``same_bytes_owners`` — every chart that already holds a source report with
  exactly these bytes. Certain; no model needed. On 2026-10-02 a blacked-out
  report already filed and analyzed on DM_09-23-1982 was dropped again, the
  clinic was asked for a name, and a wrong answer opened a second chart.
* ``near_matches`` — charts holding a report that shares a session date with
  the new one, with how many measured values agree, the printed ages against
  the chart's birthday, sex, and initials the filename hints at.
"""

from __future__ import annotations

from datetime import date, datetime
import json
from pathlib import Path
import re
from typing import Any

from sqlalchemy import select

from . import storage
from .clinic_models import ClinicArtifact
from .council.report_text import (
    _facts_from_report_text_n100_central_frontal,
    _facts_from_report_text_summary,
)
from .patient_identity import parse_canonical_patient_id
from .patient_intake import stored_full_name
from .report_composition import _session_evidence

_MAX_TEXT_BYTES = 4 * 1024 * 1024
_AGE_ROW = re.compile(
    r"\bSession\s+(\d+)\s*\(\s*(\d{1,2}/\d{1,2}/\d{4}|\d{4}-\d{1,2}-\d{1,2})\s*\)"
    r"[^\n]*?\b(\d{1,3})\s*(?:yrs?|years?)\b",
    re.I,
)
_SEX = re.compile(r"\b(Male|Female)\s*,\s*(?:\d{1,2}/\d{1,2}/\d{4}|X|[A-Z#*]{2,})", re.I)
_SEX_LOOSE = re.compile(r"[—\-]\s*(Male|Female)\b", re.I)


def _filed_at(artifact: ClinicArtifact) -> str | None:
    stamp = artifact.uploaded_at or artifact.registered_at
    if not stamp:
        return None
    seconds = stamp / 1000 if stamp > 10**11 else stamp
    return datetime.fromtimestamp(seconds).date().isoformat()


def same_bytes_owners(s, sha256: str, size: int) -> list[dict[str, Any]]:
    """Every chart holding a live source report with exactly these bytes."""
    rows = s.execute(
        select(ClinicArtifact, storage.Patient)
        .join(storage.Patient, storage.Patient.id == ClinicArtifact.patient_uuid)
        .where(
            ClinicArtifact.source_kind == "report",
            ClinicArtifact.sha256 == sha256,
            ClinicArtifact.size == size,
            ClinicArtifact.archived.is_(False),
        )
        .order_by(ClinicArtifact.registered_at, ClinicArtifact.id)
    ).all()
    owners: dict[str, dict[str, Any]] = {}
    for artifact, patient in rows:
        if patient.id in owners or s.get(storage.Report, artifact.source_id) is None:
            continue
        owners[patient.id] = dict(
            patientUuid=patient.id,
            patientId=patient.label,
            name=stored_full_name(patient) or None,
            reportId=artifact.source_id,
            filedAt=_filed_at(artifact),
        )
    return list(owners.values())


def owner_sentence(owners: list[dict[str, Any]]) -> str:
    """One plain sentence naming where these exact bytes already are."""
    parts = []
    for owner in owners:
        name = f" ({owner['name']})" if owner.get("name") else ""
        when = f", filed {_spoken(owner['filedAt'])}" if owner.get("filedAt") else ""
        parts.append(f"{owner['patientId']}'s chart{name}{when}")
    return "This exact report is already in " + " and in ".join(parts) + "."


def _spoken(iso: str) -> str:
    try:
        d = date.fromisoformat(iso)
    except (TypeError, ValueError):
        return str(iso)
    return f"{d.strftime('%B')} {d.day}, {d.year}"


def _iso(raw: str) -> str | None:
    for fmt in ("%m/%d/%Y", "%Y-%m-%d"):
        try:
            return datetime.strptime(raw.strip(), fmt).date().isoformat()
        except ValueError:
            pass
    return None


def session_dates(text: str) -> dict[int, str]:
    """Local session index to ISO date, for sessions with one readable date."""
    return {
        row["local_session_index"]: row["dates"][0]
        for row in _session_evidence(text)
        if len(row["dates"]) == 1 and not row["invalid_dates"]
    }


def printed_ages(text: str) -> dict[str, int]:
    """ISO session date to the age printed beside it."""
    ages: dict[str, int] = {}
    for match in _AGE_ROW.finditer(text):
        when = _iso(match.group(2))
        if when:
            ages.setdefault(when, int(match.group(3)))
    return ages


def printed_sex(text: str) -> str | None:
    match = _SEX.search(text) or _SEX_LOOSE.search(text)
    return match.group(1).lower() if match else None


def _values_by_session(text: str) -> dict[int, dict[str, frozenset[str]]]:
    indices = [r["local_session_index"] for r in _session_evidence(text)]
    facts = _facts_from_report_text_summary(text, expected_sessions=indices)
    facts += _facts_from_report_text_n100_central_frontal(
        text, expected_sessions=indices
    )
    out: dict[int, dict[str, set[str]]] = {}
    for fact in facts:
        local = fact.get("session_index")
        if local is None:
            continue
        key = json.dumps(
            [fact.get(k) for k in ("fact_type", "metric", "electrode", "condition", "unit")],
            sort_keys=True,
        )
        value = json.dumps([fact.get(k) for k in ("value", "sd_plus_minus")])
        out.setdefault(local, {}).setdefault(key, set()).add(value)
    return {
        local: {k: frozenset(v) for k, v in values.items()}
        for local, values in out.items()
    }


def _age_on(birthdate: str, when: str) -> int | None:
    try:
        month, day, year = (int(p) for p in birthdate.split("-"))
        on = date.fromisoformat(when)
    except (TypeError, ValueError, AttributeError):
        return None
    return on.year - year - ((on.month, on.day) < (month, day))


def _chart_birthdate(patient) -> str | None:
    if patient.birthdate:
        return patient.birthdate
    parsed = parse_canonical_patient_id(patient.label)
    return parsed.birthdate if parsed else None


def filename_initials(filename: str) -> str | None:
    """Initials a filename hints at: ``DM_09-23-1982…`` or ``Daniel_Mason…``."""
    stem = re.sub(r"^[0-9a-f]{8}__", "", Path(str(filename or "")).name)
    match = re.match(r"([A-Z]{2})(?:[_\- .]|\d|$)", stem)
    if match:
        return match.group(1)
    match = re.match(r"([A-Z])[a-z]+[ _\-.]+([A-Z])[a-z]+", stem)
    if match:
        return match.group(1) + match.group(2)
    return None


def _date_forms(iso: str) -> list[str]:
    d = date.fromisoformat(iso)
    return [f"{d.month}/{d.day}/{d.year}", d.strftime("%m/%d/%Y"), iso]


def _read_text(path: str) -> str:
    try:
        with Path(path).open("rb") as handle:
            return handle.read(_MAX_TEXT_BYTES).decode("utf-8", "replace")
    except OSError:
        return ""


def near_matches(
    s, text: str, filename: str = "", *, exclude: set[str] | frozenset[str] = frozenset()
) -> list[dict[str, Any]]:
    """Charts holding a report that shares a session date with ``text``.

    Each candidate carries the evidence a person would weigh: for each shared
    session, how many deterministic measured values agree and differ with
    that chart's report; whether the ages printed on the new report fit the
    chart's birthday; sex on both reports; and the filename's initials.
    """
    new_dates = session_dates(text)
    if not new_dates:
        return []
    by_date = {when: local for local, when in new_dates.items()}
    forms = {form for when in by_date for form in _date_forms(when)}
    new_values: dict[int, dict[str, frozenset[str]]] | None = None
    ages = printed_ages(text)
    new_sex = printed_sex(text)
    hinted = filename_initials(filename)
    candidates: dict[str, dict[str, Any]] = {}
    rows = s.execute(
        select(storage.Report, storage.Patient).join(
            storage.Patient, storage.Patient.id == storage.Report.patient_id
        )
    ).all()
    for report, patient in rows:
        if patient.id in exclude:
            continue
        stored = _read_text(report.extracted_text_path)
        if not stored or not any(form in stored for form in forms):
            continue
        shared = {
            when: (by_date[when], local)
            for local, when in session_dates(stored).items()
            if when in by_date
        }
        if not shared:
            continue
        if new_values is None:
            new_values = _values_by_session(text)
        theirs = _values_by_session(stored)
        sessions = []
        for when, (mine_local, their_local) in sorted(shared.items()):
            mine = new_values.get(mine_local, {})
            other = theirs.get(their_local, {})
            common = set(mine) & set(other)
            matched = sum(1 for k in common if mine[k] == other[k])
            sessions.append(
                dict(
                    date=when,
                    reportId=report.id,
                    filedAt=report.created_at.date().isoformat()
                    if report.created_at
                    else None,
                    valuesCompared=len(common),
                    valuesMatched=matched,
                    valuesDiffered=len(common) - matched,
                )
            )
        birthdate = _chart_birthdate(patient)
        checks = []
        for when, printed in sorted(ages.items()):
            expected = _age_on(birthdate, when) if birthdate else None
            if expected is not None:
                checks.append(
                    dict(date=when, printedAge=printed, ageFromChartBirthday=expected)
                )
        if not checks:
            age_fit = "unknown"
        elif all(c["printedAge"] == c["ageFromChartBirthday"] for c in checks):
            age_fit = "fits"
        else:
            age_fit = "does_not_fit"
        parsed = parse_canonical_patient_id(patient.label)
        chart_initials = (
            parsed.first_initial + parsed.last_initial if parsed else None
        )
        current = candidates.get(patient.id)
        best = max(sessions, key=lambda r: (r["valuesMatched"], -r["valuesDiffered"]))
        if current is not None and (
            current["_best"]["valuesMatched"],
            -current["_best"]["valuesDiffered"],
        ) >= (best["valuesMatched"], -best["valuesDiffered"]):
            continue
        candidates[patient.id] = dict(
            patientId=patient.label,
            name=stored_full_name(patient) or None,
            chartBirthdate=birthdate,
            sharedSessions=sessions,
            ageChecks=checks,
            ageFit=age_fit,
            sexOnNewReport=new_sex,
            sexOnChartReport=printed_sex(stored),
            filenameInitials=hinted,
            chartInitials=chart_initials,
            filenameInitialsMatch=None
            if not hinted or not chart_initials
            else hinted.upper() == chart_initials.upper(),
            _best=best,
        )
    out = []
    for candidate in candidates.values():
        candidate.pop("_best")
        out.append(candidate)
    out.sort(key=lambda c: c["patientId"])
    return out
