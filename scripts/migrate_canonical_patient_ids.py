#!/usr/bin/env python3
"""Move the whole clinic onto canonical patient IDs, once.

Legacy IDs are date-of-birth keys — ``09-23-1982-0`` — where the trailing number
counts patients sharing a birthdate, starting at zero. The canonical ID is
``XX_MM-DD-YYYY[_N]``: two initials, the birthdate, and a collision ordinal that
starts at one and is left off when it is one. The two ordinals do not
correspond: the legacy one counts collisions on the birthdate alone, the
canonical one counts collisions on initials *and* birthdate. Two people born the
same day with different initials both end up unsuffixed. So this migrator never
carries a legacy ordinal across: the dry run works out each patient's ordinal
from the initials-and-birthdate collisions it can see, and ``--apply`` writes
every resulting ID through ``reserve_canonical_patient_id`` so the reservation
table remembers it forever. An ID that is only worn by a live row would become
reissuable the moment that row was deleted or relabelled.

Run it twice: ``--dry-run`` first, which writes nothing and tells you exactly
what it would do and what it cannot decide, and ``--apply`` inside a maintenance
window with every writer stopped.

``--dry-run`` fails — non-zero, loudly — while any real patient's identity is
untrustworthy or any two patients would land on the same ID. That failure list
is the point: it names the folder, says what is missing, and says what someone
has to supply. Nothing is guessed.

Every patient is classified into exactly one bucket, and an unclassified row is
itself a failure. That is what stops a stray test row from being migrated as a
person, and what stops a real person from being quietly skipped.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sqlite3
import sys
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend import patient_rekey  # noqa: E402
from backend.patient_identity import (  # noqa: E402
    CANONICAL_PATIENT_ID_RE,
    canonical_patient_id,
    normalize_birthdate,
    parse_canonical_patient_id,
    reserve_canonical_patient_id,
)

# The legacy key: birthdate plus a zero-based collision counter.
LEGACY_ID_RE = re.compile(r"^(?P<mm>\d{1,2})-(?P<dd>\d{1,2})-(?P<yyyy>\d{4})(?:-(?P<n>\d+))?$")

# Rows whose stored files live in a throwaway directory were written by a test
# run, not by the clinic. The engine's DATA_DIR is relative, so a suite run with
# the wrong working directory lands in the live database.
TEST_PATH_MARKERS = ("pytest-of-", "/tmp/qeeg-hang-repro")

# The synthetic record David built to exercise the identity path end to end.
QA_FIXTURE_LABELS = frozenset({"02-29-1984-0"})

# Rows David has previously described as his own test data rather than clinic
# patients. They are NOT carved out on that basis alone — the dry run reports
# them as candidates and keeps failing until he confirms, because getting this
# wrong deletes a real patient. Confirm with --qa-candidates-confirmed.
QA_CANDIDATE_LABELS = frozenset({"01-01-1983-0", "01-01-2013-0"})

PRODUCTION_ROOTS = (
    Path("/Users/davidmontgomery/qEEG-analysis/data"),
    Path("/Users/davidmontgomery/thrylen"),
    Path("/Users/davidmontgomery/qeeg-clinic-workbench/clinic/server/data"),
)

BUCKET_MIGRATE = "migrate"
BUCKET_UNRESOLVED = "unresolved"
BUCKET_QA_FIXTURE = "qa_fixture"
BUCKET_TEST_POLLUTION = "test_pollution"
BUCKET_NON_PATIENT = "non_patient_row"
BUCKET_DUPLICATE = "duplicate_of_survivor"
BUCKET_ALREADY_CANONICAL = "already_canonical"


class MigrationStop(RuntimeError):
    """Something a person has to decide. Never guessed past."""


# --------------------------------------------------------------------------- #
# Reading the world (never writing it)
# --------------------------------------------------------------------------- #


@dataclass
class PatientRow:
    uuid: str
    label: str
    created_at: str
    report_count: int
    run_count: int
    file_count: int
    stored_paths: list[str] = field(default_factory=list)


def read_patient_rows(db_path: Path) -> list[PatientRow]:
    """Read every patient row. Opened read-only so a dry run cannot write."""
    uri = f"file:{db_path}?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            """
            SELECT p.id, p.label, p.created_at,
                   (SELECT COUNT(*) FROM reports r WHERE r.patient_id = p.id) AS nrep,
                   (SELECT COUNT(*) FROM runs u WHERE u.patient_id = p.id) AS nrun,
                   (SELECT COUNT(*) FROM patient_files f WHERE f.patient_id = p.id)
                       AS nfile
            FROM patients p ORDER BY p.label, p.created_at
            """
        ).fetchall()
        out: list[PatientRow] = []
        for row in rows:
            paths = [
                str(r[0])
                for r in conn.execute(
                    "SELECT stored_path FROM reports WHERE patient_id = ?", (row["id"],)
                )
            ]
            out.append(
                PatientRow(
                    uuid=row["id"],
                    label=row["label"],
                    created_at=str(row["created_at"]),
                    report_count=row["nrep"],
                    run_count=row["nrun"],
                    file_count=row["nfile"],
                    stored_paths=paths,
                )
            )
        return out
    finally:
        conn.close()


def read_folder_identity(portal_root: Path, patient_id: str) -> dict[str, Any] | None:
    meta_path = portal_root / patient_id / "$meta.json"
    if not meta_path.is_file():
        return None
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    identity = meta.get("identity") if isinstance(meta, dict) else None
    return identity if isinstance(identity, dict) else None


def read_sync_state(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        state = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return state if isinstance(state, dict) else {}


_FILENAME_INITIALS_RE = re.compile(r"^(?P<a>[A-Z])(?P<b>[A-Z])_")
# The same two-letter token appearing after a separator inside a longer name,
# which is how a misfiled report announces whose it really is.
_EMBEDDED_INITIALS_RE = re.compile(r"(?:^|[_\-])(?P<a>[A-Z])(?P<b>[A-Z])_(?=[A-Za-z0-9])")


def filename_initials(portal_root: Path, patient_id: str) -> set[tuple[str, str]]:
    """Initials tokens that appear at the head of a source filename.

    Original report PDFs are named by the clinic — ``PG_femal_TBI...pdf`` — so
    the two letters in front are independent evidence of who the folder holds.
    """
    folder = portal_root / patient_id
    if not folder.is_dir():
        return set()
    found: set[tuple[str, str]] = set()
    for entry in folder.iterdir():
        if not entry.is_file():
            continue
        match = _FILENAME_INITIALS_RE.match(entry.name)
        if match:
            found.add((match.group("a"), match.group("b")))
    return found


# Two-letter tokens that are conditions or labels, not people.
NON_IDENTITY_TOKENS = frozenset({"TX", "QC", "MR", "CT", "AI", "QA", "ID", "PD"})


def embedded_initials(portal_root: Path, patient_id: str) -> dict[str, list[str]]:
    """Initials tokens buried inside this folder's filenames, and which files.

    A folder holding ``03-05-2010-0__DK_20Tx_...pdf`` is holding DK's report,
    whoever the folder is named for. That is how a mixed-patient aggregate
    announces itself, and it is never something to migrate past quietly.
    """
    folder = portal_root / patient_id
    if not folder.is_dir():
        return {}
    found: dict[str, list[str]] = defaultdict(list)
    for entry in folder.iterdir():
        if not entry.is_file():
            continue
        # Strip the folder's own ID first so its digits cannot create tokens.
        stem = entry.name.replace(patient_id, "")
        for match in _EMBEDDED_INITIALS_RE.finditer(stem):
            token = match.group("a") + match.group("b")
            if token in NON_IDENTITY_TOKENS:
                continue
            found[token].append(entry.name)
    return {token: sorted(names) for token, names in found.items()}


# Words that show up in uploaded report filenames and are not anybody's name.
_NON_NAME_WORDS = frozenset(
    {
        "qeeg", "eeg", "initial", "intial", "final", "mid", "pdf", "tx", "redacted",
        "report", "treatments", "treatment", "male", "female", "femal", "the",
        "and", "patient", "new", "copy", "scan", "brain", "injury", "covid",
        "long", "dementia", "autism", "anxiety", "mci", "tbi",
    }
)


def _initials_tokens_in(name: str) -> set[tuple[str, str]]:
    """Explicit ``XX_`` initials tokens written into a filename."""
    found = set()
    head = _FILENAME_INITIALS_RE.match(name)
    if head:
        found.add((head.group("a"), head.group("b")))
    for match in _EMBEDDED_INITIALS_RE.finditer(name):
        found.add((match.group("a"), match.group("b")))
    return found


def title_name_initials(title: str) -> set[str]:
    """First letters of the words in an uploaded report filename that read as names."""
    stem = re.sub(r"\.(pdf|md|mp4|docx)$", "", title, flags=re.IGNORECASE)
    letters: set[str] = set()
    for word in re.findall(r"[A-Za-z]+", stem):
        if word.lower() in _NON_NAME_WORDS or len(word) < 1:
            continue
        letters.add(word[0].upper())
    return letters


def read_conversation_titles(conversations_dir: Path) -> dict[str, list[str]]:
    """Conversation titles per legacy ID — the uploaded filenames carry names."""
    out: dict[str, list[str]] = defaultdict(list)
    if not conversations_dir.is_dir():
        return out
    for path in sorted(conversations_dir.glob("*.json")):
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if not isinstance(record, dict):
            continue
        filed = record.get("patient_id") or record.get("patient_label")
        if isinstance(filed, str) and filed:
            out[filed].append(str(record.get("title") or ""))
    return out


def parse_legacy_id(value: str) -> tuple[str, int] | None:
    """Return (normalized birthdate, legacy ordinal) for a legacy storage key."""
    match = LEGACY_ID_RE.match((value or "").strip())
    if not match:
        return None
    try:
        birthdate = normalize_birthdate(
            f"{int(match.group('mm')):02d}-{int(match.group('dd')):02d}-"
            f"{match.group('yyyy')}"
        )
    except Exception:
        return None
    return birthdate, int(match.group("n") or 0)


# --------------------------------------------------------------------------- #
# Classification — every row lands in exactly one bucket
# --------------------------------------------------------------------------- #


@dataclass
class Classified:
    key: str
    bucket: str
    reason: str
    uuid: str | None = None
    label: str | None = None
    birthdate: str | None = None
    initials: tuple[str, str] | None = None
    evidence: dict[str, Any] = field(default_factory=dict)
    new_id: str | None = None
    survivor_uuid: str | None = None
    needs: str | None = None


def _looks_like_test_row(row: PatientRow) -> bool:
    return any(
        marker in path for path in row.stored_paths for marker in TEST_PATH_MARKERS
    )


def classify_patient_rows(
    rows: Iterable[PatientRow], *, portal_root: Path, qa_candidates_confirmed: bool = False
) -> list[Classified]:
    """Sort every database row into one bucket, explaining each decision."""
    rows = list(rows)
    by_label: dict[str, list[PatientRow]] = defaultdict(list)
    for row in rows:
        by_label[row.label].append(row)

    out: list[Classified] = []
    for row in rows:
        key = f"{row.label}#{row.uuid[:8]}"
        legacy = parse_legacy_id(row.label)
        has_folder = (portal_root / row.label).is_dir()

        if _looks_like_test_row(row):
            out.append(
                Classified(
                    key=key,
                    bucket=BUCKET_TEST_POLLUTION,
                    reason=(
                        "report files live in a throwaway test directory, so this row "
                        "was written by a suite run against the live database"
                    ),
                    uuid=row.uuid,
                    label=row.label,
                    evidence={"stored_paths": row.stored_paths[:3]},
                )
            )
            continue

        if parse_canonical_patient_id(row.label) is not None:
            out.append(
                Classified(
                    key=key,
                    bucket=BUCKET_ALREADY_CANONICAL,
                    reason="already wearing a canonical clinic ID; nothing to do",
                    uuid=row.uuid,
                    label=row.label,
                )
            )
            continue

        if row.label in QA_CANDIDATE_LABELS and qa_candidates_confirmed:
            out.append(
                Classified(
                    key=key,
                    bucket=BUCKET_QA_FIXTURE,
                    reason=(
                        "confirmed by the owner as his own test data, not a clinic "
                        "patient: bundled for rollback, then removed from the roster"
                    ),
                    uuid=row.uuid,
                    label=row.label,
                    evidence={"reports": row.report_count, "runs": row.run_count},
                )
            )
            continue

        if row.label in QA_FIXTURE_LABELS:
            out.append(
                Classified(
                    key=key,
                    bucket=BUCKET_QA_FIXTURE,
                    reason=(
                        "the synthetic identity-path QA record: bundled for rollback, "
                        "removed from the roster, never given a reservation"
                    ),
                    uuid=row.uuid,
                    label=row.label,
                    evidence={"reports": row.report_count, "runs": row.run_count},
                )
            )
            continue

        if legacy is None:
            out.append(
                Classified(
                    key=key,
                    bucket=BUCKET_NON_PATIENT,
                    reason=(
                        "the label is not a patient key at all — a scratch, diagnostic, "
                        "or browser-test row with no portal folder"
                    ),
                    uuid=row.uuid,
                    label=row.label,
                    evidence={
                        "reports": row.report_count,
                        "runs": row.run_count,
                        "has_portal_folder": has_folder,
                    },
                    needs=(
                        None
                        if not has_folder
                        else "a decision: it has a portal folder despite the label"
                    ),
                )
            )
            continue

        siblings = [other for other in by_label[row.label] if other.uuid != row.uuid]
        real_siblings = [s for s in siblings if not _looks_like_test_row(s)]
        if real_siblings:
            survivor = _pick_survivor([row, *real_siblings])
            if survivor.uuid != row.uuid:
                out.append(
                    Classified(
                        key=key,
                        bucket=BUCKET_DUPLICATE,
                        reason=(
                            f"a second row wearing the same label; reports, runs and "
                            f"patient files move to {survivor.uuid[:8]} before relabel"
                        ),
                        uuid=row.uuid,
                        label=row.label,
                        survivor_uuid=survivor.uuid,
                        evidence={
                            "reports": row.report_count,
                            "runs": row.run_count,
                            "files": row.file_count,
                        },
                    )
                )
                continue

        out.append(
            Classified(
                key=key,
                bucket=BUCKET_MIGRATE,
                reason="a clinic patient",
                uuid=row.uuid,
                label=row.label,
                birthdate=legacy[0],
                evidence={
                    "legacy_ordinal": legacy[1],
                    "reports": row.report_count,
                    "runs": row.run_count,
                    "files": row.file_count,
                    "has_portal_folder": has_folder,
                },
            )
        )
    return out


def _pick_survivor(rows: list[PatientRow]) -> PatientRow:
    """The row that keeps its UUID: most work attached, then oldest."""
    return sorted(
        rows,
        key=lambda r: (
            -(r.report_count + r.run_count + r.file_count),
            r.created_at,
        ),
    )[0]


# --------------------------------------------------------------------------- #
# Identity — evidence in, canonical initials out, or a precise question
# --------------------------------------------------------------------------- #


@dataclass
class IdentityFinding:
    patient_id: str
    resolved: tuple[str, str] | None
    sources: dict[str, Any]
    proposed: list[dict[str, str]] = field(default_factory=list)
    problem: str | None = None
    needs: str | None = None


def resolve_identity(
    patient_id: str,
    *,
    portal_root: Path,
    sync_state: dict[str, Any],
    conversation_titles: dict[str, list[str]],
) -> IdentityFinding:
    """Agree the initials across every source, or say precisely what is wrong."""
    sources: dict[str, Any] = {}

    meta = read_folder_identity(portal_root, patient_id)
    meta_pair = _pair(meta)
    if meta_pair:
        sources["folder_meta"] = "".join(meta_pair)

    sync_identity = None
    patients = sync_state.get("patients")
    if isinstance(patients, dict) and isinstance(patients.get(patient_id), dict):
        sync_identity = patients[patient_id].get("identity")
    sync_pair = _pair(sync_identity)
    if sync_pair:
        sources["sync_state"] = "".join(sync_pair)

    name_pairs = filename_initials(portal_root, patient_id)
    if name_pairs:
        sources["source_filenames"] = sorted("".join(p) for p in name_pairs)

    titles = [t for t in conversation_titles.get(patient_id, []) if t]
    if titles:
        sources["conversation_titles"] = titles

    stated = {p for p in (meta_pair, sync_pair) if p}
    if not stated:
        return IdentityFinding(
            patient_id=patient_id,
            resolved=None,
            sources=sources,
            problem="no initials on file",
            needs=(
                "the patient's first and last initial — nothing in the folder, the "
                "sync state, or the hub records says who this is"
            ),
        )

    if len(stated) > 1:
        return IdentityFinding(
            patient_id=patient_id,
            resolved=None,
            sources=sources,
            problem="the folder and the sync state disagree",
            needs="which pair of initials is right",
        )

    resolved = next(iter(stated))

    # A source filename carrying different initials is the strongest independent
    # evidence there is, and it has caught wrong entries before.
    if name_pairs and resolved not in name_pairs:
        return IdentityFinding(
            patient_id=patient_id,
            resolved=None,
            sources=sources,
            problem=(
                f"stored initials {''.join(resolved)} do not match the source report "
                f"filenames ({', '.join(sorted(''.join(p) for p in name_pairs))})"
            ),
            needs="which initials belong to this patient",
        )

    # The clinic uploads reports named after the person. When the conversation
    # that filed a report is titled with a name whose letters have nothing to do
    # with the stored initials, one of the two is wrong about who this is.
    for title in titles:
        if not title.lower().endswith(".pdf") or title.startswith(patient_id):
            continue
        # A filename that spells the initials out as its own token — `DK_20Tx…`
        # — agrees by construction; it is the same evidence, not a second source.
        if (resolved[0], resolved[1]) in _initials_tokens_in(title):
            continue
        letters = title_name_initials(title)
        if not letters:
            continue
        proposed = propose_initials_from_title(title)
        options = " or ".join(f"{p['initials']} ({p['reading']})" for p in proposed)
        if not ({resolved[0], resolved[1]} & letters):
            return IdentityFinding(
                patient_id=patient_id,
                resolved=None,
                sources=sources,
                proposed=proposed,
                problem=(
                    f"stored initials {''.join(resolved)} match nothing in the name on "
                    f"the uploaded report ({title!r}) — the stored pair looks like a "
                    f"placeholder"
                ),
                needs=(
                    f"confirm the correction: {options}"
                    if options
                    else "the patient's first and last initial"
                ),
            )
        if not ({resolved[0], resolved[1]} <= letters):
            return IdentityFinding(
                patient_id=patient_id,
                resolved=None,
                sources=sources,
                proposed=proposed,
                problem=(
                    f"stored initials {''.join(resolved)} only partly match the name on "
                    f"the uploaded report ({title!r})"
                ),
                needs=(
                    f"confirm which is right: keep {''.join(resolved)}, or {options}"
                    if options
                    else "the patient's first and last initial, confirmed"
                ),
            )

    return IdentityFinding(patient_id=patient_id, resolved=resolved, sources=sources)


def _pair(identity: Any) -> tuple[str, str] | None:
    if not isinstance(identity, dict):
        return None
    first = str(identity.get("firstInitial") or "").strip().upper()
    last = str(identity.get("lastInitial") or "").strip().upper()
    if len(first) == 1 and len(last) == 1 and first.isalpha() and last.isalpha():
        return first, last
    return None


def allocate_new_ids(
    entries: list[Classified],
) -> tuple[dict[str, str], list[str]]:
    """Give every migrating patient its canonical ID, in a stable order.

    Ordinals come from collisions on initials *and* birthdate, computed here in
    one pass so the manifest is deterministic. The live allocation still goes
    through the reservation table during ``--apply``; this is what the operator
    reviews first.
    """
    collisions: dict[tuple[str, str, str], list[Classified]] = defaultdict(list)
    for entry in entries:
        if entry.bucket != BUCKET_MIGRATE or not entry.initials or not entry.birthdate:
            continue
        collisions[(entry.initials[0], entry.initials[1], entry.birthdate)].append(entry)

    mapping: dict[str, str] = {}
    problems: list[str] = []
    for (first, last, birthdate), group in sorted(collisions.items()):
        # Oldest patient keeps the unsuffixed ID.
        for ordinal, entry in enumerate(
            sorted(group, key=lambda e: (e.evidence.get("legacy_ordinal", 0), e.key)),
            start=1,
        ):
            new_id = canonical_patient_id(first, last, birthdate, ordinal=ordinal)
            entry.new_id = new_id
            if entry.label:
                if mapping.get(entry.label) not in (None, new_id):
                    problems.append(
                        f"{entry.label} would map to both {mapping[entry.label]} "
                        f"and {new_id}"
                    )
                mapping[entry.label] = new_id
    return mapping, problems


# --------------------------------------------------------------------------- #
# Orphans and the remote manifest
# --------------------------------------------------------------------------- #


def find_orphan_pending_prefixes(
    pending_root: Path, job_markers: Iterable[str]
) -> list[str]:
    """Uploaded blobs whose job marker never got written.

    The hub writes the file first and the marker second. When the marker write
    fails the operator re-drops and gets a fresh upload id, so the first prefix
    is bytes nobody will ever claim.
    """
    if not pending_root.is_dir():
        return []
    live = {str(marker) for marker in job_markers}
    return sorted(
        entry.name
        for entry in pending_root.iterdir()
        if entry.is_dir() and entry.name not in live
    )


def report_evidence(db_path: Path, data_root: Path) -> dict[str, list[dict[str, Any]]]:
    """Every patient's reports with the digest of the file behind each one.

    Two rows holding the same bytes are the same report, whatever they are
    labelled — which is how a row filed under a mistyped ID is matched back to
    the patient it belongs to without anyone guessing.
    """
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    out: dict[str, list[dict[str, Any]]] = defaultdict(list)
    try:
        for row in conn.execute(
            "SELECT p.label, r.id, r.filename, r.stored_path "
            "FROM patients p JOIN reports r ON r.patient_id = p.id"
        ):
            stored = str(row["stored_path"])
            path = Path(stored) if stored.startswith("/") else data_root / stored
            digest = ""
            try:
                if path.is_file():
                    digest = patient_rekey._sha256_file(path)
            except OSError:
                digest = ""
            out[row["label"]].append(
                {
                    "report_id": row["id"],
                    "filename": row["filename"],
                    "stored_path": stored,
                    "sha256": digest,
                }
            )
    finally:
        conn.close()

    # A wrong data root reads as "no evidence anywhere", which would quietly
    # turn every answerable question back into one for the operator.
    hashed = sum(1 for reports in out.values() for r in reports if r["sha256"])
    total = sum(len(reports) for reports in out.values())
    if total and not hashed:
        raise MigrationStop(
            f"None of the {total} report files were readable under {data_root}. "
            f"Point --db and --portal-root at the same installation."
        )
    return out


def match_row_by_report_bytes(
    label: str, evidence: dict[str, list[dict[str, Any]]], *, candidates: set[str]
) -> dict[str, Any]:
    """Which other patients hold byte-identical copies of this row's reports.

    A row whose every report is already filed under exactly one other patient
    is that patient's, and the operator only has to confirm it. A row whose
    reports point at two different people is a filing accident that someone has
    to unpick — this says so rather than picking one.

    ``candidates`` is the set of settled patients worth matching against. Two
    unresolved rows holding each other's copies would otherwise each name the
    other and neither would be answered.
    """
    mine = evidence.get(label, [])
    per_report: list[dict[str, Any]] = []
    owners: set[str] = set()
    for report in mine:
        matches = sorted(
            other
            for other, reports in evidence.items()
            if other != label
            and other in candidates
            and report["sha256"]
            and any(r["sha256"] == report["sha256"] for r in reports)
        )
        owners.update(matches)
        per_report.append(
            {
                "filename": report["filename"],
                "sha256": report["sha256"][:16],
                "byte_identical_under": matches,
            }
        )

    conclusive = len(owners) == 1 and all(
        len(r["byte_identical_under"]) == 1 for r in per_report
    )
    return {
        "reports": per_report,
        "matched_patients": sorted(owners),
        "conclusive": conclusive,
    }


# Clinic report filenames put the surname first — "Stubner Helga", "L, Connor".
def propose_initials_from_title(title: str) -> list[dict[str, str]]:
    """Initials this report's filename suggests, for the operator to confirm."""
    stem = re.sub(r"\.(pdf|md|mp4|docx)$", "", title, flags=re.IGNORECASE)
    words = [
        word
        for word in re.findall(r"[A-Za-z]+", stem)
        if word.lower() not in _NON_NAME_WORDS
    ]
    if len(words) < 2:
        return (
            [{"initials": f"?{words[0][0].upper()}", "reading": f"surname {words[0]}"}]
            if words
            else []
        )
    surname, given = words[0], words[1]
    return [
        {
            "initials": f"{given[0].upper()}{surname[0].upper()}",
            "reading": f"{given} {surname} (surname first, as the clinic names files)",
        },
        {
            "initials": f"{surname[0].upper()}{given[0].upper()}",
            "reading": f"{surname} {given} (given name first)",
        },
    ]


def _live_upload_ids(jobs_dir: Path | None) -> list[str]:
    """Upload ids named by a pipeline job marker that still exists."""
    if jobs_dir is None or not jobs_dir.is_dir():
        return []
    live: list[str] = []
    for path in jobs_dir.rglob("*.json"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if isinstance(payload, dict) and isinstance(payload.get("uploadId"), str):
            live.append(payload["uploadId"])
    return live


# Where else a legacy ID appears in a name. Archives and backups are history
# and stay as they are; the first two are live and move with the patient.
_OTHER_LOCATIONS = (
    ("pipeline_jobs", "live"),
    ("exports", "live"),
    ("portal_patients_archives", "history"),
    ("portal_patients_precompress_backups", "history"),
    ("_manual_rescue_archive", "history"),
    ("_unsafe_partial_archive", "history"),
    ("video_quarantine", "history"),
)


def sweep_other_id_named_locations(
    data_root: Path, mapping: dict[str, str]
) -> dict[str, Any]:
    """Every directory besides the portal tree holding ID-named entries."""
    found: dict[str, Any] = {}
    for name, kind in _OTHER_LOCATIONS:
        directory = data_root / name
        if not directory.is_dir():
            continue
        hits = sorted(
            entry.name
            for entry in directory.iterdir()
            if any(entry.name.startswith(old) for old in mapping)
        )
        if hits:
            found[name] = {"kind": kind, "count": len(hits), "entries": hits[:40]}
    return found


def build_remote_manifest(
    sync_state: dict[str, Any], mapping: dict[str, str]
) -> list[dict[str, Any]]:
    work: list[dict[str, Any]] = []
    for old_id, new_id in sorted(mapping.items()):
        work.extend(patient_rekey.remote_rekey_worklist(sync_state, old_id, new_id))
    return work


# --------------------------------------------------------------------------- #
# The maintenance estimate
# --------------------------------------------------------------------------- #

# Measured against the clinic's link during the July sync incident work: the
# uploader sustained roughly this while copying portal deliverables.
ASSUMED_REMOTE_THROUGHPUT_MB_S = 2.5
# Local disk hashing on this machine, conservatively.
ASSUMED_HASH_THROUGHPUT_MB_S = 300.0
# Everything the numbers cannot predict: per-file request overhead, a retry or
# two, and someone actually looking at the result before deleting anything.
VERIFICATION_ALLOWANCE = 1.5


def estimate_window(
    *, local_bytes: int, local_files: int, remote_items: list[dict[str, Any]]
) -> dict[str, Any]:
    remote_bytes = sum(int(item.get("size") or 0) for item in remote_items)
    mb = 1024 * 1024

    hash_minutes = (local_bytes / mb) / ASSUMED_HASH_THROUGHPUT_MB_S / 60
    # Copy up, read back to verify: the bytes cross the link twice.
    remote_minutes = (remote_bytes / mb) * 2 / ASSUMED_REMOTE_THROUGHPUT_MB_S / 60
    # Renames are metadata operations; the cost is per file, not per byte.
    rename_minutes = local_files * 0.002 / 60

    subtotal = hash_minutes + remote_minutes + rename_minutes
    return {
        "local_bytes": local_bytes,
        "local_files": local_files,
        "remote_blobs": len(remote_items),
        "remote_bytes": remote_bytes,
        "local_rename_minutes": round(rename_minutes, 1),
        "hash_verify_minutes": round(hash_minutes, 1),
        "remote_copy_verify_minutes": round(remote_minutes, 1),
        "subtotal_minutes": round(subtotal, 1),
        "allowance_multiplier": VERIFICATION_ALLOWANCE,
        "estimated_minutes": round(subtotal * VERIFICATION_ALLOWANCE, 1),
        "assumptions": (
            f"remote throughput {ASSUMED_REMOTE_THROUGHPUT_MB_S} MB/s each way; "
            f"local hashing {ASSUMED_HASH_THROUGHPUT_MB_S} MB/s; "
            f"local renames are metadata only; "
            f"x{VERIFICATION_ALLOWANCE} allowance for per-file overhead, retries, "
            f"and operator verification before any delete"
        ),
    }


# --------------------------------------------------------------------------- #
# The journal
# --------------------------------------------------------------------------- #


class Journal:
    """One line per finished patient, flushed before the next one starts.

    A crash costs at most the patient in flight. A resumed run reads this,
    skips what is done, and so cannot rename a file twice or push a second copy
    of a deliverable under a fresh version number.
    """

    def __init__(self, path: Path):
        self.path = path
        self.done: dict[str, dict[str, Any]] = {}
        if path.is_file():
            for line in path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except ValueError:
                    continue
                if isinstance(record, dict) and record.get("old_id"):
                    self.done[record["old_id"]] = record

    def is_done(self, old_id: str) -> bool:
        return old_id in self.done

    def record(self, entry: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        self.done[entry["old_id"]] = entry


# --------------------------------------------------------------------------- #
# Renderer boundary inventory
# --------------------------------------------------------------------------- #

# A date-of-birth shaped regex, however many named groups are threaded through
# it: two two-digit fields and a four-digit year, in that order, on one line.
_DOB_REGEX_SHAPE = re.compile(
    r"\\d\{2\}.{0,24}?-.{0,24}?\\d\{2\}.{0,24}?-.{0,24}?\\d\{4\}"
)

# Two initials and an underscore in front of the birthdate: the canonical ID.
_CANONICAL_SHAPE = re.compile(r"\[A-Z\]\{2\}\\?_")

RENDERER_REJECT_PATTERNS = (
    (
        re.compile(r"strptime\([^)]*patient", re.IGNORECASE),
        "parses the patient ID as a bare date",
    ),
    (_DOB_REGEX_SHAPE, "matches a date-of-birth-only ID shape"),
    (re.compile(r"%m-%d-%Y"), "formats or parses an ID as a bare date"),
)

_SCAN_SKIP_DIRS = frozenset(
    {".git", "node_modules", ".venv", "venv", "__pycache__", "worktrees", "site-packages"}
)


def scan_renderer_repo(root: Path) -> list[dict[str, Any]]:
    """Find validators and parsers that would reject a canonical ID.

    A renderer that cannot read the clinic's ID off a project folder silently
    skips every real patient, so an unresolved finding here fails the dry run.
    """
    findings: list[dict[str, Any]] = []
    if not root.is_dir():
        return [{"path": str(root), "issue": "repository not found", "line": 0}]
    for path in sorted(root.rglob("*.py")):
        if _SCAN_SKIP_DIRS & set(path.parts):
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        if "patient" not in text.lower():
            continue
        for number, line in enumerate(text.splitlines(), start=1):
            if "patient" not in line.lower() and "_id_re" not in line.lower():
                continue
            # The canonical contract contains a birthdate too. What makes it
            # canonical is the two initials in front of it.
            if _CANONICAL_SHAPE.search(line):
                continue
            for pattern, issue in RENDERER_REJECT_PATTERNS:
                if pattern.search(line):
                    findings.append(
                        {
                            "path": str(path),
                            "line": number,
                            "issue": issue,
                            "source": line.strip()[:160],
                        }
                    )
                    break
    return findings


# --------------------------------------------------------------------------- #
# The run
# --------------------------------------------------------------------------- #


def _under_production(path: Path) -> bool:
    resolved = path.resolve()
    return any(
        resolved == root or root in resolved.parents for root in PRODUCTION_ROOTS
    )


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    portal_root = Path(args.portal_root)
    rows = read_patient_rows(Path(args.db))
    classified = classify_patient_rows(
        rows,
        portal_root=portal_root,
        qa_candidates_confirmed=bool(getattr(args, "qa_candidates_confirmed", False)),
    )
    sync_state = read_sync_state(portal_root / ".qeeg_portal_sync_state.json")
    titles = read_conversation_titles(Path(args.conversations_dir))

    findings: list[IdentityFinding] = []
    for entry in classified:
        if entry.bucket != BUCKET_MIGRATE or not entry.label:
            continue
        finding = resolve_identity(
            entry.label,
            portal_root=portal_root,
            sync_state=sync_state,
            conversation_titles=titles,
        )
        findings.append(finding)
        if finding.resolved:
            entry.initials = finding.resolved
        else:
            entry.bucket = BUCKET_UNRESOLVED
            entry.reason = finding.problem or "identity cannot be trusted"
            entry.needs = finding.needs
            entry.evidence["identity_sources"] = finding.sources

    # `reports.stored_path` is written relative to the repository root
    # (`data/reports/…`), not to the data directory.
    evidence = report_evidence(Path(args.db), portal_root.parent.parent)
    settled_patients = {
        e.label for e in classified if e.label and e.bucket == BUCKET_MIGRATE
    }
    qa_and_pollution = {
        e.label
        for e in classified
        if e.label and e.bucket in (BUCKET_QA_FIXTURE, BUCKET_TEST_POLLUTION)
    }

    # A folder is only trustworthy if everything in it belongs to one person.
    owned_initials = {
        "".join(e.initials): e.label
        for e in classified
        if e.bucket == BUCKET_MIGRATE and e.initials and e.label
    }
    mixed: list[dict[str, Any]] = []
    for entry in classified:
        if entry.bucket not in (BUCKET_MIGRATE, BUCKET_UNRESOLVED) or not entry.label:
            continue
        mine = "".join(entry.initials) if entry.initials else None
        foreign = {
            token: names
            for token, names in embedded_initials(portal_root, entry.label).items()
            if token != mine and token in owned_initials
        }
        if not foreign:
            continue
        mixed.append(
            {
                "patient_id": entry.label,
                "foreign_initials": {
                    token: {
                        "belongs_to": owned_initials[token],
                        "file_count": len(names),
                        "examples": names[:2],
                    }
                    for token, names in sorted(foreign.items())
                },
            }
        )
        entry.bucket = BUCKET_UNRESOLVED
        entry.reason = (
            "the folder holds reports belonging to "
            + ", ".join(
                f"{owned_initials[t]} ({t}, {len(n)} files)"
                for t, n in sorted(foreign.items())
            )
        )
        entry.needs = (
            "a decision: move each misfiled report to the patient it belongs to "
            "during the window, or freeze this folder as it stands"
        )

    # An unpadded legacy label sitting next to its padded twin is a filing
    # accident, not a second person — but which way it resolves is not ours.
    padded_labels = {e.label for e in classified if e.label}
    for entry in classified:
        if entry.bucket not in (BUCKET_MIGRATE, BUCKET_UNRESOLVED) or not entry.label:
            continue
        legacy = parse_legacy_id(entry.label)
        if legacy is None or entry.label == f"{legacy[0]}-{legacy[1]}":
            continue
        twin = f"{legacy[0]}-{legacy[1]}"
        if twin in padded_labels and twin != entry.label:
            entry.bucket = BUCKET_UNRESOLVED
            # Do not stop at "this looks odd" when the bytes can answer it.
            match = match_row_by_report_bytes(
                entry.label, evidence, candidates=settled_patients
            )
            entry.evidence["report_byte_match"] = match
            if match["conclusive"]:
                entry.reason = (
                    f"an unpadded label whose every report is byte-identical to one "
                    f"already filed under {match['matched_patients'][0]}"
                )
                entry.needs = (
                    f"confirmation only: fold this row's work into "
                    f"{match['matched_patients'][0]} and retire the label"
                )
            elif match["matched_patients"]:
                entry.reason = (
                    "an unpadded label holding reports that belong to more than one "
                    "patient: "
                    + "; ".join(
                        f"{r['filename']} is already under "
                        f"{', '.join(r['byte_identical_under']) or 'nobody else'}"
                        for r in match["reports"]
                    )
                )
                entry.needs = (
                    "a decision per report — these are copies of files already filed "
                    "elsewhere, so the row itself may simply be retired"
                )
            else:
                entry.reason = (
                    f"an unpadded label sitting next to {twin}, which is a real "
                    f"patient with the same date of birth"
                )
                entry.needs = (
                    f"whether this row's work belongs to {twin} or to a different "
                    f"person"
                )

    mapping, collisions = allocate_new_ids(classified)

    # Portal folders with no database row at all still have to be accounted for.
    known = {e.label for e in classified if e.label}
    orphan_folders = sorted(
        entry.name
        for entry in portal_root.iterdir()
        if entry.is_dir() and entry.name not in known
    ) if portal_root.is_dir() else []

    local_files = 0
    local_bytes = 0
    for old_id in mapping:
        folder = portal_root / old_id
        if not folder.is_dir():
            continue
        for path in folder.rglob("*"):
            if path.is_file():
                local_files += 1
                local_bytes += path.stat().st_size

    remote_items = build_remote_manifest(sync_state, mapping)

    renderer_findings = {
        name: scan_renderer_repo(Path(root))
        for name, root in (
            ("cathode", args.cathode_root),
            ("local-explainer-video", args.explainer_root),
        )
        if root
    }

    # The pipeline worker's per-patient status files are live routing state and
    # move with the patient.
    jobs_dir = Path(args.pipeline_jobs_dir) if args.pipeline_jobs_dir else None
    pipeline_jobs = (
        {
            old_id: f"{new_id}.json"
            for old_id, new_id in sorted(mapping.items())
            if (jobs_dir / f"{old_id}.json").is_file()
        }
        if jobs_dir and jobs_dir.is_dir()
        else {}
    )

    # Uploaded blobs whose job marker never landed. The hub writes the file
    # first, so a failed marker write leaves bytes nobody will ever claim.
    pending_dir = Path(args.pending_uploads_dir) if args.pending_uploads_dir else None
    orphan_pending = (
        find_orphan_pending_prefixes(pending_dir, _live_upload_ids(jobs_dir))
        if pending_dir
        else []
    )

    # Anywhere else on disk still carrying a legacy ID in its name, so the
    # manifest can say "these are the only places" and mean it.
    other_locations = sweep_other_id_named_locations(portal_root.parent, mapping)

    qa_bundle = qa_fixture_bundle(Path(args.db), portal_root)

    # The renderers keep a project directory per patient, named for the patient.
    # They are on-disk paths like any other, so they belong in the inventory.
    renderer_projects: dict[str, Any] = {}
    for name, root in (
        ("cathode", args.cathode_root),
        ("local-explainer-video", args.explainer_root),
    ):
        if not root:
            continue
        for subdir in ("projects", "_bettube_studio_migration"):
            directory = Path(root) / subdir
            if not directory.is_dir():
                continue
            legacy = sorted(
                entry.name
                for entry in directory.iterdir()
                if entry.is_dir()
                and any(entry.name.startswith(old) for old in mapping)
            )
            if legacy:
                renderer_projects[f"{name}/{subdir}"] = {
                    "count": len(legacy),
                    "entries": legacy[:40],
                }

    unresolved = [e for e in classified if e.bucket == BUCKET_UNRESOLVED]
    buckets: dict[str, int] = defaultdict(int)
    for entry in classified:
        buckets[entry.bucket] += 1

    blockers: list[str] = []
    for entry in unresolved:
        if entry.label in QA_CANDIDATE_LABELS:
            entry.reason = (
                "possibly the owner's own test data rather than a clinic patient "
                "(he has described this date as his test file before) — "
                + entry.reason
            )
            entry.needs = (
                "a yes or no: is this your test data? Yes removes it from the "
                "roster with --qa-candidates-confirmed; no needs the patient's "
                "initials"
            )
        blockers.append(f"{entry.label}: {entry.reason} — needs {entry.needs}")
    blockers.extend(collisions)
    if orphan_folders:
        blockers.append(
            "portal folders with no database row: " + ", ".join(orphan_folders)
        )
    for name, items in renderer_findings.items():
        for item in items:
            blockers.append(
                f"{name}: {item['path']}:{item['line']} {item['issue']}"
            )

    return {
        "buckets": dict(buckets),
        "classified": [asdict(e) for e in classified],
        "mapping": mapping,
        "identity_findings": [asdict(f) for f in findings],
        "orphan_portal_folders": orphan_folders,
        "orphan_pending_prefixes": orphan_pending,
        "mixed_patient_folders": mixed,
        "pipeline_job_files": pipeline_jobs,
        "other_id_named_locations": other_locations,
        "renderer_project_dirs": renderer_projects,
        "qa_fixture_bundle": qa_bundle,
        "notes": {
            "ordinal_rule": (
                "Legacy `-N` counts collisions on the birthdate alone and starts "
                "at 0. Canonical `_N` counts collisions on initials AND birthdate, "
                "starts at 1, and omits the suffix for 1 — so `_1` never exists. "
                "The two never correspond and the legacy ordinal is never carried "
                "across: every canonical ID here is allocated fresh from the "
                "initials-and-birthdate collisions, and --apply reserves each one "
                "through backend.patient_identity.reserve_canonical_patient_id."
            ),
            "renderer_ground_truth_dependency": (
                "local-explainer-video's qc_publish loads qEEG ground truth by "
                "querying patients.label in the engine database. Those labels "
                "become canonical in the same step as everything else, so the "
                "renderers must not be run against the engine between the label "
                "migration and their own deploy."
            ),
            "schema_prerequisite": (
                "--apply refuses a database without patient_id_reservations and "
                "the patients identity columns. Start the new engine against the "
                "database once before migrating."
            ),
        },
        "remote_manifest": remote_items,
        "renderer_findings": renderer_findings,
        "estimate": estimate_window(
            local_bytes=local_bytes,
            local_files=local_files,
            remote_items=remote_items,
        ),
        "blockers": blockers,
    }


def print_report(report: dict[str, Any]) -> None:
    print("=" * 74)
    print("CANONICAL PATIENT ID CUTOVER — DRY RUN")
    print("=" * 74)

    print("\nEvery database row, in exactly one bucket:")
    for bucket, count in sorted(report["buckets"].items()):
        print(f"  {count:4}  {bucket}")

    print("\nPatients that would be renamed:")
    for old, new in sorted(report["mapping"].items()):
        print(f"  {old:24} -> {new}")

    dupes = [
        e for e in report["classified"] if e["bucket"] == BUCKET_DUPLICATE
    ]
    if dupes:
        print("\nDuplicate rows whose work moves to the surviving record first:")
        for entry in dupes:
            print(
                f"  {entry['label']} {entry['uuid'][:8]} -> "
                f"{entry['survivor_uuid'][:8]}  "
                f"(reports {entry['evidence'].get('reports')}, "
                f"runs {entry['evidence'].get('runs')}, "
                f"files {entry['evidence'].get('files')})"
            )

    pollution = [
        e for e in report["classified"] if e["bucket"] == BUCKET_TEST_POLLUTION
    ]
    if pollution:
        print("\nRows written by a test run against the live database (not migrated):")
        for entry in pollution:
            print(f"  {entry['label']} {entry['uuid'][:8]}")

    estimate = report["estimate"]
    print("\nMaintenance window:")
    print(
        f"  local: {estimate['local_files']} files, "
        f"{estimate['local_bytes'] / (1024**3):.1f} GB "
        f"— renames are metadata only, {estimate['local_rename_minutes']} min"
    )
    print(f"  hash + verify local content: {estimate['hash_verify_minutes']} min")
    print(
        f"  remote: {estimate['remote_blobs']} blobs, "
        f"{estimate['remote_bytes'] / (1024**3):.1f} GB copy + verify "
        f"= {estimate['remote_copy_verify_minutes']} min"
    )
    print(
        f"  ESTIMATE {estimate['estimated_minutes']} min "
        f"({estimate['estimated_minutes'] / 60:.1f} h) "
        f"including a x{estimate['allowance_multiplier']} allowance"
    )
    print(f"  assumptions: {estimate['assumptions']}")

    if report["blockers"]:
        print("\n" + "!" * 74)
        print(f"{len(report['blockers'])} THING(S) TO RESOLVE BEFORE THE WINDOW")
        print("!" * 74)
        for blocker in report["blockers"]:
            print(f"  - {blocker}")
    else:
        print("\nNothing unresolved. Every patient has trustworthy identity.")


def reconcile_duplicate(
    conn: sqlite3.Connection, *, loser_uuid: str, survivor_uuid: str
) -> dict[str, int]:
    """Move one duplicate row's work onto the surviving patient record.

    Reports, runs, and patient files all point at the patient UUID, so the
    relabel that follows only has to touch one row. Done before any rename so a
    half-finished reconciliation never leaves work stranded under a label that
    no longer exists.
    """
    moved: dict[str, int] = {}
    for table in ("reports", "runs", "patient_files"):
        cursor = conn.execute(
            f"UPDATE {table} SET patient_id = ? WHERE patient_id = ?",
            (survivor_uuid, loser_uuid),
        )
        moved[table] = cursor.rowcount
    conn.execute("DELETE FROM patients WHERE id = ?", (loser_uuid,))
    return moved


REQUIRED_IDENTITY_COLUMNS = ("birthdate", "first_initial", "last_initial")


def assert_schema_ready(db_path: Path) -> None:
    """Refuse to migrate into a database the new engine has not upgraded yet.

    ``patient_id_reservations`` and the identity columns arrive with the new
    engine code. Writing canonical labels into the old five-column shape would
    leave every migrated patient with no reservation and no identity, which is
    exactly the state the cutover exists to end.
    """
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        tables = {
            row[0] for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }
        if "patient_id_reservations" not in tables:
            raise MigrationStop(
                "This database has no patient_id_reservations table. Start the "
                "new engine against it once so the schema upgrades, then migrate."
            )
        columns = {
            row[1] for row in conn.execute("PRAGMA table_info(patients)")
        }
        missing = [c for c in REQUIRED_IDENTITY_COLUMNS if c not in columns]
        if missing:
            raise MigrationStop(
                f"The patients table is missing {', '.join(missing)}. Start the "
                f"new engine against this database once, then migrate."
            )
    finally:
        conn.close()


def qa_fixture_bundle(db_path: Path, portal_root: Path) -> dict[str, Any]:
    """Everything the QA record owns, listed so the rollback bundle can hold it.

    The synthetic record is removed from the roster during apply rather than
    migrated, so this is the only place its rows and artifacts are written down.
    """
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    bundle: dict[str, Any] = {"patients": [], "reports": [], "runs": [],
                              "patient_files": [], "portal_artifacts": []}
    try:
        for label in sorted(QA_FIXTURE_LABELS):
            for row in conn.execute(
                "SELECT id, label, created_at FROM patients WHERE label = ?", (label,)
            ):
                bundle["patients"].append(dict(row))
                for table in ("reports", "runs", "patient_files"):
                    bundle[table].extend(
                        dict(r)
                        for r in conn.execute(
                            f"SELECT * FROM {table} WHERE patient_id = ?", (row["id"],)
                        )
                    )
            folder = portal_root / label
            if folder.is_dir():
                bundle["portal_artifacts"].extend(
                    {
                        "path": str(path),
                        "size": path.stat().st_size,
                        "sha256": patient_rekey._sha256_file(path),
                    }
                    for path in sorted(folder.rglob("*"))
                    if path.is_file()
                )
    finally:
        conn.close()
    return bundle


def remove_qa_fixture(conn: sqlite3.Connection, portal_root: Path) -> dict[str, int]:
    """Take the synthetic record off the active roster. Bundle it first."""
    import shutil

    removed = {"patients": 0, "rows": 0, "folders": 0}
    for label in sorted(QA_FIXTURE_LABELS):
        uuids = [
            row[0]
            for row in conn.execute("SELECT id FROM patients WHERE label = ?", (label,))
        ]
        for uuid in uuids:
            for table in ("reports", "runs", "patient_files"):
                removed["rows"] += conn.execute(
                    f"DELETE FROM {table} WHERE patient_id = ?", (uuid,)
                ).rowcount
            removed["patients"] += conn.execute(
                "DELETE FROM patients WHERE id = ?", (uuid,)
            ).rowcount
        folder = portal_root / label
        if folder.is_dir():
            shutil.rmtree(folder)
            removed["folders"] += 1
    conn.commit()
    return removed


def reserve_migrated_id(
    db_path: Path,
    new_id: str,
    *,
    uuid: str | None,
    birthdate: str,
    first_initial: str,
    last_initial: str,
) -> None:
    """Relabel one patient through the identity module, not around it.

    Goes through ``backend.storage``'s session so the reservation is written by
    ``reserve_canonical_patient_id`` itself — the same code path the running
    engine uses — and so the patient's identity columns are filled in from the
    evidence the dry run already resolved. A migrated patient with NULL identity
    would send the very next intake through name matching with nothing to match.
    """
    from backend import storage

    storage.reset_engine(f"sqlite:///{db_path}")
    with storage.session_scope() as session:
        reserve_canonical_patient_id(session, new_id)
        patient = session.get(storage.Patient, uuid) if uuid else None
        if patient is None:
            return
        patient.label = new_id
        if birthdate:
            patient.birthdate = birthdate
        if first_initial:
            patient.first_initial = first_initial
        if last_initial:
            patient.last_initial = last_initial
        session.commit()


def run_apply(args: argparse.Namespace, report: dict[str, Any]) -> int:
    """Carry out the migration, one patient at a time, journalling as it goes."""
    if report["blockers"]:
        print(
            f"\nRefusing to apply: {len(report['blockers'])} thing(s) are unresolved.",
            file=sys.stderr,
        )
        return 1

    portal_root = Path(args.portal_root)
    assert_schema_ready(Path(args.db))
    journal = Journal(Path(args.journal or "migration-journal.jsonl"))
    sync_paths = [
        portal_root / name
        for name in (
            ".qeeg_portal_sync_state.json",
            ".qeeg_portal_local_pipeline_state.json",
            ".qeeg_portal_sync_watch_state.json",
        )
    ]

    # Only the surviving row carries the label forward. Keying on every row
    # would point the relabel at a duplicate that reconciliation just deleted.
    by_label = {
        entry["label"]: entry
        for entry in report["classified"]
        if entry.get("label") and entry["bucket"] == BUCKET_MIGRATE
    }
    conn = sqlite3.connect(args.db)
    failures: list[str] = []
    try:
        for entry in report["classified"]:
            if entry["bucket"] != BUCKET_DUPLICATE:
                continue
            if journal.is_done(f"dup:{entry['uuid']}"):
                continue
            moved = reconcile_duplicate(
                conn, loser_uuid=entry["uuid"], survivor_uuid=entry["survivor_uuid"]
            )
            conn.commit()
            journal.record(
                {
                    "old_id": f"dup:{entry['uuid']}",
                    "kind": "duplicate_reconciled",
                    "survivor": entry["survivor_uuid"],
                    "moved": moved,
                }
            )
            print(f"  reconciled duplicate {entry['uuid'][:8]} -> "
                  f"{entry['survivor_uuid'][:8]} {moved}")

        for old_id, new_id in sorted(report["mapping"].items()):
            if journal.is_done(old_id):
                print(f"  {old_id}: already done, skipping")
                continue
            # One patient's failure must not abandon the rest of the work.
            try:
                plan = patient_rekey.plan_patient_rekey(
                    old_id,
                    new_id,
                    portal_root=portal_root,
                    conversations_dir=Path(args.conversations_dir),
                    sync_state_paths=sync_paths,
                    pipeline_jobs_dir=Path(args.pipeline_jobs_dir)
                    if args.pipeline_jobs_dir
                    else None,
                )
                entry = by_label.get(old_id) or {}
                uuid = entry.get("uuid")
                initials = entry.get("initials") or ("", "")
                # The reservation is the whole point of the identity module: an
                # ID that is only worn by a live row becomes reissuable the
                # moment that row is deleted or relabelled. Write it first, so a
                # crash after this leaves the ID retired rather than free.
                reserve_migrated_id(
                    Path(args.db),
                    new_id,
                    uuid=uuid,
                    birthdate=entry.get("birthdate") or "",
                    first_initial=initials[0],
                    last_initial=initials[1],
                )
                conn.commit()
                result = patient_rekey.apply_patient_rekey(plan)
                journal.record(
                    {
                        "old_id": old_id,
                        "new_id": new_id,
                        "kind": "patient_rekeyed",
                        "files_renamed": result.files_renamed,
                        "folder_moved": result.folder_moved,
                        "conversations": result.conversations_repointed,
                        "sync_entries": result.sync_entries_repointed,
                        "message_hashes": result.message_hashes,
                    }
                )
                print(
                    f"  {old_id} -> {new_id}: {result.files_renamed} files, "
                    f"{result.conversations_repointed} conversations"
                )
            except Exception as error:  # noqa: BLE001 - collected and reported
                failures.append(f"{old_id}: {error}")
                print(f"  {old_id}: FAILED — {error}", file=sys.stderr)

        # The synthetic QA record leaves the roster rather than migrating. Its
        # rows and artifacts are listed in the manifest for the rollback bundle
        # first, so this removal is recoverable.
        if patient_rekey.rewrite_portal_readme(portal_root):
            print("  rewrote the share folder README onto the clinic ID format")

        if not journal.is_done("qa-fixture-removed"):
            removed = remove_qa_fixture(conn, portal_root)
            journal.record(
                {"old_id": "qa-fixture-removed", "kind": "qa_fixture_removed",
                 "removed": removed}
            )
            print(f"  removed the QA fixture from the roster: {removed}")
    finally:
        conn.close()

    worklist = Path(args.journal or ".").parent / "remote-rekey-worklist.json"
    worklist.write_text(
        json.dumps(report["remote_manifest"], indent=2), encoding="utf-8"
    )
    print(f"\nRemote rekey worklist written to {worklist}")

    if failures:
        print(f"\n{len(failures)} patient(s) failed:", file=sys.stderr)
        for failure in failures:
            print(f"  - {failure}", file=sys.stderr)
        return 1
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--apply", action="store_true")
    parser.add_argument("--db", default="data/app.db")
    parser.add_argument("--portal-root", default="data/portal_patients")
    parser.add_argument(
        "--conversations-dir",
        default="/Users/davidmontgomery/qeeg-clinic-workbench/clinic/server/data/conversations",
    )
    parser.add_argument("--cathode-root", default="")
    parser.add_argument("--explainer-root", default="")
    parser.add_argument("--manifest-out", default="")
    parser.add_argument("--journal", default="")
    parser.add_argument("--pipeline-jobs-dir", default="")
    parser.add_argument("--pending-uploads-dir", default="")
    parser.add_argument(
        "--qa-candidates-confirmed",
        action="store_true",
        help=(
            "The owner has confirmed the candidate QA rows are his own test data. "
            "Without this they stay unresolved and the dry run keeps failing."
        ),
    )
    parser.add_argument(
        "--window-confirmed",
        action="store_true",
        help="Required by --apply. Confirms every writer is stopped.",
    )
    parser.add_argument(
        "--this-is-the-scheduled-cutover",
        action="store_true",
        help=(
            "Also required to --apply against the live clinic data. Only the "
            "scheduled maintenance window passes this."
        ),
    )
    args = parser.parse_args(argv)

    if args.apply:
        if not args.window_confirmed:
            print(
                "--apply needs --window-confirmed: stop the workbench, the engine, "
                "the portal workers, and the hub watchers first.",
                file=sys.stderr,
            )
            return 2
        for label, path in (("database", args.db), ("portal root", args.portal_root)):
            if _under_production(Path(path)) and not args.this_is_the_scheduled_cutover:
                print(
                    f"Refusing to --apply against the live {label} at {path}. "
                    f"Development invocations run against fixtures. The scheduled "
                    f"cutover passes --this-is-the-scheduled-cutover as well.",
                    file=sys.stderr,
                )
                return 2

    report = build_report(args)
    print_report(report)

    if args.manifest_out:
        out = Path(args.manifest_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
        print(f"\nManifest written to {out}")

    if args.apply:
        print("\nApplying:")
        return run_apply(args, report)

    return 1 if report["blockers"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
