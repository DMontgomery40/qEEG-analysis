#!/usr/bin/env python3
"""Move the whole clinic onto canonical patient IDs, once.

Legacy IDs are date-of-birth keys — ``09-23-1982-0`` — where the trailing number
counts patients sharing a birthdate, starting at zero. The canonical ID is
``XX_MM-DD-YYYY[_N]``: two initials, the birthdate, and a collision ordinal that
starts at one and is left off when it is one. The two ordinals do not
correspond: the legacy one counts collisions on the birthdate alone, the
canonical one counts collisions on initials *and* birthdate. Two people born the
same day with different initials both end up unsuffixed. So this migrator never
carries a legacy ordinal across — it asks ``backend.patient_identity`` to
allocate, and the reservation table remembers the answer forever.

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
)

# The legacy key: birthdate plus a zero-based collision counter.
LEGACY_ID_RE = re.compile(r"^(?P<mm>\d{1,2})-(?P<dd>\d{1,2})-(?P<yyyy>\d{4})(?:-(?P<n>\d+))?$")

# Rows whose stored files live in a throwaway directory were written by a test
# run, not by the clinic. The engine's DATA_DIR is relative, so a suite run with
# the wrong working directory lands in the live database.
TEST_PATH_MARKERS = ("pytest-of-", "/tmp/qeeg-hang-repro")

# The synthetic record David built to exercise the identity path end to end.
QA_FIXTURE_LABELS = frozenset({"02-29-1984-0"})

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
    rows: Iterable[PatientRow], *, portal_root: Path
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
        if not ({resolved[0], resolved[1]} & letters):
            return IdentityFinding(
                patient_id=patient_id,
                resolved=None,
                sources=sources,
                problem=(
                    f"stored initials {''.join(resolved)} match nothing in the name on "
                    f"the uploaded report ({title!r})"
                ),
                needs=(
                    "confirmation of which initials are this patient's — the stored "
                    "pair looks like a placeholder"
                ),
            )
        if not ({resolved[0], resolved[1]} <= letters):
            return IdentityFinding(
                patient_id=patient_id,
                resolved=None,
                sources=sources,
                problem=(
                    f"stored initials {''.join(resolved)} only partly match the name on "
                    f"the uploaded report ({title!r})"
                ),
                needs="the patient's first and last initial, confirmed",
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

RENDERER_REJECT_PATTERNS = (
    (r"strptime\(\s*[A-Za-z_]*patient_id", "parses the patient ID as a bare date"),
    (r'r?["\']\^?\\d\{2\}-\\d\{2\}-\\d\{4\}', "matches a DOB-only ID shape"),
    (r"%m-%d-%Y", "formats or parses an ID as a bare date"),
)


def scan_renderer_repo(root: Path) -> list[dict[str, Any]]:
    """Find validators and parsers that would reject a canonical ID."""
    findings: list[dict[str, Any]] = []
    if not root.is_dir():
        return [{"path": str(root), "issue": "repository not found", "line": 0}]
    for path in sorted(root.rglob("*.py")):
        parts = set(path.parts)
        if parts & {".git", "node_modules", ".venv", "venv", "__pycache__"}:
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        if "patient" not in text.lower():
            continue
        for number, line in enumerate(text.splitlines(), start=1):
            for pattern, issue in RENDERER_REJECT_PATTERNS:
                if re.search(pattern, line):
                    findings.append(
                        {
                            "path": str(path),
                            "line": number,
                            "issue": issue,
                            "source": line.strip()[:160],
                        }
                    )
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
    classified = classify_patient_rows(rows, portal_root=portal_root)
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
            entry.reason = (
                f"an unpadded label sitting next to {twin}, which is a real patient "
                f"with the same date of birth"
            )
            entry.needs = (
                f"whether this row's work belongs to {twin} or to a different person"
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

    unresolved = [e for e in classified if e.bucket == BUCKET_UNRESOLVED]
    buckets: dict[str, int] = defaultdict(int)
    for entry in classified:
        buckets[entry.bucket] += 1

    blockers: list[str] = []
    for entry in unresolved:
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
        "mixed_patient_folders": mixed,
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


def run_apply(args: argparse.Namespace, report: dict[str, Any]) -> int:
    """Carry out the migration, one patient at a time, journalling as it goes."""
    if report["blockers"]:
        print(
            f"\nRefusing to apply: {len(report['blockers'])} thing(s) are unresolved.",
            file=sys.stderr,
        )
        return 1

    portal_root = Path(args.portal_root)
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
                )
                uuid = (by_label.get(old_id) or {}).get("uuid")
                conn.execute(
                    "UPDATE patients SET label = ? WHERE id = ?", (new_id, uuid)
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
    parser.add_argument(
        "--window-confirmed",
        action="store_true",
        help="Required by --apply. Confirms every writer is stopped.",
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
            if _under_production(Path(path)):
                print(
                    f"Refusing to --apply against the live {label} at {path}. "
                    f"The real cutover runs from the scheduled window runbook, "
                    f"not from a development invocation.",
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
