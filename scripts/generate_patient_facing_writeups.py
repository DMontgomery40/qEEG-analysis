#!/usr/bin/env python
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from backend.config import CLIPROXY_API_KEY, CLIPROXY_BASE_URL, ensure_data_dirs
from backend.llm_client import AsyncOpenAICompatClient, UpstreamError
from backend.patient_facing_pdf import render_patient_facing_markdown_to_pdf
from backend.storage import Artifact, Patient, Run, find_patients_by_label, init_db, list_artifacts, list_runs, session_scope


@dataclass(frozen=True)
class SourceBundle:
    patient: Patient
    run: Run
    artifacts: list[Artifact]


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _discover_portal_patient_labels(portal_dir: Path) -> list[str]:
    if not portal_dir.exists():
        return []
    labels: list[str] = []
    for p in sorted(portal_dir.iterdir()):
        if not p.is_dir():
            continue
        if p.name.startswith("_"):
            continue
        labels.append(p.name)
    return labels


def _best_source_bundle_for_label(label: str) -> SourceBundle | None:
    """
    Resolve a portal patient label -> (patient, latest complete run, 2–3 'final' artifacts).

    IMPORTANT: We intentionally do NOT read raw qEEG PDFs/data packs. Inputs are council markdown artifacts
    (prefer Stage 6 final drafts).
    """
    with session_scope() as session:
        patients = find_patients_by_label(session, label)
        if not patients:
            return None

        # Multiple DB patient rows may share the same portal label. Prefer the newest COMPLETE run across them.
        complete_runs: list[tuple[Patient, Run]] = []
        for p in patients:
            for r in list_runs(session, p.id):
                if (r.status or "") == "complete":
                    complete_runs.append((p, r))
        if not complete_runs:
            return None
        patient, run = max(
            complete_runs,
            key=lambda pr: pr[1].created_at or datetime.min.replace(tzinfo=timezone.utc),
        )
        arts = list_artifacts(session, run.id)

        # Prefer 2–3 Stage 6 final drafts.
        stage6 = [a for a in arts if a.stage_num == 6 and a.kind == "final_draft"]
        stage6_sorted = sorted(stage6, key=lambda a: a.created_at, reverse=True)

        chosen: list[Artifact] = []
        if run.selected_artifact_id:
            sel = next((a for a in stage6_sorted if a.id == run.selected_artifact_id), None)
            if sel is not None:
                chosen.append(sel)
                stage6_sorted = [a for a in stage6_sorted if a.id != sel.id]

        for a in stage6_sorted:
            if len(chosen) >= 3:
                break
            chosen.append(a)

        # If we still don't have at least 2 distinct "perspectives", fall back to consolidation then revision.
        if len(chosen) < 2:
            stage4 = [a for a in arts if a.stage_num == 4 and a.kind == "consolidation"]
            if stage4:
                chosen.append(sorted(stage4, key=lambda a: a.created_at, reverse=True)[0])
        if len(chosen) < 2:
            stage3 = [a for a in arts if a.stage_num == 3 and a.kind == "revision"]
            for a in sorted(stage3, key=lambda a: a.created_at, reverse=True):
                if len(chosen) >= 2:
                    break
                chosen.append(a)

        if not chosen:
            return None

        return SourceBundle(patient=patient, run=run, artifacts=chosen)


def _read_text(path: str) -> str:
    p = Path(path)
    return p.read_text(encoding="utf-8", errors="replace")


def _example_writeup_text() -> str:
    return Path("examples/final-patient-facing-writeup-example.md").read_text(encoding="utf-8", errors="replace")


def _count_words(text: str) -> int:
    return len([w for w in (text or "").split() if w.strip()])


def _pick_model_id(preferred: str, discovered: list[str]) -> str:
    pref = (preferred or "").strip()
    if not pref:
        raise ValueError("Model id is empty")

    # Exact match first.
    if pref in discovered:
        return pref

    # Case-insensitive exact match.
    pref_lower = pref.lower()
    for mid in discovered:
        if mid.lower() == pref_lower:
            return mid

    # Substring match (prefer non-preview variants if both exist).
    matches = [mid for mid in discovered if pref_lower in mid.lower()]
    if not matches:
        raise ValueError(f"Model '{pref}' not found in /v1/models")

    def rank(mid: str) -> tuple[int, int, str]:
        lower = mid.lower()
        preview_penalty = 1 if "preview" in lower else 0
        # Prefer newer dated variants if present (YYYYMMDD suffix)
        date_bonus = 0
        parts = lower.split("-")
        if parts and parts[-1].isdigit() and len(parts[-1]) >= 6:
            date_bonus = -int(parts[-1][-6:])  # reverse sort: larger dates win
        return (preview_penalty, date_bonus, len(mid), mid)

    return sorted(matches, key=rank)[0]


def _build_prompt(
    *,
    patient_label: str,
    example_text: str,
    source_reports: list[tuple[str, str]],
    target_body_words: int,
) -> str:
    sources_block = []
    for i, (name, text) in enumerate(source_reports, start=1):
        sources_block.append(
            f"=== COUNCIL SOURCE REPORT {i}: {name} ===\n"
            f"{text.strip()}\n"
        )

    return (
        "You are writing a patient-facing qEEG brain assessment report. This will be a beautifully designed PDF.\n"
        "The patient is intelligent but not a neuroscientist. Write with warmth and clarity.\n\n"
        "DOCUMENT STRUCTURE:\n\n"
        "# Your Brain Assessment Summary\n"
        "(Opening paragraph - 2-3 sentences of warm context about what this document is)\n\n"
        "Then an EXECUTIVE SUMMARY paragraph (no heading) - the emotional heart of the report.\n"
        "What's the story? What happened? Use an analogy if it helps. ~100-150 words.\n\n"
        "## 2. Processing Speed and Attention\n"
        "- A markdown table with the key P300/processing metrics (4-6 rows)\n"
        "- 3-5 bullet points explaining what each metric means in plain language\n"
        "- Weave in occasional analogies naturally\n\n"
        "## 3. Cognitive Performance\n"
        "- A markdown table with Trail Making, reaction time, etc. (3-5 rows)\n"
        "- 3-4 bullet points about what improved, what's notable\n\n"
        "## 4. Brain Rhythm Patterns\n"
        "- A markdown table or bullet list covering alpha, theta/beta, asymmetry\n"
        "- 3-4 bullet points explaining what these patterns suggest\n\n"
        "## 5. What This May Mean\n"
        "- Possible interpretations (what's encouraging)\n"
        "- Important uncertainties (what we can't know from this alone)\n"
        "- Next steps / recommendations\n\n"
        "---\n\n"
        "# Technical Appendix\n\n"
        "*Detailed clinical data for specialist consultations.*\n\n"
        "This section is for clinicians. Write more technically here. Use clinical terminology.\n"
        "Tables should be COMPLETE - do not leave empty cells or truncate.\n\n"
        "## Detailed P300 Site Data\n"
        "(Complete table with all 6 central-parietal sites: C3, CZ, C4, P3, PZ, P4)\n"
        "(Include voltage, latency, and % change for each)\n\n"
        "## Coherence and Network Connectivity\n"
        "(Tables organized by frequency band: Theta, Alpha, Beta)\n"
        "(Include specific coherence values and % changes)\n"
        "- Clinical interpretation using proper terminology\n\n"
        "## Spectral Power Summary\n"
        "(If data available: power changes by site and band)\n\n"
        "TOTAL LENGTH: ~700-900 words for main document, ~400-500 for appendix.\n"
        "IMPORTANT: Complete all tables fully. No empty cells. No truncated sections.\n\n"
        "WRITING STYLE:\n"
        "- Use 'you' and 'your' - this is THEIR brain\n"
        "- Occasional SHORT analogies woven into sentences (not extended metaphors)\n"
        "- Warm but not patronizing. Confident but humble about uncertainty.\n"
        "- Tables should be clean and readable\n\n"
        "HARD RULES:\n"
        "- Do NOT mention LLMs, councils, peer review, AI, or models\n"
        "- Do NOT include patient ID, patient label, DOB, or any identifying info in the document\n"
        "- Synthesize ONLY facts from the source reports—do not invent data\n"
        "- Context: patient had treatment (do NOT name LUMIT or claim causality)\n"
        "- Use '→' for arrows, not LaTeX\n"
        "- No device jargon (WAVi, yield flags) except in Technical Appendix\n"
        "- COMPLETE all tables fully—no empty rows, no truncated content\n\n"
        "EXAMPLE STYLE (for reference):\n"
        f"{example_text.strip()}\n\n"
        "---\n\n"
        "SOURCE MATERIAL (synthesize these):\n"
        + "\n".join(sources_block).strip()
        + "\n\n---\n\n"
        "Now write the complete patient-facing report with appendix.\n"
    )


def _output_stem(*, patient_label: str, version: str, date_str: str) -> str:
    return f"{patient_label}__patient-facing__{version}__{date_str}"


def _portal_markdown_sources(portal_patient_dir: Path) -> list[tuple[str, str]]:
    """
    Fallback path when the DB doesn't have a complete run: use existing council markdown files already
    in the portal folder (excluding previously generated patient-facing outputs).
    """
    if not portal_patient_dir.exists():
        return []
    md_files = [
        p
        for p in portal_patient_dir.glob("*.md")
        if p.is_file() and "__patient-facing__" not in p.name
    ]
    md_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    out: list[tuple[str, str]] = []
    for p in md_files[:3]:
        out.append((f"portal:{p.name}", p.read_text(encoding="utf-8", errors="replace")))
    return out


async def main() -> int:
    ap = argparse.ArgumentParser(description="Generate patient-facing qEEG writeups (Opus 4.5) into portal folders.")
    ap.add_argument("--portal-dir", default="data/portal_patients", help="Portal share dir (default: data/portal_patients)")
    ap.add_argument("--patient-label", action="append", default=[], help="Process only this patient label (repeatable)")
    ap.add_argument("--model", default="claude-opus-4-5", help="Preferred model id (default: claude-opus-4-5)")
    ap.add_argument("--max-tokens", type=int, default=4000, help="Max output tokens (default: 4000)")
    ap.add_argument("--temperature", type=float, default=0.2, help="LLM temperature (default: 0.2)")
    ap.add_argument("--version", default="v1", help="Version tag for output filenames (default: v1)")
    ap.add_argument("--date", default="", help="Override date YYYY-MM-DD (default: today)")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing outputs")
    ap.add_argument("--dry-run", action="store_true", help="Do everything except call the LLM / write outputs")
    args = ap.parse_args()

    ensure_data_dirs()
    init_db()

    portal_dir = Path(args.portal_dir)
    labels = args.patient_label or _discover_portal_patient_labels(portal_dir)
    if not labels:
        print(f"ERROR: no patients found in {portal_dir}", file=sys.stderr)
        return 2

    example = _example_writeup_text()
    target_body_words = _count_words(example)

    date_str = args.date.strip() or _utcnow().date().isoformat()

    llm: AsyncOpenAICompatClient | None = None
    model_id = args.model
    discovered: list[str] = []

    if not args.dry_run:
        llm = AsyncOpenAICompatClient(base_url=CLIPROXY_BASE_URL, api_key=CLIPROXY_API_KEY, timeout_s=600.0)
        try:
            discovered = await llm.list_models()
        except UpstreamError as e:
            print(f"ERROR: failed to reach CLIProxyAPI at {CLIPROXY_BASE_URL}: {e}", file=sys.stderr)
            await llm.aclose()
            return 2
        model_id = _pick_model_id(args.model, discovered)
        print(f"Using model: {model_id}")

    for label in labels:
        bundle = _best_source_bundle_for_label(label)

        # Read 2–3 council final reports (markdown). Prefer DB Stage 6 artifacts; fallback to portal markdowns.
        sources: list[tuple[str, str]] = []
        source_meta: dict[str, Any] = {}
        if bundle is not None:
            for a in bundle.artifacts:
                name = f"stage-{a.stage_num}:{a.kind}:{a.model_id}"
                sources.append((name, _read_text(a.content_path)))
            source_meta = {
                "patient_id": bundle.patient.id,
                "run_id": bundle.run.id,
                "source_artifacts": [
                    {
                        "artifact_id": a.id,
                        "stage_num": a.stage_num,
                        "kind": a.kind,
                        "model_id": a.model_id,
                        "content_path": a.content_path,
                    }
                    for a in bundle.artifacts
                ],
            }
        else:
            out_dir = portal_dir / label
            sources = _portal_markdown_sources(out_dir)
            if not sources:
                print(f"SKIP {label}: no complete DB run, and no portal markdown sources found")
                continue
            source_meta = {
                "patient_id": None,
                "run_id": None,
                "portal_source_files": [name for name, _ in sources],
            }

        prompt = _build_prompt(
            patient_label=label,
            example_text=example,
            source_reports=sources,
            target_body_words=target_body_words,
        )

        out_dir = portal_dir / label
        out_dir.mkdir(parents=True, exist_ok=True)
        stem = _output_stem(patient_label=label, version=args.version, date_str=date_str)
        md_path = out_dir / f"{stem}.md"
        pdf_path = out_dir / f"{stem}.pdf"
        meta_path = out_dir / f"{stem}__meta.json"

        if not args.overwrite and (md_path.exists() or pdf_path.exists()):
            print(f"SKIP {label}: outputs exist (use --overwrite): {md_path.name}")
            continue

        meta: dict[str, Any] = {
            "patient_label": label,
            "llm_model_id": model_id,
            "generated_at": _utcnow().isoformat(),
            **source_meta,
        }

        print(f"GENERATE {label}: {len(sources)} source reports -> {pdf_path.name}")
        if args.dry_run:
            continue

        assert llm is not None
        md = await llm.chat_completions(
            model_id=model_id,
            messages=[{"role": "user", "content": prompt}],
            temperature=float(args.temperature),
            max_tokens=int(args.max_tokens),
            stream=False,
        )
        md = (md or "").strip()

        md_path.write_text(md + "\n", encoding="utf-8")
        meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        render_patient_facing_markdown_to_pdf(md, pdf_path, patient_label=label)

    if llm is not None:
        await llm.aclose()

    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))


