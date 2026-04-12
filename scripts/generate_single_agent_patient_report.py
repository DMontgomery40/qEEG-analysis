#!/usr/bin/env python
from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from backend.config import CLIPROXY_API_KEY, CLIPROXY_BASE_URL, ensure_data_dirs  # noqa: E402
from backend.council.report_text import (  # noqa: E402
    _find_summary_pages,
    _page_session_alias_map,
)
from backend.llm_client import AsyncOpenAICompatClient, UpstreamError  # noqa: E402
from backend.patient_facing_pdf import render_patient_facing_markdown_to_pdf  # noqa: E402
from backend.portal_sync import sync_patient_to_thrylen  # noqa: E402


@dataclass(frozen=True)
class MetricPoint:
    token: str
    unit: str
    source_page: int
    target_range: str | None = None
    sd_token: str | None = None
    questionable: bool = False


@dataclass(frozen=True)
class SessionInfo:
    index: int
    date_iso: str
    age_years: int | None
    summary_page: int
    source_file: str


METRIC_SPECS: tuple[tuple[str, str, str], ...] = (
    ("Physical Reaction Time", "physical_reaction_time", "reaction_time"),
    ("Trail Making Test A", "trail_making_test_a", "simple_numeric"),
    ("Trail Making Test B", "trail_making_test_b", "simple_numeric"),
    ("Audio P300 Delay", "audio_p300_delay", "simple_numeric"),
    ("Audio P300 Voltage", "audio_p300_voltage", "simple_numeric"),
    ("CZ Eyes Closed Theta/Beta (Power)", "cz_theta_beta_ratio", "simple_numeric"),
    ("F3/F4 Eyes Closed Alpha (Power)", "f3_f4_alpha_ratio", "simple_numeric"),
    ("Frontal", "frontal_peak_frequency", "simple_numeric"),
    ("Central-Parietal", "central_parietal_peak_frequency", "simple_numeric"),
    ("Occipital", "occipital_peak_frequency", "simple_numeric"),
)

DISPLAY_ROWS: tuple[tuple[str, tuple[tuple[str, str], ...]], ...] = (
    (
        "Performance",
        (
            ("Physical Reaction Time (ms)", "physical_reaction_time"),
            ("Trail Making Test A (sec)", "trail_making_test_a"),
            ("Trail Making Test B (sec)", "trail_making_test_b"),
        ),
    ),
    (
        "Attention and Evoked Response",
        (
            ("Audio P300 Delay (ms)", "audio_p300_delay"),
            ("Audio P300 Voltage (uV)", "audio_p300_voltage"),
        ),
    ),
    (
        "State and Rhythm",
        (
            ("CZ Eyes Closed Theta/Beta", "cz_theta_beta_ratio"),
            ("F3/F4 Eyes Closed Alpha", "f3_f4_alpha_ratio"),
            ("Frontal Peak Frequency (Hz)", "frontal_peak_frequency"),
            ("Central-Parietal Peak Frequency (Hz)", "central_parietal_peak_frequency"),
            ("Occipital Peak Frequency (Hz)", "occipital_peak_frequency"),
        ),
    ),
)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def _iso_from_mdy(date_text: str) -> str:
    return datetime.strptime(date_text, "%m/%d/%Y").date().isoformat()


def _display_date(date_iso: str) -> str:
    return datetime.strptime(date_iso, "%Y-%m-%d").strftime("%m/%d/%Y")


def _normalize_dash(text: str) -> str:
    return (
        (text or "")
        .replace("\u2013", "-")
        .replace("\u2014", "-")
        .replace("\u2212", "-")
        .replace("\u2018", "'")
        .replace("\u2019", "'")
        .replace("\u00a0", " ")
    )


def _find_line(text: str, label: str) -> str:
    pattern = re.compile(rf"(?im)^\s*{re.escape(label)}(?:\s|$).*$")
    match = pattern.search(text)
    if not match:
        raise ValueError(f"Could not find summary line for {label!r}")
    return _normalize_dash(match.group(0).strip())


def _tail_after_label(line: str, label: str) -> str:
    idx = line.lower().find(label.lower())
    if idx == -1:
        return line
    return line[idx + len(label) :].strip()


def _extract_target_range(tail: str, unit: str) -> str | None:
    match = re.search(
        rf"(\d+(?:\.\d+)?)\s*-\s*(\d+(?:\.\d+)?)\s*{re.escape(unit)}\b",
        tail,
        flags=re.IGNORECASE,
    )
    if not match:
        return None
    return f"{match.group(1)}-{match.group(2)} {unit}"


def _session_rows_from_summary_page(
    page_text: str,
    *,
    aliases: dict[int, int],
    page_num: int,
    source_file: str,
) -> dict[int, SessionInfo]:
    out: dict[int, SessionInfo] = {}
    for match in re.finditer(
        r"Session\s+(\d+)\s*\((\d{1,2}/\d{1,2}/\d{4})\).*?(\d+)\s+yrs",
        page_text,
        flags=re.IGNORECASE,
    ):
        local_idx = int(match.group(1))
        global_idx = aliases.get(local_idx, local_idx)
        out[global_idx] = SessionInfo(
            index=global_idx,
            date_iso=_iso_from_mdy(match.group(2)),
            age_years=int(match.group(3)),
            summary_page=page_num,
            source_file=source_file,
        )
    return out


def _parse_simple_metric(
    line: str,
    *,
    label: str,
    unit: str,
    local_to_global: dict[int, int],
    page_num: int,
) -> dict[int, MetricPoint]:
    tail = _tail_after_label(line, label)
    tokens = re.findall(r"\d+(?:\.\d+)?", tail)
    session_count = len(local_to_global)
    if len(tokens) < session_count:
        raise ValueError(f"Not enough numeric tokens for {label!r}: {line}")
    value_tokens = tokens[:session_count]
    target_range = None
    if len(tokens) >= session_count + 2:
        target_range = f"{tokens[session_count]}-{tokens[session_count + 1]}"
        if unit in {"ms", "sec", "Hz", "uV"}:
            target_range = f"{target_range} {unit}"
    if target_range is None:
        target_range = _extract_target_range(tail, unit)
    questionable = "?" in tail
    out: dict[int, MetricPoint] = {}
    for idx, raw_token in enumerate(value_tokens, start=1):
        global_idx = local_to_global[idx]
        out[global_idx] = MetricPoint(
            token=raw_token,
            unit=unit,
            source_page=page_num,
            target_range=target_range,
            questionable=questionable and idx == 1,
        )
    return out


def _parse_reaction_time(
    line: str,
    *,
    local_to_global: dict[int, int],
    page_num: int,
) -> dict[int, MetricPoint]:
    tail = _tail_after_label(line, "Physical Reaction Time")
    pairs = re.findall(
        r"(\d+(?:\.\d+)?)\s*\(\s*[+£]?\s*(\d+(?:\.\d+)?)\s*\)\s*ms",
        tail,
        flags=re.IGNORECASE,
    )
    if len(pairs) < len(local_to_global):
        raise ValueError(f"Could not parse reaction time line: {line}")
    target_range = _extract_target_range(tail, "ms")
    out: dict[int, MetricPoint] = {}
    for idx, (value_token, sd_token) in enumerate(
        pairs[: len(local_to_global)], start=1
    ):
        global_idx = local_to_global[idx]
        out[global_idx] = MetricPoint(
            token=value_token,
            sd_token=sd_token,
            unit="ms",
            source_page=page_num,
            target_range=target_range,
        )
    return out


def _metric_unit(metric_key: str) -> str:
    units = {
        "physical_reaction_time": "ms",
        "trail_making_test_a": "sec",
        "trail_making_test_b": "sec",
        "audio_p300_delay": "ms",
        "audio_p300_voltage": "uV",
        "cz_theta_beta_ratio": "ratio",
        "f3_f4_alpha_ratio": "ratio",
        "frontal_peak_frequency": "Hz",
        "central_parietal_peak_frequency": "Hz",
        "occipital_peak_frequency": "Hz",
    }
    return units[metric_key]


def _merge_metric_points(
    store: dict[str, dict[int, MetricPoint]],
    *,
    metric_key: str,
    parsed: dict[int, MetricPoint],
    duplicate_checks: list[str],
) -> None:
    target = store.setdefault(metric_key, {})
    for session_idx, point in parsed.items():
        existing = target.get(session_idx)
        if existing is None:
            target[session_idx] = point
            continue
        if existing.token == point.token and existing.sd_token == point.sd_token:
            duplicate_checks.append(
                f"{metric_key} session {session_idx}: page {existing.source_page} matches page {point.source_page}"
            )
            continue
        raise ValueError(
            f"Conflicting values for {metric_key} session {session_idx}: "
            f"page {existing.source_page}={existing.token} vs page {point.source_page}={point.token}"
        )


def _parse_summary_metrics(
    *,
    report_text: str,
    report_dir: Path,
    metadata: dict[str, Any],
) -> tuple[
    dict[int, SessionInfo], dict[str, dict[int, MetricPoint]], list[str], list[str]
]:
    page_aliases = _page_session_alias_map(report_text)
    candidate_summary_pages = _find_summary_pages(
        report_text, page_count=int(metadata.get("page_count") or 0)
    )
    summary_pages: list[int] = []
    for page_num in candidate_summary_pages:
        page_text = _read_text(
            report_dir / "sources" / f"page-{page_num}.tesseract.txt"
        )
        page_text_upper = page_text.upper()
        if (
            "ASSESSMENT SCORES" in page_text_upper
            and "PHYSICAL REACTION TIME" in page_text_upper
            and "AUDIO P300 DELAY" in page_text_upper
        ):
            summary_pages.append(page_num)

    if not summary_pages:
        raise ValueError("Could not locate any summary pages in combined extraction")

    source_files_by_page: dict[int, str] = {}
    for page in metadata.get("pages", []):
        if not isinstance(page, dict):
            continue
        page_num = page.get("page")
        source_file = page.get("source_file")
        if isinstance(page_num, int) and isinstance(source_file, str):
            source_files_by_page[page_num] = source_file

    sessions: dict[int, SessionInfo] = {}
    metrics: dict[str, dict[int, MetricPoint]] = {}
    duplicate_checks: list[str] = []
    notes: list[str] = []

    for page_num in summary_pages:
        local_to_global = page_aliases.get(page_num) or {}
        if not local_to_global:
            raise ValueError(
                f"Summary page {page_num} is missing session alias markers"
            )

        page_text = _read_text(
            report_dir / "sources" / f"page-{page_num}.tesseract.txt"
        )
        source_file = source_files_by_page.get(page_num, f"page-{page_num}")

        for session_idx, info in _session_rows_from_summary_page(
            page_text,
            aliases=local_to_global,
            page_num=page_num,
            source_file=source_file,
        ).items():
            existing = sessions.get(session_idx)
            if existing is None:
                sessions[session_idx] = info
                continue
            if existing.date_iso != info.date_iso:
                raise ValueError(
                    f"Conflicting dates for session {session_idx}: "
                    f"{existing.date_iso} vs {info.date_iso}"
                )

        if (
            "SYNC BLINKS REPORTED WHICH MAY AFFECT FRONTAL DEPTH VALUES"
            in page_text.upper()
        ):
            notes.append(
                f"Summary page {page_num} ({source_file}) notes sync blinks that may affect frontal depth values."
            )
        if (
            "BLACK Xs INDICATE LOCATIONS WITH LESS THAN 20 CLEAN P300 RARE RESPONSES"
            in page_text
        ):
            notes.append(
                f"Summary page {page_num} ({source_file}) warns that some topographic colors may be affected by low-yield P300 responses."
            )

        for label, metric_key, parser_kind in METRIC_SPECS:
            line = _find_line(page_text, label)
            if parser_kind == "reaction_time":
                parsed = _parse_reaction_time(
                    line, local_to_global=local_to_global, page_num=page_num
                )
            else:
                parsed = _parse_simple_metric(
                    line,
                    label=label,
                    unit=_metric_unit(metric_key),
                    local_to_global=local_to_global,
                    page_num=page_num,
                )
            _merge_metric_points(
                metrics,
                metric_key=metric_key,
                parsed=parsed,
                duplicate_checks=duplicate_checks,
            )

    if len(sessions) != 5:
        raise ValueError(f"Expected 5 sessions, found {sorted(sessions)}")

    for metric_key, by_session in metrics.items():
        if sorted(by_session) != [1, 2, 3, 4, 5]:
            raise ValueError(
                f"Metric {metric_key} does not cover all five sessions: {sorted(by_session)}"
            )

    notes.append("HAM-A and PHQ-9 are marked N/A on both source summary pages.")
    if metrics["frontal_peak_frequency"][1].questionable:
        notes.append(
            "Session 1 frontal peak frequency is explicitly flagged as questionable in the source report."
        )
    if metrics["central_parietal_peak_frequency"][1].questionable:
        notes.append(
            "Session 1 central-parietal peak frequency is explicitly flagged as questionable in the source report."
        )
    if metrics["occipital_peak_frequency"][1].questionable:
        notes.append(
            "Session 1 occipital peak frequency is explicitly flagged as questionable in the source report."
        )

    return sessions, metrics, duplicate_checks, notes


def _target_summary(points: dict[int, MetricPoint]) -> str:
    grouped: dict[str, list[int]] = {}
    for session_idx, point in sorted(points.items()):
        target = point.target_range or "n/a"
        grouped.setdefault(target, []).append(session_idx)
    if len(grouped) == 1:
        return next(iter(grouped))

    parts: list[str] = []
    for target, session_indices in grouped.items():
        if session_indices == [1, 2, 3, 4]:
            label = "S1-S4"
        elif len(session_indices) == 1:
            label = f"S{session_indices[0]}"
        else:
            label = ",".join(f"S{i}" for i in session_indices)
        parts.append(f"{label}: {target}")
    return "; ".join(parts)


def _format_cell(metric_key: str, point: MetricPoint) -> str:
    if metric_key == "physical_reaction_time":
        value = f"{point.token} (+{point.sd_token})"
    else:
        value = point.token
    if point.questionable:
        value = f"{value} ?"
    return value


def _markdown_table(
    *,
    title: str,
    rows: tuple[tuple[str, str], ...],
    sessions: dict[int, SessionInfo],
    metrics: dict[str, dict[int, MetricPoint]],
    include_reference_range: bool,
) -> str:
    header = ["Metric"]
    for idx in sorted(sessions):
        info = sessions[idx]
        header.append(f"S{idx} {_display_date(info.date_iso)}")
    if include_reference_range:
        header.append("Reference Range")

    lines = [
        f"## {title}",
        "",
        "| " + " | ".join(header) + " |",
        "|" + "|".join("---" for _ in header) + "|",
    ]

    for row_label, metric_key in rows:
        cells = [row_label]
        by_session = metrics[metric_key]
        for idx in sorted(sessions):
            cells.append(_format_cell(metric_key, by_session[idx]))
        if include_reference_range:
            cells.append(_target_summary(by_session))
        lines.append("| " + " | ".join(cells) + " |")

    return "\n".join(lines)


def _chronology_table(sessions: dict[int, SessionInfo]) -> str:
    lines = [
        "## Session Chronology",
        "",
        "| Session | Date | Age | Summary Source |",
        "|---|---|---|---|",
    ]
    for idx in sorted(sessions):
        info = sessions[idx]
        age = str(info.age_years) if info.age_years is not None else "n/a"
        lines.append(
            f"| Session {idx} | {_display_date(info.date_iso)} | {age} | {info.source_file} page {info.summary_page} |"
        )
    return "\n".join(lines)


def _analysis_fact_pack(
    *,
    sessions: dict[int, SessionInfo],
    metrics: dict[str, dict[int, MetricPoint]],
    duplicate_checks: list[str],
    notes: list[str],
    manifest: dict[str, Any],
) -> str:
    sections = [
        "Chronology:",
        _chronology_table(sessions),
    ]
    for title, rows in DISPLAY_ROWS:
        sections.append(
            _markdown_table(
                title=title,
                rows=rows,
                sessions=sessions,
                metrics=metrics,
                include_reference_range=True,
            )
        )
    sections.append("Manifest Notes:")
    sections.append(f"- {manifest.get('notes', '').strip()}")
    sections.append("- The second PDF's local Session 1 maps to global Session 4.")
    sections.append("- The second PDF's local Session 2 maps to global Session 5.")
    sections.append(
        "- Session 4 appears in both PDFs and should remain distinct from Session 5."
    )
    if duplicate_checks:
        sections.append("Duplicate Session 4 Checks:")
        sections.extend(f"- {item}" for item in duplicate_checks)
    if notes:
        sections.append("Source Cautions:")
        sections.extend(f"- {item}" for item in notes)
    return "\n\n".join(sections).strip()


def _analysis_prompt(*, fact_pack: str) -> str:
    return (
        "You are writing an internal qEEG analysis markdown for a patient-facing report.\n"
        "Use only the facts below.\n"
        "Do not invent diagnoses, treatment causes, or unsupported physiology.\n"
        "Treat Session 4 (01/14/2026) and Session 5 (02/18/2026) as two separate global sessions.\n"
        "Be explicit about what looks improved, what is mixed, and what remains uncertain.\n"
        "The source material has limited symptom scales and a few quality flags; mention that plainly.\n\n"
        "Return markdown with exactly these top-level headings:\n"
        "## Longitudinal Interpretation\n"
        "## Improvement Signals\n"
        "## Mixed or Uncertain Findings\n"
        "## Data Quality and Limitations\n"
        "## Suggested Human Review Checks\n\n"
        "Fact pack:\n\n"
        f"{fact_pack}\n"
    )


def _patient_prompt(*, fact_pack: str, analysis_summary: str) -> str:
    return (
        "You are writing only the narrative sections for a patient-facing qEEG summary.\n"
        "Write for an intelligent non-specialist. Use warm, clear, clinically careful language.\n"
        "Use only the facts below and the internal analysis below.\n"
        "Do not include the patient label or any identifying details.\n"
        "Do not mention AI, OCR, source PDFs, or model names.\n"
        "Do not claim treatment causality or make diagnostic claims.\n"
        "Avoid fake certainty. When something is mixed or unclear, say so directly.\n"
        "Do not add tables. The exact tables will be inserted separately.\n\n"
        "Return markdown with exactly these top-level headings:\n"
        "## Big Picture\n"
        "## What Seems Improved\n"
        "## What Looks Mixed or Uncertain\n"
        "## What This May Mean\n"
        "## Important Limits\n\n"
        "Exact facts:\n\n"
        f"{fact_pack}\n\n"
        "Internal analysis:\n\n"
        f"{analysis_summary.strip()}\n"
    )


def _pick_openai_model(discovered: list[str], preferred: str) -> str:
    if preferred in discovered:
        return preferred

    plain_models = [
        mid
        for mid in discovered
        if mid.startswith("gpt-5") and "codex" not in mid.lower()
    ]
    if not plain_models:
        raise ValueError("No OpenAI GPT-5 model is available from CLIProxyAPI")

    def rank(model_id: str) -> tuple[int, int, int, str]:
        if model_id == "gpt-5":
            return (5, 0, 0, model_id)
        match = re.fullmatch(r"gpt-5(?:\.(\d+))?", model_id)
        if match:
            minor = int(match.group(1) or 0)
            return (5, minor, 1, model_id)
        return (4, 0, 0, model_id)

    return sorted(plain_models, key=rank, reverse=True)[0]


async def _generate_markdown(
    llm: AsyncOpenAICompatClient,
    *,
    model_id: str,
    prompt: str,
    max_output_tokens: int,
) -> str:
    if model_id.startswith("gpt-5"):
        return await llm.responses(
            model_id=model_id,
            input_data=[
                {
                    "role": "user",
                    "content": [{"type": "input_text", "text": prompt}],
                }
            ],
            reasoning_effort="xhigh",
            max_output_tokens=max_output_tokens,
        )
    return await llm.chat_completions(
        model_id=model_id,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=max_output_tokens,
        stream=False,
    )


def _patient_markdown(
    *,
    narrative_md: str,
    sessions: dict[int, SessionInfo],
    metrics: dict[str, dict[int, MetricPoint]],
) -> str:
    parts = [
        "# Your Brain Assessment Summary",
        "",
        narrative_md.strip(),
        "",
        _markdown_table(
            title="Five-Session Snapshot",
            rows=DISPLAY_ROWS[0][1] + DISPLAY_ROWS[1][1] + DISPLAY_ROWS[2][1],
            sessions=sessions,
            metrics=metrics,
            include_reference_range=False,
        ),
        "",
        "# Technical Appendix",
        "",
        _chronology_table(sessions),
    ]
    for title, rows in DISPLAY_ROWS:
        parts.append("")
        parts.append(
            _markdown_table(
                title=title,
                rows=rows,
                sessions=sessions,
                metrics=metrics,
                include_reference_range=True,
            )
        )
    parts.append("")
    parts.append(
        "*Reference ranges in the source report shift slightly at Session 5 because that summary page lists age 14 instead of 13.*"
    )
    return "\n".join(parts).strip() + "\n"


def _analysis_markdown(
    *,
    patient_label: str,
    analysis_narrative: str,
    sessions: dict[int, SessionInfo],
    metrics: dict[str, dict[int, MetricPoint]],
    duplicate_checks: list[str],
    notes: list[str],
    manifest: dict[str, Any],
) -> str:
    parts = [
        f"# Single-Agent 5-Session Analysis: {patient_label}",
        "",
        f"Generated at {_utcnow().isoformat()}",
        "",
        _chronology_table(sessions),
    ]
    for title, rows in DISPLAY_ROWS:
        parts.append("")
        parts.append(
            _markdown_table(
                title=title,
                rows=rows,
                sessions=sessions,
                metrics=metrics,
                include_reference_range=True,
            )
        )
    if duplicate_checks:
        parts.extend(
            [
                "",
                "## Duplicate Session 4 Consistency Check",
                "",
                *[f"- {item}" for item in duplicate_checks],
            ]
        )
    if notes:
        parts.extend(
            [
                "",
                "## Source Notes",
                "",
                *[f"- {item}" for item in notes],
            ]
        )
    if manifest.get("notes"):
        parts.extend(
            [
                "",
                "## Manifest Notes",
                "",
                f"- {manifest['notes'].strip()}",
            ]
        )
    parts.extend(["", analysis_narrative.strip(), ""])
    return "\n".join(parts).strip() + "\n"


def _meta_payload(
    *,
    manifest_path: Path,
    report_dir: Path,
    model_id: str,
    sessions: dict[int, SessionInfo],
    metrics: dict[str, dict[int, MetricPoint]],
    duplicate_checks: list[str],
    notes: list[str],
    analysis_path: Path,
    patient_md_path: Path,
    patient_pdf_path: Path,
    outputs_written: bool = False,
) -> dict[str, Any]:
    return {
        "generated_at": _utcnow().isoformat(),
        "manifest_path": str(manifest_path),
        "combined_report_dir": str(report_dir),
        "llm_model_id": model_id,
        "reasoning_effort": "xhigh" if model_id.startswith("gpt-5") else None,
        "sessions": {
            str(idx): {
                "date": info.date_iso,
                "age_years": info.age_years,
                "summary_page": info.summary_page,
                "source_file": info.source_file,
            }
            for idx, info in sorted(sessions.items())
        },
        "metrics": {
            metric_key: {
                str(session_idx): {
                    "value": point.token,
                    "sd_plus_minus": point.sd_token,
                    "unit": point.unit,
                    "target_range": point.target_range,
                    "source_page": point.source_page,
                    "questionable": point.questionable,
                }
                for session_idx, point in sorted(by_session.items())
            }
            for metric_key, by_session in sorted(metrics.items())
        },
        "duplicate_checks": duplicate_checks,
        "notes": notes,
        "outputs": {
            "analysis_markdown": str(analysis_path),
            "patient_markdown": str(patient_md_path),
            "patient_pdf": str(patient_pdf_path),
        },
        "verification": {
            "has_five_sessions": sorted(sessions) == [1, 2, 3, 4, 5],
            "session_4_date": sessions[4].date_iso,
            "session_5_date": sessions[5].date_iso,
            "session_4_duplicate_checks": len(duplicate_checks),
            "analysis_exists": outputs_written or analysis_path.exists(),
            "patient_markdown_exists": outputs_written or patient_md_path.exists(),
            "patient_pdf_exists": outputs_written or patient_pdf_path.exists(),
        },
    }


def _resolved_manifest_source_paths(
    *, manifest: dict[str, Any], manifest_path: Path
) -> list[Path]:
    raw_sources = manifest.get("sources")
    if not isinstance(raw_sources, list) or not raw_sources:
        raise ValueError(
            "Manifest requires a non-empty sources array for auto-discovery"
        )

    resolved: list[Path] = []
    for item in raw_sources:
        if not isinstance(item, dict) or not isinstance(item.get("path"), str):
            raise ValueError("Manifest sources must contain path strings")
        source_path = Path(item["path"]).expanduser()
        if not source_path.is_absolute():
            repo_relative = (_REPO_ROOT / source_path).resolve()
            manifest_relative = (manifest_path.parent / source_path).resolve()
            source_path = repo_relative if repo_relative.exists() else manifest_relative
        resolved.append(source_path.resolve())
    return resolved


def _find_combined_report_dir(
    *,
    manifest: dict[str, Any],
    manifest_path: Path,
    reports_root: Path,
) -> Path:
    expected_source_paths = {
        str(path)
        for path in _resolved_manifest_source_paths(
            manifest=manifest, manifest_path=manifest_path
        )
    }
    expected_source_names = {Path(path).name for path in expected_source_paths}
    manifest_path_resolved = str(manifest_path.resolve())

    candidates: list[tuple[int, Path]] = []
    for meta_path in reports_root.glob("*/*/metadata.json"):
        try:
            metadata = json.loads(_read_text(meta_path))
        except Exception:
            continue

        synthetic = metadata.get("synthetic_combined")
        synthetic = synthetic if isinstance(synthetic, dict) else {}

        metadata_source_paths = {
            str(Path(item["path"]).expanduser().resolve())
            for item in synthetic.get("source_files", [])
            if isinstance(item, dict) and isinstance(item.get("path"), str)
        }
        score = 0
        if synthetic.get("manifest_path") == manifest_path_resolved:
            score = 3
        elif metadata_source_paths == expected_source_paths:
            score = 2
        elif metadata_source_paths:
            continue
        else:
            page_entries = metadata.get("pages")
            if not isinstance(page_entries, list):
                continue
            source_files = {
                item.get("source_file")
                for item in page_entries
                if isinstance(item, dict) and isinstance(item.get("source_file"), str)
            }
            if source_files == expected_source_names:
                score = 1

        if score:
            candidates.append((score, meta_path.parent))

    if not candidates:
        raise FileNotFoundError(
            "Could not locate a combined report directory matching the manifest sources"
        )
    best_score = max(score for score, _path in candidates)
    best = sorted(path for score, path in candidates if score == best_score)
    if len(best) > 1:
        raise RuntimeError(
            "Multiple combined report directories match the manifest sources: "
            + ", ".join(str(path) for path in best)
        )
    return best[0]


async def main() -> int:
    ap = argparse.ArgumentParser(
        description="Generate a single-agent five-session patient-facing qEEG report."
    )
    ap.add_argument(
        "--manifest",
        default="data/portal_patients/01-01-2013-0/combined_5sessions.manifest.json",
        help="Combined manifest path",
    )
    ap.add_argument(
        "--combined-report-dir",
        default="",
        help="Combined extracted report dir (auto-discovered from manifest when omitted)",
    )
    ap.add_argument(
        "--reports-root",
        default="data/reports",
        help="Reports root used for auto-discovery",
    )
    ap.add_argument(
        "--output-dir",
        default="",
        help="Output directory (defaults to manifest parent)",
    )
    ap.add_argument(
        "--preferred-model",
        default="gpt-5.4",
        help="Preferred OpenAI model id (default: gpt-5.4)",
    )
    ap.add_argument(
        "--analysis-max-output-tokens",
        type=int,
        default=2600,
        help="Max output tokens for the internal analysis narrative",
    )
    ap.add_argument(
        "--patient-max-output-tokens",
        type=int,
        default=2200,
        help="Max output tokens for the patient-facing narrative",
    )
    ap.add_argument(
        "--version",
        default="v1",
        help="Version tag used in output filenames",
    )
    ap.add_argument(
        "--date",
        default="",
        help="Override output date (YYYY-MM-DD). Defaults to today UTC.",
    )
    ap.add_argument(
        "--overwrite", action="store_true", help="Overwrite existing outputs"
    )
    args = ap.parse_args()

    ensure_data_dirs()

    manifest_path = Path(args.manifest).expanduser()
    manifest = json.loads(_read_text(manifest_path))
    patient_label = str(manifest.get("patient_label") or "").strip()
    if not patient_label:
        raise ValueError("Manifest is missing patient_label")

    if args.combined_report_dir:
        report_dir = Path(args.combined_report_dir).expanduser()
    else:
        report_dir = _find_combined_report_dir(
            manifest=manifest,
            manifest_path=manifest_path,
            reports_root=Path(args.reports_root).expanduser(),
        )

    output_dir = (
        Path(args.output_dir).expanduser() if args.output_dir else manifest_path.parent
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    extracted_path = report_dir / "extracted_enhanced.txt"
    metadata_path = report_dir / "metadata.json"
    if not extracted_path.exists():
        raise FileNotFoundError(f"Missing extracted text: {extracted_path}")
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing metadata: {metadata_path}")

    report_text = _read_text(extracted_path)
    metadata = json.loads(_read_text(metadata_path))
    sessions, metrics, duplicate_checks, notes = _parse_summary_metrics(
        report_text=report_text,
        report_dir=report_dir,
        metadata=metadata,
    )

    fact_pack = _analysis_fact_pack(
        sessions=sessions,
        metrics=metrics,
        duplicate_checks=duplicate_checks,
        notes=notes,
        manifest=manifest,
    )

    stem_date = args.date.strip() or _utcnow().date().isoformat()
    analysis_stem = (
        f"{patient_label}__single-agent-5session-analysis__{args.version}__{stem_date}"
    )
    patient_stem = f"{patient_label}__patient-facing__{args.version}__{stem_date}"
    analysis_path = output_dir / f"{analysis_stem}.md"
    patient_md_path = output_dir / f"{patient_stem}.md"
    patient_pdf_path = output_dir / f"{patient_stem}.pdf"
    meta_path = output_dir / f"{patient_stem}__meta.json"

    if not args.overwrite:
        existing = [
            path
            for path in (analysis_path, patient_md_path, patient_pdf_path, meta_path)
            if path.exists()
        ]
        if existing:
            raise FileExistsError(
                "Output files already exist (use --overwrite): "
                + ", ".join(path.name for path in existing)
            )

    llm = AsyncOpenAICompatClient(
        base_url=CLIPROXY_BASE_URL,
        api_key=CLIPROXY_API_KEY,
        timeout_s=600.0,
    )
    try:
        discovered = await llm.list_models()
    except UpstreamError as exc:
        await llm.aclose()
        raise RuntimeError(
            f"Failed to list models from CLIProxyAPI at {CLIPROXY_BASE_URL}: {exc}"
        ) from exc

    model_id = _pick_openai_model(discovered, args.preferred_model)
    print(f"Using model: {model_id}", flush=True)
    print(f"Combined report dir: {report_dir}", flush=True)

    try:
        analysis_narrative = await _generate_markdown(
            llm,
            model_id=model_id,
            prompt=_analysis_prompt(fact_pack=fact_pack),
            max_output_tokens=args.analysis_max_output_tokens,
        )
        print("Generated internal analysis narrative", flush=True)
        analysis_md = _analysis_markdown(
            patient_label=patient_label,
            analysis_narrative=analysis_narrative,
            sessions=sessions,
            metrics=metrics,
            duplicate_checks=duplicate_checks,
            notes=notes,
            manifest=manifest,
        )

        patient_narrative = await _generate_markdown(
            llm,
            model_id=model_id,
            prompt=_patient_prompt(
                fact_pack=fact_pack,
                analysis_summary=analysis_narrative,
            ),
            max_output_tokens=args.patient_max_output_tokens,
        )
        print("Generated patient-facing narrative", flush=True)
        patient_md = _patient_markdown(
            narrative_md=patient_narrative,
            sessions=sessions,
            metrics=metrics,
        )
    finally:
        await llm.aclose()

    with tempfile.TemporaryDirectory(
        dir=output_dir, prefix=f".{patient_stem}."
    ) as temp_dir_raw:
        temp_dir = Path(temp_dir_raw)
        temp_analysis_path = temp_dir / analysis_path.name
        temp_patient_md_path = temp_dir / patient_md_path.name
        temp_patient_pdf_path = temp_dir / patient_pdf_path.name
        temp_meta_path = temp_dir / meta_path.name

        temp_analysis_path.write_text(analysis_md, encoding="utf-8")
        temp_patient_md_path.write_text(patient_md, encoding="utf-8")
        render_patient_facing_markdown_to_pdf(
            patient_md,
            temp_patient_pdf_path,
            patient_label=patient_label,
        )

        meta = _meta_payload(
            manifest_path=manifest_path,
            report_dir=report_dir,
            model_id=model_id,
            sessions=sessions,
            metrics=metrics,
            duplicate_checks=duplicate_checks,
            notes=notes,
            analysis_path=analysis_path,
            patient_md_path=patient_md_path,
            patient_pdf_path=patient_pdf_path,
            outputs_written=True,
        )
        temp_meta_path.write_text(
            json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

        if not args.overwrite:
            existing = [
                path
                for path in (
                    analysis_path,
                    patient_md_path,
                    patient_pdf_path,
                    meta_path,
                )
                if path.exists()
            ]
            if existing:
                raise FileExistsError(
                    "Output files already exist (use --overwrite): "
                    + ", ".join(path.name for path in existing)
                )

        moved_paths: list[Path] = []
        try:
            for src, dst in (
                (temp_analysis_path, analysis_path),
                (temp_patient_md_path, patient_md_path),
                (temp_patient_pdf_path, patient_pdf_path),
                (temp_meta_path, meta_path),
            ):
                src.replace(dst)
                moved_paths.append(dst)
        except Exception:
            for path in moved_paths:
                try:
                    path.unlink(missing_ok=True)
                except Exception:
                    pass
            raise

    print(f"Wrote analysis markdown: {analysis_path}", flush=True)
    print(f"Wrote patient markdown: {patient_md_path}", flush=True)
    print(f"Wrote patient PDF: {patient_pdf_path}", flush=True)
    print(f"Wrote metadata: {meta_path}", flush=True)
    synced = sync_patient_to_thrylen(patient_label)
    print(
        f"Portal sync {'completed' if synced else 'skipped'} for {patient_label}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
