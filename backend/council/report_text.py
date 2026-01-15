from __future__ import annotations

import re
from typing import Any


def _expected_session_indices(report_text: str) -> list[int]:
    # Prefer explicit "Session N (" markers, fall back to any "Session N".
    indices = {int(m.group(1)) for m in re.finditer(r"Session\s+(\d+)\s*\(", report_text)}
    if not indices:
        indices = {int(m.group(1)) for m in re.finditer(r"\bSession\s+(\d+)\b", report_text)}
    return sorted(indices) if indices else [1]


def _page_section(report_text: str, *, page_num: int) -> str:
    marker = f"=== PAGE {page_num} /"
    start = report_text.find(marker)
    if start == -1:
        return report_text
    next_marker = f"=== PAGE {page_num + 1} /"
    end = report_text.find(next_marker, start)
    if end == -1:
        end = len(report_text)
    return report_text[start:end]


def _page_count_from_markers(report_text: str) -> int | None:
    m = re.search(r"=== PAGE 1 / (\d+) ===", report_text)
    if not m:
        return None
    try:
        return int(m.group(1))
    except Exception:
        return None


def _find_p300_rare_comparison_pages(report_text: str, *, page_count: int) -> list[int]:
    """
    Best-effort deterministic locator for WAVi-style "P300 Rare Comparison" pages using OCR text.

    We avoid relying on model-provided page inventory because strict data availability requires stable targeting.
    """
    matches: list[int] = []
    for page in range(1, max(1, page_count) + 1):
        section = _page_section(report_text, page_num=page)
        s = section.lower()
        if "p300 rare comparison" in s:
            matches.append(page)
            continue
        if "rare comparison" in s and "p300" in s:
            matches.append(page)
            continue
        # Heuristic: the rare comparison page typically mentions yield display threshold and shows µV/MS.
        if "yield display threshold" in s and "uv" in s and "ms" in s and "p300" in s:
            matches.append(page)

    # Preserve order and uniqueness.
    seen: set[int] = set()
    out: list[int] = []
    for p in matches:
        if p in seen:
            continue
        seen.add(p)
        out.append(p)
    return out


def _number_tokens(text: str) -> list[str]:
    # Keep as strings initially to preserve things like ">=260" in shown_as.
    return re.findall(r"\d+(?:\.\d+)?", text)


def _safe_float(token: str) -> float | None:
    try:
        return float(token)
    except Exception:
        return None


def _safe_int(token: str) -> int | None:
    try:
        return int(float(token))
    except Exception:
        return None


def _facts_from_report_text_summary(report_text: str, *, expected_sessions: list[int]) -> list[dict[str, Any]]:
    """
    Deterministically extract key summary metrics from the OCR text (usually PAGE 1).

    This reduces nondeterminism in strict mode by not relying on a vision model to re-transcribe values that are
    already present in extracted_enhanced.txt.
    """
    page1 = _page_section(report_text, page_num=1)

    def find_line_contains(needle: str) -> str | None:
        m = re.search(rf"(?im)^.*{re.escape(needle)}.*$", page1)
        return m.group(0).strip() if m else None

    def find_line_startswith(label: str) -> str | None:
        m = re.search(rf"(?im)^{re.escape(label)}\b.*$", page1)
        return m.group(0).strip() if m else None

    out: list[dict[str, Any]] = []

    def tail_after(haystack: str, needle: str) -> str:
        h = haystack or ""
        n = needle or ""
        if not h or not n:
            return h
        idx = h.lower().find(n.lower())
        if idx == -1:
            return h
        return h[idx + len(n) :]

    # Physical Reaction Time (includes SDs).
    rt_line = find_line_contains("Physical Reaction Time")
    if rt_line:
        # Try a structured parse first.
        m = re.search(
            r"Physical Reaction Time\s+(\d+)\s*\(\s*\+?(\d+)\s*\)\s*ms\s+(\d+)\s*\(\s*\+?(\d+)\s*\)\s*ms\s+(\d+)\s*\(\s*\+?(\d+)\s*\)\s*ms\s+(\d+)\s*[-–]\s*(\d+)\s*ms",
            rt_line,
            flags=re.IGNORECASE,
        )
        if m:
            vals = [int(m.group(1)), int(m.group(3)), int(m.group(5))]
            sds = [int(m.group(2)), int(m.group(4)), int(m.group(6))]
            target = f"{m.group(7)}-{m.group(8)} ms"
            for sess, val, sd in zip(expected_sessions, vals, sds):
                out.append(
                    {
                        "fact_type": "performance_metric",
                        "metric": "physical_reaction_time",
                        "session_index": sess,
                        "value": val,
                        "unit": "ms",
                        "sd_plus_minus": sd,
                        "target_range": target,
                        "shown_as": None,
                        "source_page": 1,
                    }
                )
        else:
            toks = _number_tokens(rt_line)
            if len(toks) >= 3:
                vals = [toks[0], toks[2], toks[4]] if len(toks) >= 5 else toks[:3]
                for sess, tok in zip(expected_sessions, vals):
                    val = _safe_int(tok)
                    if val is None:
                        continue
                    out.append(
                        {
                            "fact_type": "performance_metric",
                            "metric": "physical_reaction_time",
                            "session_index": sess,
                            "value": val,
                            "unit": "ms",
                            "shown_as": rt_line.strip(),
                            "source_page": 1,
                        }
                    )

    # Trail Making Test A/B.
    for label, metric in (("Trail Making Test A", "trail_making_test_a"), ("Trail Making Test B", "trail_making_test_b")):
        line = find_line_contains(label)
        if not line:
            continue
        m = re.search(
            rf"{re.escape(label)}\s+(\d+)\s*sec\s+(\d+)\s*sec\s+(\d+)\s*sec\s+(\d+)\s*[-–]\s*(\d+)\s*sec",
            line,
            flags=re.IGNORECASE,
        )
        if m:
            vals = [int(m.group(1)), int(m.group(2)), int(m.group(3))]
            target = f"{m.group(4)}-{m.group(5)} sec"
            for sess, val in zip(expected_sessions, vals):
                out.append(
                    {
                        "fact_type": "performance_metric",
                        "metric": metric,
                        "session_index": sess,
                        "value": val,
                        "unit": "sec",
                        "target_range": target,
                        "shown_as": None,
                        "source_page": 1,
                    }
                )
        else:
            toks = _number_tokens(line)
            if len(toks) >= len(expected_sessions):
                for sess, tok in zip(expected_sessions, toks[: len(expected_sessions)]):
                    val = _safe_int(tok)
                    if val is None:
                        continue
                    out.append(
                        {
                            "fact_type": "performance_metric",
                            "metric": metric,
                            "session_index": sess,
                            "value": val,
                            "unit": "sec",
                            "shown_as": line.strip(),
                            "source_page": 1,
                        }
                    )

    # Audio P300 Delay/Voltage.
    delay_line = find_line_contains("Audio P300 Delay")
    if delay_line:
        # Avoid pulling digits from the label itself (e.g., "P300" -> 300).
        toks = _number_tokens(tail_after(delay_line, "Audio P300 Delay"))
        if len(toks) >= len(expected_sessions):
            vals = toks[: len(expected_sessions)]
            range_toks = toks[len(expected_sessions) : len(expected_sessions) + 2]
            target = None
            if len(range_toks) == 2:
                target = f"{range_toks[0]}-{range_toks[1]} ms"
            for sess, tok in zip(expected_sessions, vals):
                val = _safe_int(tok)
                if val is None:
                    continue
                out.append(
                    {
                        "fact_type": "evoked_potential",
                        "metric": "audio_p300_delay",
                        "session_index": sess,
                        "value": val,
                        "unit": "ms",
                        "target_range": target,
                        "shown_as": delay_line.strip(),
                        "source_page": 1,
                    }
                )

    volt_line = find_line_contains("Audio P300 Voltage")
    if volt_line:
        # Avoid pulling digits from the label itself (e.g., "P300" -> 300).
        toks = _number_tokens(tail_after(volt_line, "Audio P300 Voltage"))
        if len(toks) >= len(expected_sessions):
            vals = toks[: len(expected_sessions)]
            range_toks = toks[len(expected_sessions) : len(expected_sessions) + 2]
            target = None
            if len(range_toks) == 2:
                target = f"{range_toks[0]}-{range_toks[1]} µV"
            for sess, tok in zip(expected_sessions, vals):
                val = _safe_float(tok)
                if val is None:
                    continue
                out.append(
                    {
                        "fact_type": "evoked_potential",
                        "metric": "audio_p300_voltage",
                        "session_index": sess,
                        "value": val,
                        "unit": "µV",
                        "target_range": target,
                        "shown_as": volt_line.strip(),
                        "source_page": 1,
                    }
                )

    # State summary metrics (ratios).
    state_lines = [
        ("CZ Eyes Closed Theta/Beta", "cz_theta_beta_ratio_ec", "ratio"),
        ("F3/F4 Eyes Closed Alpha", "f3_f4_alpha_ratio_ec", "ratio"),
    ]
    for needle, metric, unit in state_lines:
        line = find_line_contains(needle)
        if not line:
            continue
        # Avoid pulling digits from the label itself (e.g., "F3/F4" -> 3, 4).
        toks = _number_tokens(tail_after(line, needle))
        if len(toks) < len(expected_sessions):
            continue
        vals = toks[: len(expected_sessions)]
        range_toks = toks[len(expected_sessions) : len(expected_sessions) + 2]
        target = None
        if len(range_toks) == 2:
            target = f"{range_toks[0]}-{range_toks[1]}"
        for sess, tok in zip(expected_sessions, vals):
            val = _safe_float(tok)
            if val is None:
                continue
            out.append(
                {
                    "fact_type": "state_metric",
                    "metric": metric,
                    "session_index": sess,
                    "value": val,
                    "unit": unit,
                    "target_range": target,
                    "shown_as": line.strip(),
                    "source_page": 1,
                }
            )

    # Peak frequency by region (typically PAGE 1).
    peak_lines = [
        ("Frontal", "frontal_peak_frequency_ec"),
        ("Central-Parietal", "central_parietal_peak_frequency_ec"),
        ("Occipital", "occipital_peak_frequency_ec"),
    ]
    for label, metric in peak_lines:
        line = find_line_startswith(label)
        if not line:
            continue

        target = None
        value_part = line
        m_range = re.search(r"(?i)(\d+(?:\.\d+)?)\s*[-–—]\s*(\d+(?:\.\d+)?)\s*Hz\b", line)
        if m_range:
            target = f"{m_range.group(1)}-{m_range.group(2)} Hz"
            value_part = line[: m_range.start()].strip()

        value_part = re.sub(rf"(?im)^{re.escape(label)}\b", "", value_part, count=1).strip()
        tokens = re.findall(r"(?i)(?:-?\d+(?:\.\d+)?|N/?A)\b", value_part)
        if len(tokens) < len(expected_sessions):
            tokens = _number_tokens(value_part)
        if len(tokens) < len(expected_sessions):
            continue

        for sess, tok in zip(expected_sessions, tokens[: len(expected_sessions)]):
            t = (tok or "").strip()
            if t.upper().replace("/", "") == "NA":
                out.append(
                    {
                        "fact_type": "peak_frequency",
                        "metric": metric,
                        "session_index": sess,
                        "value": None,
                        "unit": "Hz",
                        "target_range": target,
                        "shown_as": "N/A",
                        "source_page": 1,
                    }
                )
                continue

            val = _safe_float(t)
            if val is None:
                continue
            out.append(
                {
                    "fact_type": "peak_frequency",
                    "metric": metric,
                    "session_index": sess,
                    "value": val,
                    "unit": "Hz",
                    "target_range": target,
                    "shown_as": None,
                    "source_page": 1,
                }
            )

    return out


def _facts_from_report_text_n100_central_frontal(report_text: str, *, expected_sessions: list[int]) -> list[dict[str, Any]]:
    """
    Deterministically extract CENTRAL-FRONTAL AVERAGE N100 values from OCR text (typically PAGE 2).

    Expected OCR shape (order corresponds to sessions):
      CENTRAL-FRONTAL AVERAGE
      N100-UV MS
      36 -4.4 120
      37 -5.4 120
      36 N/A
    """
    page2 = _page_section(report_text, page_num=2)
    lines = [ln.strip() for ln in page2.splitlines() if ln.strip()]
    if not lines:
        return []

    start_idx = next(
        (i for i, ln in enumerate(lines) if re.search(r"(?i)central\s*[- ]\s*frontal\s+average", ln)),
        None,
    )
    if start_idx is None:
        return []

    n100_idx = next(
        (i for i in range(start_idx, min(len(lines), start_idx + 40)) if re.search(r"(?i)\bN100\b", lines[i])),
        None,
    )
    if n100_idx is None:
        return []

    value_lines: list[str] = []
    for ln in lines[n100_idx + 1 : n100_idx + 30]:
        if re.match(r"^\d+\b", ln):
            value_lines.append(ln)
            if len(value_lines) >= len(expected_sessions):
                break
        elif value_lines:
            break

    out: list[dict[str, Any]] = []
    for sess, ln in zip(expected_sessions, value_lines):
        m = re.match(r"^(?P<yield>\d+)\s+(?P<uv>-?\d+(?:\.\d+)?)\s+(?P<ms>\d+)\b", ln)
        if m:
            out.append(
                {
                    "fact_type": "n100_central_frontal_average",
                    "session_index": sess,
                    "yield": _safe_int(m.group("yield")),
                    "uv": _safe_float(m.group("uv")),
                    "ms": _safe_int(m.group("ms")),
                    "shown_as": None,
                    "source_page": 2,
                }
            )
            continue

        m = re.match(r"^(?P<yield>\d+)\s+(?P<na>N/?A)\b(?:\s+(?P<ms>\d+)\b)?", ln, flags=re.IGNORECASE)
        if m:
            out.append(
                {
                    "fact_type": "n100_central_frontal_average",
                    "session_index": sess,
                    "yield": _safe_int(m.group("yield")),
                    "uv": None,
                    "ms": _safe_int(m.group("ms")) if m.group("ms") else None,
                    "shown_as": "N/A",
                    "source_page": 2,
                }
            )
            continue

    cleaned: list[dict[str, Any]] = []
    for f in out:
        if f.get("shown_as") == "N/A":
            cleaned.append(f)
            continue
        if f.get("uv") is not None and f.get("ms") is not None:
            cleaned.append(f)
    return cleaned

