from __future__ import annotations

import asyncio
import base64
import json
import os
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Awaitable, Callable

from .config import ARTIFACTS_DIR, is_vision_capable
from .llm_client import AsyncOpenAICompatClient, UpstreamError
from .reports import extract_pdf_full, extract_text_from_pdf, get_page_images_base64, get_enhanced_text
from .storage import (
    Artifact,
    Report,
    Run,
    create_artifact,
    get_report,
    get_run,
    set_run_label_map,
    update_run_status,
)
from .storage import session_scope


OnEvent = Callable[[dict[str, Any]], Awaitable[None]] | None


@dataclass(frozen=True)
class StageDef:
    num: int
    name: str
    kind: str
    content_type: str
    ext: str


STAGES: list[StageDef] = [
    StageDef(1, "initial_analysis", "analysis", "text/markdown", ".md"),
    StageDef(2, "peer_review", "peer_review", "application/json", ".json"),
    StageDef(3, "revision", "revision", "text/markdown", ".md"),
    StageDef(4, "consolidation", "consolidation", "text/markdown", ".md"),
    StageDef(5, "final_review", "final_review", "application/json", ".json"),
    StageDef(6, "final_draft", "final_draft", "text/markdown", ".md"),
]

DATA_PACK_SCHEMA_VERSION = 1
DATA_PACK_FILENAME = "_data_pack.json"
VISION_TRANSCRIPT_FILENAME = "_vision_transcript.md"


@dataclass(frozen=True)
class PageImage:
    page: int
    base64_png: str
    label: str | None = None


def _try_build_p300_cp_site_crops(page_image: PageImage) -> list[PageImage]:
    """
    Best-effort helper for WAVi-style "P300 Rare Comparison" pages.

    Produces:
    - a legend/header crop (for session color mapping)
    - one crop per CP site (C3, CZ, C4, P3, PZ, P4)

    If OCR-based label localization fails, returns [].
    """
    try:
        import io

        import pytesseract
        from PIL import Image
        from pytesseract import Output
    except Exception:
        return []

    try:
        raw = base64.b64decode(page_image.base64_png)
    except Exception:
        return []

    try:
        img = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception:
        return []

    w, h = img.size

    # Legend crop: top-right area where the session color key usually appears.
    legend = img.crop(
        (
            int(w * 0.55),
            int(h * 0.06),
            int(w * 0.98),
            int(h * 0.22),
        )
    ).resize((int(w * 0.55), int(h * 0.22)), resample=Image.Resampling.LANCZOS)

    targets = {"C3", "CZ", "C4", "P3", "PZ", "P4"}
    crop_w = int(w * 0.26)
    crop_h = int(h * 0.22)

    # Upscale for better label detection.
    detect_img = img.resize((w * 2, h * 2), resample=Image.Resampling.LANCZOS)
    data = pytesseract.image_to_data(detect_img, output_type=Output.DICT, config="--psm 6")

    best: dict[str, tuple[int, int, int, int, float]] = {}
    n = len(data.get("text") or [])
    for i in range(n):
        text = (data["text"][i] or "").strip()
        if not text:
            continue
        t = text.upper().replace(":", "")
        if t not in targets:
            continue
        try:
            conf = float(data.get("conf", [0])[i])
        except Exception:
            conf = 0.0
        x = int(data["left"][i] / 2)
        y = int(data["top"][i] / 2)
        bw = int(data["width"][i] / 2)
        bh = int(data["height"][i] / 2)
        prev = best.get(t)
        if prev is None or conf > prev[4]:
            best[t] = (x, y, bw, bh, conf)

    if len(best) < 4:
        # Heuristic fallback for the common WAVi "P300 Rare Comparison" page layout.
        # Coordinates are relative to page size and cover the CP panels grid.
        def frac_box(x0: float, y0: float, x1: float, y1: float) -> tuple[int, int, int, int]:
            return (int(w * x0), int(h * y0), int(w * x1), int(h * y1))

        legend_box = frac_box(0.54, 0.04, 0.98, 0.20)
        site_boxes = {
            "C3_panel": frac_box(0.19, 0.35, 0.41, 0.56),
            "CZ_panel": frac_box(0.38, 0.35, 0.60, 0.56),
            "C4_panel": frac_box(0.57, 0.35, 0.79, 0.56),
            "P3_panel": frac_box(0.19, 0.55, 0.41, 0.77),
            "PZ_panel": frac_box(0.38, 0.55, 0.60, 0.77),
            "P4_panel": frac_box(0.57, 0.55, 0.79, 0.77),
            # Central-frontal average block near the bottom (contains N100 lines).
            "central_frontal_avg": frac_box(0.28, 0.74, 0.72, 0.92),
        }

        try:
            legend_img = img.crop(legend_box).resize(
                (max(800, legend_box[2] - legend_box[0]) * 2, max(250, legend_box[3] - legend_box[1]) * 2),
                resample=Image.Resampling.LANCZOS,
            )
        except Exception:
            legend_img = None

        crops: list[PageImage] = []
        if legend_img is not None:
            buf = io.BytesIO()
            legend_img.save(buf, format="PNG")
            crops.append(
                PageImage(
                    page=page_image.page,
                    base64_png=base64.b64encode(buf.getvalue()).decode("utf-8"),
                    label="legend",
                )
            )

        for label, box in site_boxes.items():
            try:
                crop = img.crop(box).resize(
                    (max(900, box[2] - box[0]) * 2, max(600, box[3] - box[1]) * 2),
                    resample=Image.Resampling.LANCZOS,
                )
            except Exception:
                continue
            buf = io.BytesIO()
            crop.save(buf, format="PNG")
            crops.append(
                PageImage(
                    page=page_image.page,
                    base64_png=base64.b64encode(buf.getvalue()).decode("utf-8"),
                    label=label,
                )
            )

        return crops if len(crops) >= 4 else []

    crops: list[PageImage] = []
    # Add legend/header first.
    buf = io.BytesIO()
    legend.save(buf, format="PNG")
    crops.append(PageImage(page=page_image.page, base64_png=base64.b64encode(buf.getvalue()).decode("utf-8"), label="legend"))

    for site in ["C3", "CZ", "C4", "P3", "PZ", "P4"]:
        if site not in best:
            continue
        x, y, bw, bh, _conf = best[site]
        cx = x + bw // 2
        cy = y + bh // 2
        x0 = max(0, cx - crop_w // 2)
        y0 = max(0, cy - int(crop_h * 0.15))
        x1 = min(w, x0 + crop_w)
        y1 = min(h, y0 + crop_h)
        crop = img.crop((x0, y0, x1, y1)).resize((crop_w * 2, crop_h * 2), resample=Image.Resampling.LANCZOS)
        buf = io.BytesIO()
        crop.save(buf, format="PNG")
        crops.append(
            PageImage(
                page=page_image.page,
                base64_png=base64.b64encode(buf.getvalue()).decode("utf-8"),
                label=f"{site}_panel",
            )
        )

    # Add central-frontal average block (for N100) via a stable heuristic crop.
    try:
        x0 = int(w * 0.28)
        y0 = int(h * 0.74)
        x1 = int(w * 0.72)
        y1 = int(h * 0.92)
        cf = img.crop((x0, y0, x1, y1)).resize(
            (max(900, x1 - x0) * 2, max(500, y1 - y0) * 2),
            resample=Image.Resampling.LANCZOS,
        )
        buf = io.BytesIO()
        cf.save(buf, format="PNG")
        crops.append(
            PageImage(
                page=page_image.page,
                base64_png=base64.b64encode(buf.getvalue()).decode("utf-8"),
                label="central_frontal_avg",
            )
        )
    except Exception:
        pass

    return crops


def _try_build_summary_table_crops(page_image: PageImage) -> list[PageImage]:
    """
    Best-effort helper for WAVi-style PAGE 1 summary tables.

    Produces upscaled crops for:
    - performance/evoked/state summary table block
    - peak frequency block (frontal/central-parietal/occipital)
    """
    try:
        import io

        from PIL import Image
    except Exception:
        return []

    try:
        raw = base64.b64decode(page_image.base64_png)
    except Exception:
        return []

    try:
        img = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception:
        return []

    w, h = img.size

    def crop_frac(label: str, x0: float, y0: float, x1: float, y1: float) -> PageImage | None:
        try:
            box = (int(w * x0), int(h * y0), int(w * x1), int(h * y1))
            crop = img.crop(box).resize(
                (max(900, box[2] - box[0]) * 2, max(650, box[3] - box[1]) * 2),
                resample=Image.Resampling.LANCZOS,
            )
            buf = io.BytesIO()
            crop.save(buf, format="PNG")
            return PageImage(
                page=page_image.page,
                base64_png=base64.b64encode(buf.getvalue()).decode("utf-8"),
                label=label,
            )
        except Exception:
            return None

    crops: list[PageImage] = []
    # Summary table area (performance, evoked potentials, state).
    c1 = crop_frac("summary_table", 0.05, 0.20, 0.95, 0.58)
    if c1 is not None:
        crops.append(c1)
    # Peak frequency region, just below the state metrics.
    c2 = crop_frac("peak_frequency", 0.08, 0.48, 0.92, 0.70)
    if c2 is not None:
        crops.append(c2)

    return crops


def _chunked(items: list[Any], size: int) -> list[list[Any]]:
    if size <= 0:
        return [items]
    return [items[i : i + size] for i in range(0, len(items), size)]


def _truthy_env(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    raw = raw.strip().lower()
    if raw in {"1", "true", "yes", "y", "on"}:
        return True
    if raw in {"0", "false", "no", "n", "off"}:
        return False
    return default


def _save_debug_images(*, model_dir: Path, stem: str, images: list[PageImage]) -> list[str]:
    """
    Best-effort helper to persist the exact PNGs sent to a multimodal call.

    This is intentionally "best effort": failures should never break the run.
    """
    try:
        images_dir = model_dir / f"{stem}.images"
        images_dir.mkdir(parents=True, exist_ok=True)
    except Exception:
        return []

    def safe_component(raw: str) -> str:
        cleaned = "".join(c if c.isalnum() or c in {"-", "_", "."} else "_" for c in (raw or ""))
        return cleaned.strip("_")[:80]

    saved: list[str] = []
    for idx, img in enumerate(images, start=1):
        label = safe_component(img.label or "")
        name = f"img-{idx:02d}-page-{img.page}"
        if label:
            name += f"-{label}"
        name += ".png"
        out_path = images_dir / name
        try:
            out_path.write_bytes(base64.b64decode(img.base64_png))
            saved.append(str(out_path))
        except Exception:
            continue
    return saved


def _expected_session_indices(report_text: str) -> list[int]:
    # Prefer explicit "Session N (" markers, fall back to any "Session N".
    indices = {int(m.group(1)) for m in re.finditer(r"Session\s+(\d+)\s*\(", report_text)}
    if not indices:
        indices = {int(m.group(1)) for m in re.finditer(r"\bSession\s+(\d+)\b", report_text)}
    return sorted(indices) if indices else [1]


def _derive_report_dir(report: Report) -> Path:
    report_text_path = Path(report.extracted_text_path)
    report_dir = report_text_path.parent
    if report_dir.exists():
        return report_dir
    try:
        stored_dir = Path(report.stored_path).parent
        if stored_dir.exists():
            return stored_dir
    except Exception:
        pass
    return report_text_path.parent


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


def _load_best_report_text(report: Report, report_dir: Path) -> str:
    report_text_path = Path(report.extracted_text_path)
    report_text = report_text_path.read_text(encoding="utf-8", errors="replace")

    enhanced_path = report_dir / "extracted_enhanced.txt"
    enhanced_text: str | None = None
    if enhanced_path.exists():
        enhanced_text = enhanced_path.read_text(encoding="utf-8", errors="replace")
    else:
        try:
            enhanced_text = get_enhanced_text(report.patient_id, report.id)
        except Exception:
            enhanced_text = None

    if enhanced_text and enhanced_text.strip():
        report_text = enhanced_text

    # Never re-extract on the fly here: to avoid widening the error surface in later stages, we require
    # extraction artifacts to already be present and well-formed.
    if "=== PAGE 1 /" not in report_text and Path(report.stored_path).suffix.lower() == ".pdf":
        raise RuntimeError(
            "Report text is missing page markers (expected '=== PAGE 1 /').\n"
            f"Report: {report.filename} ({report.id})\n"
            f"Paths checked: {enhanced_path} and {report_text_path}\n"
            "Fix: re-generate extraction artifacts via POST /api/reports/{report_id}/reextract"
        )

    return report_text


def _load_page_images(report: Report, report_dir: Path) -> list[PageImage]:
    pages_dir = report_dir / "pages"
    if pages_dir.exists():
        page_files = sorted(
            pages_dir.glob("page-*.png"),
            key=lambda p: int(p.stem.split("-")[1]) if "-" in p.stem and p.stem.split("-")[1].isdigit() else 0,
        )
        out: list[PageImage] = []
        for p in page_files:
            try:
                page_num = int(p.stem.split("-")[1])
            except Exception:
                continue
            try:
                out.append(PageImage(page=page_num, base64_png=base64.b64encode(p.read_bytes()).decode("utf-8")))
            except Exception:
                continue
        if out:
            return out

    # Fallback for older artifact formats: no page numbers, assume sequential.
    try:
        images = get_page_images_base64(report.patient_id, report.id)
    except Exception:
        images = []
    return [PageImage(page=i + 1, base64_png=b64) for i, b64 in enumerate(images) if isinstance(b64, str)]


def _data_pack_path(run_id: str) -> Path:
    return _stage_dir(run_id, 1) / DATA_PACK_FILENAME


def _vision_transcript_path(run_id: str) -> Path:
    return _stage_dir(run_id, 1) / VISION_TRANSCRIPT_FILENAME


def _workflow_context_block(*, stage_num: int, stage_name: str) -> str:
    extra = ""
    if stage_num == 1:
        extra = (
            "- Stage 1 must be grounded in the full report. If a PDF is >10 pages, ingestion may occur in multiple\n"
            "  multimodal passes before writing the final narrative.\n"
        )
    elif stage_num == 2:
        extra = (
            "- Stage 2 peer review must be evidence-based: evaluate each analysis against the ORIGINAL report text\n"
            "  and (when provided) the DATA PACK extracted from ALL pages.\n"
        )
    elif stage_num == 3:
        extra = (
            "- Stage 3 revision must incorporate peer review feedback while remaining faithful to the ORIGINAL report\n"
            "  text and (when provided) the DATA PACK.\n"
        )
    elif stage_num == 4:
        extra = (
            "- Stage 4 consolidation must synthesize revised analyses while keeping numeric claims aligned with the\n"
            "  ORIGINAL report text and (when provided) the DATA PACK.\n"
        )
    elif stage_num == 5:
        extra = (
            "- Stage 5 final review must vote REVISE if required tables/sections are missing or if numeric claims do\n"
            "  not match the ORIGINAL report text / DATA PACK.\n"
        )
    elif stage_num == 6:
        extra = (
            "- Stage 6 final draft must apply required changes and keep all numeric claims aligned with the ORIGINAL\n"
            "  report text / DATA PACK.\n"
        )

    return (
        "WORKFLOW CONTEXT (read carefully):\n"
        f"- You are working in Stage {stage_num}: {stage_name}.\n"
        "- A structured DATA PACK may be provided below. It is generated by processing ALL PDF pages, "
        "including graphics/tables, using multi-pass vision when needed.\n"
        "- A MULTIMODAL VISION TRANSCRIPT may also be provided. Use it to capture image-only tables/figures that OCR\n"
        "  may miss; do not claim data is missing when it is present there.\n"
        "- Treat the DATA PACK as authoritative for transcribed numeric values; do not claim data is missing "
        "when it is present in the DATA PACK or the report text.\n"
        "- Do not invent numbers. If a value is shown as N/A, report it as present but N/A.\n"
        f"{extra}"
    )


def _data_pack_prompt(*, pages: list[int], focus: str) -> str:
    pages_str = ", ".join(str(p) for p in pages) if pages else "(unknown)"
    return (
        "You are performing STRICT DATA TRANSCRIPTION from qEEG report page images.\n"
        "Return JSON ONLY (no markdown, no prose).\n\n"
        f"Pages provided in this call: {pages_str}\n"
        f"Extraction focus: {focus}\n\n"
        "Canonical identifiers (use EXACT strings):\n"
        "- performance_metric.metric: physical_reaction_time | trail_making_test_a | trail_making_test_b\n"
        "- evoked_potential.metric: audio_p300_delay | audio_p300_voltage\n"
        "- state_metric.metric: cz_theta_beta_ratio_ec | f3_f4_alpha_ratio_ec\n"
        "- peak_frequency.metric: frontal_peak_frequency_ec | central_parietal_peak_frequency_ec | occipital_peak_frequency_ec\n"
        "- p300_cp_site.site: C3 | CZ | C4 | P3 | PZ | P4\n"
        "- n100_central_frontal_average.fact_type: n100_central_frontal_average\n\n"
        "JSON schema:\n"
        "{\n"
        f'  \"schema_version\": {DATA_PACK_SCHEMA_VERSION},\n'
        "  \"pages_seen\": [1, 2],\n"
        "  \"page_inventory\": [\n"
        "    {\"page\": 1, \"title\": \"...\", \"contains\": [\"summary_table\"], \"notes\": \"...\"}\n"
        "  ],\n"
        "  \"facts\": [\n"
        "    {\n"
        "      \"fact_type\": \"performance_metric\",\n"
        "      \"metric\": \"physical_reaction_time\",\n"
        "      \"session_index\": 1,\n"
        "      \"value\": 283,\n"
        "      \"unit\": \"ms\",\n"
        "      \"sd_plus_minus\": 38,\n"
        "      \"target_range\": \"255-367 ms\",\n"
        "      \"shown_as\": null,\n"
        "      \"source_page\": 1\n"
        "    },\n"
        "    {\n"
        "      \"fact_type\": \"performance_metric\",\n"
        "      \"metric\": \"trail_making_test_a\",\n"
        "      \"session_index\": 1,\n"
        "      \"value\": 56,\n"
        "      \"unit\": \"sec\",\n"
        "      \"target_range\": \"37-63 sec\",\n"
        "      \"shown_as\": null,\n"
        "      \"source_page\": 1\n"
        "    },\n"
        "    {\n"
        "      \"fact_type\": \"evoked_potential\",\n"
        "      \"metric\": \"audio_p300_delay\",\n"
        "      \"session_index\": 1,\n"
        "      \"value\": 304,\n"
        "      \"unit\": \"ms\",\n"
        "      \"target_range\": \"247-321 ms\",\n"
        "      \"shown_as\": null,\n"
        "      \"source_page\": 1\n"
        "    },\n"
        "    {\n"
        "      \"fact_type\": \"state_metric\",\n"
        "      \"metric\": \"cz_theta_beta_ratio_ec\",\n"
        "      \"session_index\": 1,\n"
        "      \"value\": 3.1,\n"
        "      \"unit\": \"ratio\",\n"
        "      \"target_range\": \"0.9-2.3\",\n"
        "      \"shown_as\": null,\n"
        "      \"source_page\": 1\n"
        "    },\n"
        "    {\n"
        "      \"fact_type\": \"peak_frequency\",\n"
        "      \"metric\": \"frontal_peak_frequency_ec\",\n"
        "      \"session_index\": 1,\n"
        "      \"value\": 8.5,\n"
        "      \"unit\": \"Hz\",\n"
        "      \"target_range\": \"9.0-11.0 Hz\",\n"
        "      \"shown_as\": null,\n"
        "      \"source_page\": 1\n"
        "    },\n"
        "    {\n"
        "      \"fact_type\": \"p300_cp_site\",\n"
        "      \"site\": \"C3\",\n"
        "      \"session_index\": 1,\n"
        "      \"yield\": 38,\n"
        "      \"uv\": 11.9,\n"
        "      \"ms\": 308,\n"
        "      \"shown_as\": null,\n"
        "      \"source_page\": 2\n"
        "    },\n"
        "    {\n"
        "      \"fact_type\": \"n100_central_frontal_average\",\n"
        "      \"session_index\": 1,\n"
        "      \"yield\": 36,\n"
        "      \"uv\": -4.4,\n"
        "      \"ms\": 120,\n"
        "      \"shown_as\": null,\n"
        "      \"source_page\": 2\n"
        "    }\n"
        "  ],\n"
        "  \"unparsed_required\": [\n"
        "    {\"field\": \"p300_cp_site\", \"page\": 2, \"reason\": \"...\"}\n"
        "  ]\n"
        "}\n\n"
        "Rules:\n"
        "- Transcribe only what you can read confidently from the images. Do not infer or normalize.\n"
        "- Use JSON numbers (not strings) whenever possible.\n"
        "- If the report shows N/A, set the numeric field to null and set \"shown_as\" to \"N/A\".\n"
        "- Always include \"source_page\" for each fact.\n"
        "- Maintain session_index mapping exactly as shown in the legend (Session 1/2/3 color key).\n"
        "- When Peak Frequency is shown, extract it for Frontal, Central-Parietal, and Occipital for every session.\n"
        "- Critical: When the page includes P300 Rare Comparison, extract central-parietal per-site values for "
        "C3, CZ, C4, P3, PZ, P4 for EVERY session shown. Capture yield (#), uv (µV), and ms.\n"
        "- Also extract CENTRAL-FRONTAL AVERAGE N100 values (yield, µV, ms) if present on the page.\n"
        "- Ignore large coherence matrices unless they contain the required facts above.\n"
    )


def _prompt_path(name: str) -> Path:
    return Path(__file__).parent / "prompts" / name


def _load_prompt(name: str) -> str:
    path = _prompt_path(name)
    return path.read_text(encoding="utf-8")


def _stage_dir(run_id: str, stage_num: int) -> Path:
    return ARTIFACTS_DIR / run_id / f"stage-{stage_num}"


def _artifact_path(run_id: str, stage_num: int, model_id: str, ext: str) -> Path:
    # model_id is safe enough for filenames in practice; fall back to a stable replacement if needed.
    safe_model = "".join(c if c.isalnum() or c in {"-", "_", "."} else "_" for c in model_id) or "model"
    return _stage_dir(run_id, stage_num) / f"{safe_model}{ext}"


def _stable_label_map(run_id: str, model_ids: list[str]) -> dict[str, str]:
    rng = random.Random(run_id)
    ids = list(model_ids)
    rng.shuffle(ids)
    labels = [chr(ord("A") + i) for i in range(len(ids))]
    return {label: model_id for label, model_id in zip(labels, ids)}


def _strip_to_json(text: str) -> str:
    s = text.strip()
    if not s:
        return s
    if s[0] in "{[" and s[-1] in "}]":
        return s
    # Try to extract the outermost JSON-looking object/array.
    starts = [s.find("{"), s.find("[")]
    first = min((i for i in starts if i != -1), default=-1)
    if first == -1:
        return s
    last_curly = s.rfind("}")
    last_square = s.rfind("]")
    last = max(last_curly, last_square)
    if last <= first:
        return s
    return s[first : last + 1].strip()


def _json_loads_loose(text: str) -> Any:
    return json.loads(_strip_to_json(text))


async def _sleep_backoff(attempt: int) -> None:
    # Exponential backoff with jitter; attempt starts at 0.
    base = min(8.0, 0.5 * (2**attempt))
    jitter = random.random() * 0.2
    await asyncio.sleep(base + jitter)


class QEEGCouncilWorkflow:
    def __init__(self, *, llm: AsyncOpenAICompatClient):
        self._llm = llm

    @staticmethod
    def _dedupe_facts(facts: list[dict[str, Any]]) -> list[dict[str, Any]]:
        deduped: list[dict[str, Any]] = []
        seen: set[str] = set()
        for fact in facts:
            ftype = fact.get("fact_type")
            if not isinstance(ftype, str):
                continue
            key_parts = [ftype]
            for k in ("metric", "site", "region", "session_index"):
                v = fact.get(k)
                if v is None:
                    continue
                key_parts.append(f"{k}={v}")
            key_parts.append(f"page={fact.get('source_page')}")
            key = "|".join(key_parts)
            if key in seen:
                continue
            seen.add(key)
            deduped.append(fact)
        return deduped

    @staticmethod
    def _fact_key(fact: dict[str, Any]) -> str | None:
        ftype = fact.get("fact_type")
        if not isinstance(ftype, str):
            return None
        key_parts = [ftype]
        for k in ("metric", "site", "region", "session_index"):
            v = fact.get(k)
            if v is None:
                continue
            key_parts.append(f"{k}={v}")
        key_parts.append(f"page={fact.get('source_page')}")
        return "|".join(key_parts)

    @staticmethod
    def _fact_numeric_signature(fact: dict[str, Any]) -> tuple[Any, ...] | None:
        """
        Return a tuple representing the numeric payload of a fact for conflict detection.

        We intentionally ignore incidental fields (e.g., labels, units, target ranges) and focus on required numbers.
        """
        def coerce(val: Any) -> Any:
            if isinstance(val, bool):
                return val
            if isinstance(val, (int, float)):
                if isinstance(val, float) and val.is_integer():
                    return int(val)
                return val
            if isinstance(val, str):
                s = val.strip().replace("−", "-")
                if re.fullmatch(r"-?\d+(?:\.\d+)?", s):
                    try:
                        num = float(s)
                    except Exception:
                        return val
                    return int(num) if num.is_integer() else num
            return val

        shown_as = fact.get("shown_as")
        if isinstance(shown_as, str) and shown_as.strip().upper() == "N/A":
            return ("NA",)
        ftype = fact.get("fact_type")
        if ftype in {"performance_metric", "evoked_potential", "state_metric", "peak_frequency"}:
            val = coerce(fact.get("value"))
            if val is None:
                return None
            return ("value", val)
        if ftype == "p300_cp_site":
            uv = coerce(fact.get("uv"))
            ms = coerce(fact.get("ms"))
            if uv is None or ms is None:
                return None
            return ("uv_ms", uv, ms)
        if ftype == "n100_central_frontal_average":
            uv = coerce(fact.get("uv"))
            ms = coerce(fact.get("ms"))
            if uv is None or ms is None:
                return None
            return ("uv_ms", uv, ms)
        # Default: best-effort, include common numeric fields if present.
        payload: list[Any] = ["generic"]
        for k in ("value", "uv", "ms", "yield"):
            if k in fact:
                payload.append(coerce(fact.get(k)))
        return tuple(payload)

    @classmethod
    def _find_fact_conflicts(cls, facts: list[dict[str, Any]]) -> list[dict[str, Any]]:
        by_key: dict[str, list[dict[str, Any]]] = {}
        for f in facts:
            if not isinstance(f, dict):
                continue
            key = cls._fact_key(f)
            if not key:
                continue
            by_key.setdefault(key, []).append(f)

        conflicts: list[dict[str, Any]] = []
        for key, items in by_key.items():
            if len(items) < 2:
                continue
            sigs: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
            for f in items:
                sig = cls._fact_numeric_signature(f)
                if sig is None:
                    continue
                sigs.setdefault(sig, []).append(f)
            if len(sigs) > 1:
                conflicts.append(
                    {
                        "key": key,
                        "variants": [
                            {
                                "signature": list(sig),
                                "facts": group,
                            }
                            for sig, group in sigs.items()
                        ],
                    }
                )
        return conflicts

    async def _ensure_data_pack(
        self,
        *,
        run_id: str,
        report: Report,
        report_text: str,
        page_images: list[PageImage],
        candidate_extractor_model_ids: list[str],
        strict: bool,
    ) -> dict[str, Any] | None:
        """
        Build (or load) a structured data pack that makes image-only metrics available to ALL stages.

        Notes:
        - This runs multi-pass multimodal extraction across ALL available page images.
        - If strict is True, missing required numeric fields raises an error (no silent degradation).
        """
        out_path = _data_pack_path(run_id)
        expected_sessions = _expected_session_indices(report_text)
        expected_pages = sorted({img.page for img in page_images if isinstance(img.page, int)})

        # If an existing data pack is present, upgrade it in-place with deterministic facts and derived views.
        if out_path.exists():
            try:
                existing = json.loads(out_path.read_text(encoding="utf-8"))
            except Exception:
                existing = None

            if isinstance(existing, dict) and existing.get("schema_version") == DATA_PACK_SCHEMA_VERSION:
                # If the report was re-extracted or page images changed, ensure the cached data pack actually
                # covers the current pages (older runs could have partial page coverage).
                try:
                    existing_pages = existing.get("pages_processed") or existing.get("pages_seen") or []
                    existing_pages_set = {p for p in existing_pages if isinstance(p, int)}
                except Exception:
                    existing_pages_set = set()
                existing_page_count = existing.get("page_count")
                if expected_pages and (
                    existing_page_count != len(expected_pages) or not existing_pages_set.issuperset(set(expected_pages))
                ):
                    existing = None

            if isinstance(existing, dict) and existing.get("schema_version") == DATA_PACK_SCHEMA_VERSION:
                facts = existing.get("facts")
                if isinstance(facts, list):
                    base_facts = [f for f in facts if isinstance(f, dict)]
                    add_facts: list[dict[str, Any]] = []
                    add_facts.extend(_facts_from_report_text_summary(report_text, expected_sessions=expected_sessions))
                    add_facts.extend(_facts_from_report_text_n100_central_frontal(report_text, expected_sessions=expected_sessions))
                    for f in add_facts:
                        if isinstance(f, dict):
                            f.setdefault("extraction_method", "deterministic_report_text")
                    for f in base_facts:
                        if isinstance(f, dict):
                            f.setdefault("extraction_method", f.get("extraction_method") or "vision_llm")
                    conflicts = self._find_fact_conflicts(add_facts + base_facts)
                    if conflicts and strict:
                        # Cached pack is inconsistent; rebuild from scratch so strict runs are never silently
                        # grounded in conflicting numbers.
                        existing = None
                    else:
                        # Deterministic facts come first so they override duplicates from older model output.
                        existing["facts"] = self._dedupe_facts(add_facts + base_facts)

            if isinstance(existing, dict) and existing.get("schema_version") == DATA_PACK_SCHEMA_VERSION:
                existing["derived"] = self._derive_data_pack_views(existing, expected_sessions=expected_sessions)
                missing = self._missing_required_fields(existing, expected_sessions=expected_sessions)
                if not (missing and strict):
                    try:
                        out_path.parent.mkdir(parents=True, exist_ok=True)
                        out_path.write_text(json.dumps(existing, indent=2, sort_keys=True), encoding="utf-8")
                    except Exception:
                        pass
                    return existing

        if not page_images or not candidate_extractor_model_ids:
            if strict:
                report_dir = _derive_report_dir(report)
                raise RuntimeError(
                    "Strict data availability requested, but Stage 1 multimodal extraction cannot run.\n"
                    f"Report: {report.filename} ({report.id})\n"
                    f"Vision-capable extractor models selected: {candidate_extractor_model_ids}\n"
                    f"Page images available: {len(page_images)}\n"
                    f"Expected page images directory: {report_dir / 'pages'}\n"
                    "If report assets look incomplete, re-run: POST /api/reports/{report_id}/reextract"
                )
            return None

        chunk_size = int(os.getenv("QEEG_VISION_PAGES_PER_CALL", "8") or "8")
        if chunk_size <= 0:
            chunk_size = 8
        # Hard requirement: PDFs >10 pages must be processed in 2+ multimodal passes.
        if len(page_images) > 10 and chunk_size > 10:
            chunk_size = 10

        debug_dir: Path | None = None
        attempt_log: list[dict[str, Any]] = []
        if strict or _truthy_env("QEEG_SAVE_DATA_PACK_DEBUG", False):
            try:
                debug_dir = _stage_dir(run_id, 1) / "_data_pack_debug"
                debug_dir.mkdir(parents=True, exist_ok=True)
            except Exception:
                debug_dir = None

        def apply_deterministic_facts(merged: dict[str, Any]) -> list[dict[str, Any]]:
            merged_facts = merged.get("facts")
            if not isinstance(merged_facts, list):
                return []
            model_facts: list[dict[str, Any]] = [f for f in merged_facts if isinstance(f, dict)]
            for f in model_facts:
                f.setdefault("extraction_method", "vision_llm")

            add_facts: list[dict[str, Any]] = []
            add_facts.extend(_facts_from_report_text_summary(report_text, expected_sessions=expected_sessions))
            add_facts.extend(_facts_from_report_text_n100_central_frontal(report_text, expected_sessions=expected_sessions))
            for f in add_facts:
                if isinstance(f, dict):
                    f.setdefault("extraction_method", "deterministic_report_text")

            combined = add_facts + model_facts
            if combined:
                # Deterministic facts come first so they override duplicates from model output.
                merged["facts"] = self._dedupe_facts(combined)
            return self._find_fact_conflicts(combined)

        # Multi-pass extraction across all pages.
        errors: list[str] = []
        for extractor_model_id in candidate_extractor_model_ids:
            try:
                parts: list[dict[str, Any]] = []
                for chunk_index, chunk in enumerate(_chunked(page_images, chunk_size), start=1):
                    pages = [img.page for img in chunk]
                    prompt_text = _data_pack_prompt(
                        pages=pages,
                        focus=(
                            "Extract required numeric facts (performance, P300 delay/voltage, CP per-site P300, N100, "
                            "theta/beta, alpha ratio, peak frequency) from ONLY these pages."
                        ),
                    )
                    raw = await self._call_model_multimodal(
                        model_id=extractor_model_id,
                        prompt_text=prompt_text,
                        images=chunk,
                        temperature=0.0,
                        max_tokens=3500,
                        allow_text_fallback=False,
                    )
                    if debug_dir is not None:
                        try:
                            safe_model = "".join(
                                c if c.isalnum() or c in {"-", "_", "."} else "_" for c in extractor_model_id
                            ) or "model"
                            model_dir = debug_dir / safe_model
                            model_dir.mkdir(parents=True, exist_ok=True)
                            pages_key = "-".join(str(p) for p in pages)
                            stem = f"pass-{chunk_index:02d}-pages-{pages_key}"
                            (model_dir / f"{stem}.prompt.txt").write_text(prompt_text, encoding="utf-8")
                            (model_dir / f"{stem}.raw.txt").write_text(raw, encoding="utf-8")
                            image_paths = _save_debug_images(model_dir=model_dir, stem=stem, images=chunk)
                            attempt_log.append(
                                {
                                    "kind": "chunk",
                                    "model_id": extractor_model_id,
                                    "chunk_index": chunk_index,
                                    "pages": pages,
                                    "raw_path": str(model_dir / f"{stem}.raw.txt"),
                                    "image_paths": image_paths,
                                }
                            )
                        except Exception:
                            pass
                    part = _json_loads_loose(raw)
                    if not isinstance(part, dict):
                        raise ValueError("Data pack part must be a JSON object")
                    if part.get("schema_version") != DATA_PACK_SCHEMA_VERSION:
                        raise ValueError("Data pack schema_version mismatch")
                    if debug_dir is not None:
                        try:
                            safe_model = "".join(
                                c if c.isalnum() or c in {"-", "_", "."} else "_" for c in extractor_model_id
                            ) or "model"
                            model_dir = debug_dir / safe_model
                            pages_key = "-".join(str(p) for p in pages)
                            stem = f"pass-{chunk_index:02d}-pages-{pages_key}"
                            (model_dir / f"{stem}.parsed.json").write_text(
                                json.dumps(part, indent=2, sort_keys=True), encoding="utf-8"
                            )
                        except Exception:
                            pass
                    parts.append(part)

                merged = self._merge_data_pack_parts(
                    parts,
                    meta={
                        "schema_version": DATA_PACK_SCHEMA_VERSION,
                        "run_id": run_id,
                        "report_id": report.id,
                        "report_filename": report.filename,
                        "extraction_model_id": extractor_model_id,
                        "expected_session_indices": expected_sessions,
                        "page_count": len(page_images),
                        "pages_processed": [img.page for img in page_images],
                        "pages_per_call": chunk_size,
                    },
                )

                # Fill in deterministic facts from OCR/PDF text (page-grounded markers) to reduce flakiness.
                fact_conflicts = apply_deterministic_facts(merged)

                merged["derived"] = self._derive_data_pack_views(merged, expected_sessions=expected_sessions)

                missing = self._missing_required_fields(merged, expected_sessions=expected_sessions)
                if missing:
                    # Targeted retry for the most common failure mode: CP per-site P300 values.
                    candidate_pages = _find_p300_rare_comparison_pages(report_text, page_count=len(page_images))
                    merged = await self._targeted_retry_missing(
                        extractor_model_id=extractor_model_id,
                        page_images=page_images,
                        merged=merged,
                        expected_sessions=expected_sessions,
                        missing=missing,
                        candidate_pages=candidate_pages,
                        debug_dir=debug_dir,
                        attempt_log=attempt_log,
                    )
                    fact_conflicts = apply_deterministic_facts(merged)
                    merged["derived"] = self._derive_data_pack_views(merged, expected_sessions=expected_sessions)
                    missing = self._missing_required_fields(merged, expected_sessions=expected_sessions)

                if missing:
                    merged = await self._targeted_retry_summary_missing(
                        extractor_model_id=extractor_model_id,
                        page_images=page_images,
                        merged=merged,
                        expected_sessions=expected_sessions,
                        missing=missing,
                        debug_dir=debug_dir,
                        attempt_log=attempt_log,
                    )
                    fact_conflicts = apply_deterministic_facts(merged)
                    merged["derived"] = self._derive_data_pack_views(merged, expected_sessions=expected_sessions)
                    missing = self._missing_required_fields(merged, expected_sessions=expected_sessions)

                if strict and (missing or fact_conflicts):
                    failure_path = _stage_dir(run_id, 1) / "_data_pack_failure.json"
                    try:
                        failure_payload = {
                            "run_id": run_id,
                            "report_id": report.id,
                            "report_filename": report.filename,
                            "extraction_model_id": extractor_model_id,
                            "expected_session_indices": expected_sessions,
                            "pages_processed": [img.page for img in page_images],
                            "pages_per_call": chunk_size,
                            "missing_fields": sorted(missing),
                            "conflicts": fact_conflicts,
                            "debug_dir": str(debug_dir) if debug_dir is not None else None,
                            "attempt_log": attempt_log,
                            "partial_data_pack": merged,
                        }
                        failure_path.write_text(json.dumps(failure_payload, indent=2, sort_keys=True), encoding="utf-8")
                    except Exception:
                        pass

                    chunk_pages = [pages for pages in _chunked([img.page for img in page_images], chunk_size)]
                    retry_kinds = sorted(
                        {
                            a.get("kind")
                            for a in attempt_log
                            if isinstance(a, dict) and isinstance(a.get("kind"), str) and a.get("kind") != "chunk"
                        }
                    )
                    msg_lines = [
                        "Required data could not be extracted from the PDF images.",
                        f"Missing fields: {', '.join(sorted(missing)) if missing else '(none)'}",
                        f"Conflicts: {len(fact_conflicts)}" if fact_conflicts else "Conflicts: (none)",
                        f"Expected sessions: {expected_sessions}",
                        f"Pages processed: {[img.page for img in page_images]}",
                        f"Multimodal passes (pages_per_call={chunk_size}): {chunk_pages}",
                        f"Targeted retries: {', '.join(retry_kinds) if retry_kinds else 'none'}",
                        f"Extractor model: {extractor_model_id}",
                        f"Failure artifact: {failure_path}",
                    ]
                    if debug_dir is not None:
                        msg_lines.append(f"Debug artifacts: {debug_dir}")
                    msg_lines.append("If report assets look incomplete, re-run: POST /api/reports/{report_id}/reextract")
                    raise RuntimeError("\n".join(msg_lines))

                out_path.parent.mkdir(parents=True, exist_ok=True)
                out_path.write_text(json.dumps(merged, indent=2, sort_keys=True), encoding="utf-8")

                # Store as an artifact for traceability/debugging.
                with session_scope() as session:
                    create_artifact(
                        session,
                        run_id=run_id,
                        stage_num=1,
                        stage_name=STAGES[0].name,
                        model_id="_data_pack",
                        kind="data_pack",
                        content_path=out_path,
                        content_type="application/json",
                    )

                return merged
            except Exception as e:
                errors.append(f"{extractor_model_id}: {e}")
                continue

        if strict:
            msg = "All extractor models failed to build a data pack."
            if errors:
                msg = f"{msg}\n\nAttempts:\n- " + "\n- ".join(errors)
            raise RuntimeError(msg)

        return None

    async def _ensure_vision_transcript(
        self,
        *,
        run_id: str,
        report: Report,
        page_images: list[PageImage],
        transcript_model_id: str | None,
        strict: bool,
    ) -> str | None:
        """
        Build a run-level multimodal transcript so later stages (2-6) can access image-only tables/figures.

        This is intentionally separate from the structured data pack:
        - Data pack is strict + normalized for required metrics.
        - Vision transcript is broad + best-effort for everything visible on the pages.
        """
        out_path = _vision_transcript_path(run_id)
        if out_path.exists():
            try:
                existing = out_path.read_text(encoding="utf-8", errors="replace")
                if existing.strip():
                    # Ensure cached transcripts cover all current pages (older runs could have partial coverage).
                    expected_pages = sorted({img.page for img in page_images if isinstance(img.page, int)})
                    found_pages = {
                        int(m.group(1))
                        for m in re.finditer(r"(?m)^##\\s*Page\\s+(\\d+)\\b", existing)
                        if m.group(1).isdigit()
                    }
                    if not expected_pages or found_pages.issuperset(set(expected_pages)):
                        return existing
            except Exception:
                pass

        if not page_images or not transcript_model_id:
            if strict:
                report_dir = _derive_report_dir(report)
                raise RuntimeError(
                    "Strict data availability requested, but no vision transcript can be generated.\n"
                    f"Report: {report.filename} ({report.id})\n"
                    f"Transcript model: {transcript_model_id}\n"
                    f"Page images available: {len(page_images)}\n"
                    f"Expected page images directory: {report_dir / 'pages'}\n"
                    "If report assets look incomplete, re-run: POST /api/reports/{report_id}/reextract"
                )
            return None

        chunk_size = int(os.getenv("QEEG_VISION_TRANSCRIPT_PAGES_PER_CALL", "2") or "2")
        if chunk_size <= 0:
            chunk_size = 2
        # Hard requirement: PDFs >10 pages must be processed in 2+ multimodal passes.
        if len(page_images) > 10 and chunk_size > 10:
            chunk_size = 10

        max_tokens = int(os.getenv("QEEG_VISION_TRANSCRIPT_MAX_TOKENS", "4000") or "4000")
        if max_tokens <= 0:
            max_tokens = 4000

        parts: list[str] = []
        errors: list[str] = []
        chunks = _chunked(page_images, chunk_size)
        for chunk_index, chunk in enumerate(chunks, start=1):
            pages = [img.page for img in chunk]
            pages_str = ", ".join(str(p) for p in pages) if pages else "(unknown)"
            prompt_text = (
                "You are performing LOSSLESS MULTIMODAL TRANSCRIPTION from qEEG report page images.\n\n"
                "Goal: Ensure ALL information embedded in page graphics/tables is available downstream as text.\n\n"
                "Output format: Markdown.\n\n"
                "Rules:\n"
                "- Do NOT interpret or summarize. Do NOT diagnose.\n"
                "- Do NOT invent numbers. Transcribe only what is visible.\n"
                "- For each page, start a section with: \"## Page <n>\".\n"
                "- Under each page, transcribe every table/figure caption and any clearly printed numeric values.\n"
                "- When a page contains a table, reproduce it as a markdown table (keep row/column labels).\n"
                "- Preserve units, target ranges, N/A markings, and any symbols (e.g., *, #) when shown.\n"
                "- If a color key maps sessions to colors, transcribe the key explicitly.\n\n"
                f"Pages in this pass: {pages_str}\n"
                f"Pass {chunk_index} of {len(chunks)}\n"
            )
            try:
                text = await self._call_model_multimodal(
                    model_id=transcript_model_id,
                    prompt_text=prompt_text,
                    images=chunk,
                    temperature=0.0,
                    max_tokens=max_tokens,
                    allow_text_fallback=not strict,
                )
                if isinstance(text, str) and text.strip():
                    parts.append(text.strip())
            except Exception as e:
                errors.append(f"pass {chunk_index} pages {pages_str}: {e}")
                if strict:
                    raise RuntimeError(
                        "Strict data availability requested, but multimodal vision transcript generation failed.\n"
                        f"Model: {transcript_model_id}\n"
                        f"Failed pass: {chunk_index} / {len(chunks)}\n"
                        f"Pages: {pages_str}\n"
                        f"Output path: {out_path}\n"
                        "If report assets look incomplete, re-run: POST /api/reports/{report_id}/reextract"
                    ) from e
                continue

        transcript_body = "\n\n".join(parts).strip()
        if not transcript_body:
            if strict:
                msg = "Strict data availability requested, but vision transcript output was empty."
                if errors:
                    msg = f"{msg}\n\nAttempts:\n- " + "\n- ".join(errors)
                raise RuntimeError(msg)
            return None

        header = (
            "# Multimodal Vision Transcript\n"
            f"- run_id: {run_id}\n"
            f"- report: {report.filename} ({report.id})\n"
            f"- model_id: {transcript_model_id}\n"
            f"- pages_per_call: {chunk_size}\n"
            f"- page_count: {len(page_images)}\n\n"
        )
        transcript = f"{header}{transcript_body}\n"

        try:
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(transcript, encoding="utf-8")
        except Exception:
            pass

        try:
            with session_scope() as session:
                create_artifact(
                    session,
                    run_id=run_id,
                    stage_num=1,
                    stage_name=STAGES[0].name,
                    model_id="_vision_transcript",
                    kind="vision_transcript",
                    content_path=out_path,
                    content_type="text/markdown",
                )
        except Exception:
            pass

        return transcript

    @staticmethod
    def _merge_data_pack_parts(parts: list[dict[str, Any]], meta: dict[str, Any]) -> dict[str, Any]:
        pages_seen: set[int] = set()
        page_inventory: list[dict[str, Any]] = []
        facts: list[dict[str, Any]] = []
        unparsed_required: list[dict[str, Any]] = []

        for part in parts:
            for p in part.get("pages_seen", []) or []:
                if isinstance(p, int):
                    pages_seen.add(p)
            inv = part.get("page_inventory")
            if isinstance(inv, list):
                page_inventory.extend([x for x in inv if isinstance(x, dict)])
            f = part.get("facts")
            if isinstance(f, list):
                facts.extend([x for x in f if isinstance(x, dict)])
            ur = part.get("unparsed_required")
            if isinstance(ur, list):
                unparsed_required.extend([x for x in ur if isinstance(x, dict)])

        deduped = QEEGCouncilWorkflow._dedupe_facts(facts)

        return {
            **meta,
            "pages_seen": sorted(pages_seen),
            "page_inventory": page_inventory,
            "facts": deduped,
            "unparsed_required": unparsed_required,
        }

    @staticmethod
    def _derive_data_pack_views(data_pack: dict[str, Any], *, expected_sessions: list[int]) -> dict[str, Any]:
        facts = data_pack.get("facts")
        if not isinstance(facts, list):
            return {}

        def fmt_num(val: Any) -> str:
            if isinstance(val, int):
                return str(val)
            if isinstance(val, float):
                # Keep small clinical values readable without over-rounding.
                return f"{val:.2f}".rstrip("0").rstrip(".")
            return ""

        def shown_as_or_na(f: dict[str, Any]) -> str | None:
            shown_as = f.get("shown_as")
            if isinstance(shown_as, str) and shown_as.strip():
                return shown_as.strip()
            return None

        def get_fact(*, fact_type: str, metric: str, session_index: int) -> dict[str, Any] | None:
            for f in facts:
                if not isinstance(f, dict):
                    continue
                if f.get("fact_type") != fact_type:
                    continue
                if f.get("metric") != metric:
                    continue
                if f.get("session_index") != session_index:
                    continue
                return f
            return None

        def short_shown_value(f: dict[str, Any]) -> str | None:
            shown = shown_as_or_na(f)
            if not shown:
                return None
            if shown.strip().upper() == "N/A":
                return "N/A"
            if re.match(r"^[<>≥≤]?\s*-?\d+(?:\.\d+)?\s*(?:ms|µV|uV|Hz|sec|s|ratio)?\s*$", shown):
                return shown
            return None

        # Summary tables (PAGE 1 style).
        perf_specs = [
            ("Physical Reaction Time", "physical_reaction_time"),
            ("Trail Making Test A", "trail_making_test_a"),
            ("Trail Making Test B", "trail_making_test_b"),
        ]
        perf_headers = ["Metric"] + [f"Session {i}" for i in expected_sessions] + ["Target"]
        perf_rows = ["| " + " | ".join(perf_headers) + " |", "|" + "|".join(["---"] * len(perf_headers)) + "|"]
        for label, metric in perf_specs:
            target = ""
            cells: list[str] = [label]
            for sess in expected_sessions:
                f = get_fact(fact_type="performance_metric", metric=metric, session_index=sess)
                if not f:
                    cells.append("MISSING")
                    continue
                if not target and isinstance(f.get("target_range"), str):
                    target = f["target_range"]
                shown = short_shown_value(f)
                if shown == "N/A":
                    cells.append("N/A")
                    continue
                val = f.get("value")
                if val is None:
                    cells.append(shown_as_or_na(f) or "N/A")
                    continue
                unit = f.get("unit") if isinstance(f.get("unit"), str) else ""
                sd = f.get("sd_plus_minus")
                if metric == "physical_reaction_time" and isinstance(sd, (int, float)):
                    cells.append(f"{fmt_num(val)} ±{fmt_num(sd)} {unit}".strip())
                else:
                    cells.append(f"{fmt_num(val)} {unit}".strip())
            cells.append(target)
            perf_rows.append("| " + " | ".join(cells) + " |")
        performance_table = "\n".join(perf_rows)

        evoked_specs = [
            ("Audio P300 Delay", "audio_p300_delay"),
            ("Audio P300 Voltage", "audio_p300_voltage"),
        ]
        evoked_headers = ["Metric"] + [f"Session {i}" for i in expected_sessions] + ["Target"]
        evoked_rows = ["| " + " | ".join(evoked_headers) + " |", "|" + "|".join(["---"] * len(evoked_headers)) + "|"]
        for label, metric in evoked_specs:
            target = ""
            cells = [label]
            for sess in expected_sessions:
                f = get_fact(fact_type="evoked_potential", metric=metric, session_index=sess)
                if not f:
                    cells.append("MISSING")
                    continue
                if not target and isinstance(f.get("target_range"), str):
                    target = f["target_range"]
                shown = short_shown_value(f)
                if shown == "N/A":
                    cells.append("N/A")
                    continue
                val = f.get("value")
                if val is None:
                    cells.append(shown_as_or_na(f) or "N/A")
                    continue
                unit = f.get("unit") if isinstance(f.get("unit"), str) else ""
                cells.append(f"{fmt_num(val)} {unit}".strip())
            cells.append(target)
            evoked_rows.append("| " + " | ".join(cells) + " |")
        evoked_table = "\n".join(evoked_rows)

        state_specs = [
            ("CZ Eyes Closed Theta/Beta (Power)", "cz_theta_beta_ratio_ec"),
            ("F3/F4 Eyes Closed Alpha (Power)", "f3_f4_alpha_ratio_ec"),
        ]
        state_headers = ["Metric"] + [f"Session {i}" for i in expected_sessions] + ["Target"]
        state_rows = ["| " + " | ".join(state_headers) + " |", "|" + "|".join(["---"] * len(state_headers)) + "|"]
        for label, metric in state_specs:
            target = ""
            cells = [label]
            for sess in expected_sessions:
                f = get_fact(fact_type="state_metric", metric=metric, session_index=sess)
                if not f:
                    cells.append("MISSING")
                    continue
                if not target and isinstance(f.get("target_range"), str):
                    target = f["target_range"]
                shown = short_shown_value(f)
                if shown:
                    cells.append(shown)
                    continue
                val = f.get("value")
                cells.append(fmt_num(val) if val is not None else (shown_as_or_na(f) or "N/A"))
            cells.append(target)
            state_rows.append("| " + " | ".join(cells) + " |")
        state_table = "\n".join(state_rows)

        peak_specs = [
            ("Frontal", "frontal_peak_frequency_ec"),
            ("Central-Parietal", "central_parietal_peak_frequency_ec"),
            ("Occipital", "occipital_peak_frequency_ec"),
        ]
        peak_headers = ["Region"] + [f"Session {i}" for i in expected_sessions] + ["Target"]
        peak_rows = ["| " + " | ".join(peak_headers) + " |", "|" + "|".join(["---"] * len(peak_headers)) + "|"]
        for label, metric in peak_specs:
            target = ""
            cells = [label]
            for sess in expected_sessions:
                f = get_fact(fact_type="peak_frequency", metric=metric, session_index=sess)
                if not f:
                    cells.append("MISSING")
                    continue
                if not target and isinstance(f.get("target_range"), str):
                    target = f["target_range"]
                shown = short_shown_value(f)
                if shown == "N/A":
                    cells.append("N/A")
                    continue
                val = f.get("value")
                if val is None:
                    cells.append(shown_as_or_na(f) or "N/A")
                    continue
                unit = f.get("unit") if isinstance(f.get("unit"), str) else ""
                cells.append(f"{fmt_num(val)} {unit}".strip())
            cells.append(target)
            peak_rows.append("| " + " | ".join(cells) + " |")
        peak_table = "\n".join(peak_rows)

        # CP per-site P300 table (C3, CZ, C4, P3, PZ, P4).
        cp_sites = ["C3", "CZ", "C4", "P3", "PZ", "P4"]
        cp_map: dict[str, dict[int, dict[str, Any]]] = {s: {} for s in cp_sites}
        for f in facts:
            if f.get("fact_type") != "p300_cp_site":
                continue
            site = f.get("site")
            sess = f.get("session_index")
            if site not in cp_map or not isinstance(sess, int):
                continue
            cp_map[site][sess] = f

        headers = ["Site"] + [f"Session {i} (µV / ms)" for i in expected_sessions]
        rows = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
        for site in cp_sites:
            cells: list[str] = [site]
            for sess in expected_sessions:
                f = cp_map.get(site, {}).get(sess)
                if not f:
                    cells.append("MISSING")
                    continue
                uv = f.get("uv")
                ms = f.get("ms")
                if uv is None or ms is None:
                    cells.append(shown_as_or_na(f) or "N/A")
                else:
                    cells.append(f"{fmt_num(uv)} / {fmt_num(ms)}")
            rows.append("| " + " | ".join(cells) + " |")
        p300_cp_table = "\n".join(rows)

        # CENTRAL-FRONTAL AVERAGE N100 table.
        n100_by_sess: dict[int, dict[str, Any]] = {}
        for f in facts:
            if f.get("fact_type") != "n100_central_frontal_average":
                continue
            sess = f.get("session_index")
            if not isinstance(sess, int):
                continue
            n100_by_sess[sess] = f

        n100_headers = ["Metric"] + [f"Session {i}" for i in expected_sessions]
        n100_rows = ["| " + " | ".join(n100_headers) + " |", "|" + "|".join(["---"] * len(n100_headers)) + "|"]
        cells = ["Central-frontal N100 (yield, µV, ms)"]
        for sess in expected_sessions:
            f = n100_by_sess.get(sess)
            if not f:
                cells.append("MISSING")
                continue
            shown = shown_as_or_na(f)
            if shown and shown.strip().upper() == "N/A":
                yv = f.get("yield")
                cells.append(f"{fmt_num(yv)} / N/A" if yv is not None else "N/A")
                continue
            yv = f.get("yield")
            uv = f.get("uv")
            ms = f.get("ms")
            if uv is None or ms is None:
                cells.append(shown or "N/A")
            else:
                ytxt = fmt_num(yv) if yv is not None else ""
                # Keep a consistent "yield / µV / ms" shape.
                cells.append(f"{ytxt} / {fmt_num(uv)} / {fmt_num(ms)}".strip(" /"))
        n100_rows.append("| " + " | ".join(cells) + " |")
        n100_table = "\n".join(n100_rows)

        return {
            "summary_performance_table_markdown": performance_table,
            "summary_evoked_table_markdown": evoked_table,
            "summary_state_table_markdown": state_table,
            "peak_frequency_table_markdown": peak_table,
            "p300_cp_table_markdown": p300_cp_table,
            "n100_central_frontal_table_markdown": n100_table,
        }

    @staticmethod
    def _missing_required_fields(data_pack: dict[str, Any], *, expected_sessions: list[int]) -> set[str]:
        facts = data_pack.get("facts")
        if not isinstance(facts, list):
            return {"facts"}

        missing: set[str] = set()

        def has_values_or_na(f: dict[str, Any], *, fields: tuple[str, ...]) -> bool:
            shown_as = f.get("shown_as")
            if isinstance(shown_as, str) and shown_as.strip().upper() == "N/A":
                return True
            return all(f.get(field) is not None for field in fields)

        def has_fact(predicate) -> bool:
            for f in facts:
                if not isinstance(f, dict):
                    continue
                if predicate(f):
                    return True
            return False

        # Performance metrics (must exist for each expected session).
        perf_metrics = [
            ("performance_metric", "physical_reaction_time"),
            ("performance_metric", "trail_making_test_a"),
            ("performance_metric", "trail_making_test_b"),
        ]
        for _, metric in perf_metrics:
            for sess in expected_sessions:
                if not has_fact(
                    lambda f, m=metric, s=sess: f.get("fact_type") == "performance_metric"
                    and f.get("metric") == m
                    and f.get("session_index") == s
                    and has_values_or_na(f, fields=("value",))
                ):
                    missing.add(f"performance:{metric}:session_{sess}")

        # Summary P300 delay/voltage.
        for metric in ("audio_p300_delay", "audio_p300_voltage"):
            for sess in expected_sessions:
                if not has_fact(
                    lambda f, m=metric, s=sess: f.get("fact_type") == "evoked_potential"
                    and f.get("metric") == m
                    and f.get("session_index") == s
                    and has_values_or_na(f, fields=("value",))
                ):
                    missing.add(f"evoked:{metric}:session_{sess}")

        # Background EEG summary metrics.
        for metric in ("cz_theta_beta_ratio_ec", "f3_f4_alpha_ratio_ec"):
            for sess in expected_sessions:
                if not has_fact(
                    lambda f, m=metric, s=sess: f.get("fact_type") == "state_metric"
                    and f.get("metric") == m
                    and f.get("session_index") == s
                    and has_values_or_na(f, fields=("value",))
                ):
                    missing.add(f"state:{metric}:session_{sess}")

        # Peak frequency by region (Frontal, Central-Parietal, Occipital).
        for metric in (
            "frontal_peak_frequency_ec",
            "central_parietal_peak_frequency_ec",
            "occipital_peak_frequency_ec",
        ):
            for sess in expected_sessions:
                if not has_fact(
                    lambda f, m=metric, s=sess: f.get("fact_type") == "peak_frequency"
                    and f.get("metric") == m
                    and f.get("session_index") == s
                    and has_values_or_na(f, fields=("value",))
                ):
                    missing.add(f"peak_frequency:{metric}:session_{sess}")

        # CP per-site P300 table.
        cp_sites = ["C3", "CZ", "C4", "P3", "PZ", "P4"]
        for site in cp_sites:
            for sess in expected_sessions:
                if not has_fact(
                    lambda f, st=site, s=sess: f.get("fact_type") == "p300_cp_site"
                    and f.get("site") == st
                    and f.get("session_index") == s
                    and has_values_or_na(f, fields=("uv", "ms"))
                ):
                    missing.add(f"p300_cp_site:{site}:session_{sess}")

        # CENTRAL-FRONTAL AVERAGE N100.
        for sess in expected_sessions:
            if not has_fact(
                lambda f, s=sess: f.get("fact_type") == "n100_central_frontal_average"
                and f.get("session_index") == s
                and has_values_or_na(f, fields=("uv", "ms"))
            ):
                missing.add(f"n100:central_frontal_average:session_{sess}")

        return missing

    async def _targeted_retry_missing(
        self,
        *,
        extractor_model_id: str,
        page_images: list[PageImage],
        merged: dict[str, Any],
        expected_sessions: list[int],
        missing: set[str],
        candidate_pages: list[int] | None = None,
        debug_dir: Path | None = None,
        attempt_log: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        # Focused retry for P300 Rare Comparison page(s).
        need_cp = any(k.startswith("p300_cp_site:") for k in missing)
        need_n100 = any(k.startswith("n100:") for k in missing)
        if not (need_cp or need_n100):
            return merged

        pages_to_try: list[int] = []
        if candidate_pages:
            pages_to_try = [p for p in candidate_pages if isinstance(p, int)]

        # Fallback: use model page inventory if present, else default to page 2.
        if not pages_to_try:
            pages_to_try = [2]
            inv = merged.get("page_inventory")
            if isinstance(inv, list):
                pages = []
                for item in inv:
                    if not isinstance(item, dict):
                        continue
                    title = item.get("title")
                    page = item.get("page")
                    if isinstance(title, str) and "p300" in title.lower() and isinstance(page, int):
                        pages.append(page)
                if pages:
                    pages_to_try = sorted(set(pages))

        img_by_page = {img.page: img for img in page_images}
        base_pages = [img_by_page[p] for p in pages_to_try if p in img_by_page]
        if not base_pages:
            return merged

        retry_images: list[PageImage] = []
        for page_img in base_pages:
            crops = _try_build_p300_cp_site_crops(page_img)
            if crops:
                retry_images = crops
                break
        if not retry_images:
            retry_images = base_pages

        focus = (
            "RETRY: Extract central-parietal per-site P300 values (C3, CZ, C4, P3, PZ, P4) for every session shown.\n"
            "- If images are cropped panels (labels like C3_panel), each panel corresponds to ONE site.\n"
            "- Use the legend crop (if provided) to map Session 1/2/3 to colors.\n"
            "- For each site+session, extract yield (#), uv (µV), and ms.\n"
            "- If a value is shown as N/A, set uv/ms to null and shown_as=\"N/A\".\n"
            "Also extract CENTRAL-FRONTAL AVERAGE N100 (yield, uv, ms) if present (look for a crop labeled "
            "central_frontal_avg if provided)."
        )
        prompt_text = _data_pack_prompt(pages=[img.page for img in retry_images], focus=focus)

        raw = await self._call_model_multimodal(
            model_id=extractor_model_id,
            prompt_text=prompt_text,
            images=retry_images,
            temperature=0.0,
            max_tokens=2500,
            allow_text_fallback=False,
        )

        if debug_dir is not None:
            try:
                safe_model = "".join(c if c.isalnum() or c in {"-", "_", "."} else "_" for c in extractor_model_id) or "model"
                model_dir = debug_dir / safe_model
                model_dir.mkdir(parents=True, exist_ok=True)
                pages_key = "-".join(str(p) for p in sorted({img.page for img in retry_images}))
                stem = f"retry-p300-{pages_key}"
                (model_dir / f"{stem}.prompt.txt").write_text(prompt_text, encoding="utf-8")
                (model_dir / f"{stem}.raw.txt").write_text(raw, encoding="utf-8")
                image_paths = _save_debug_images(model_dir=model_dir, stem=stem, images=retry_images)
                if attempt_log is not None:
                    attempt_log.append(
                        {
                            "kind": "retry_p300_cp_n100",
                            "model_id": extractor_model_id,
                            "pages": sorted({img.page for img in retry_images}),
                            "labels_sample": [img.label for img in retry_images[:12]],
                            "raw_path": str(model_dir / f"{stem}.raw.txt"),
                            "image_paths": image_paths,
                        }
                    )
            except Exception:
                pass
        part = _json_loads_loose(raw)
        if not isinstance(part, dict) or part.get("schema_version") != DATA_PACK_SCHEMA_VERSION:
            return merged

        if debug_dir is not None:
            try:
                safe_model = "".join(c if c.isalnum() or c in {"-", "_", "."} else "_" for c in extractor_model_id) or "model"
                model_dir = debug_dir / safe_model
                pages_key = "-".join(str(p) for p in sorted({img.page for img in retry_images}))
                stem = f"retry-p300-{pages_key}"
                (model_dir / f"{stem}.parsed.json").write_text(
                    json.dumps(part, indent=2, sort_keys=True), encoding="utf-8"
                )
            except Exception:
                pass

        merged2 = self._merge_data_pack_parts([merged, part], meta={k: v for k, v in merged.items() if k != "derived"})
        merged2["derived"] = self._derive_data_pack_views(merged2, expected_sessions=expected_sessions)
        return merged2

    async def _targeted_retry_summary_missing(
        self,
        *,
        extractor_model_id: str,
        page_images: list[PageImage],
        merged: dict[str, Any],
        expected_sessions: list[int],
        missing: set[str],
        debug_dir: Path | None = None,
        attempt_log: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        need_summary = any(
            k.startswith("performance:")
            or k.startswith("evoked:")
            or k.startswith("state:")
            or k.startswith("peak_frequency:")
            for k in missing
        )
        if not need_summary:
            return merged

        img_by_page = {img.page: img for img in page_images}
        page1 = img_by_page.get(1)
        if page1 is None:
            return merged

        retry_images = _try_build_summary_table_crops(page1) or [page1]
        focus = (
            "RETRY: Extract PAGE-1 SUMMARY metrics for every session shown:\n"
            "- Performance: Physical Reaction Time (ms), Trail Making Test A (sec), Trail Making Test B (sec)\n"
            "- Evoked: Audio P300 Delay (ms), Audio P300 Voltage (µV)\n"
            "- State: CZ Eyes Closed Theta/Beta (Power), F3/F4 Eyes Closed Alpha (Power)\n"
            "- Peak Frequency: Frontal, Central-Parietal, Occipital (Hz)\n"
            "Use the canonical identifiers and include target ranges when shown."
        )
        prompt_text = _data_pack_prompt(pages=[img.page for img in retry_images], focus=focus)

        raw = await self._call_model_multimodal(
            model_id=extractor_model_id,
            prompt_text=prompt_text,
            images=retry_images,
            temperature=0.0,
            max_tokens=2500,
            allow_text_fallback=False,
        )

        if debug_dir is not None:
            try:
                safe_model = "".join(c if c.isalnum() or c in {"-", "_", "."} else "_" for c in extractor_model_id) or "model"
                model_dir = debug_dir / safe_model
                model_dir.mkdir(parents=True, exist_ok=True)
                stem = "retry-summary-page-1"
                (model_dir / f"{stem}.prompt.txt").write_text(prompt_text, encoding="utf-8")
                (model_dir / f"{stem}.raw.txt").write_text(raw, encoding="utf-8")
                image_paths = _save_debug_images(model_dir=model_dir, stem=stem, images=retry_images)
                if attempt_log is not None:
                    attempt_log.append(
                        {
                            "kind": "retry_summary_page_1",
                            "model_id": extractor_model_id,
                            "pages": sorted({img.page for img in retry_images}),
                            "labels_sample": [img.label for img in retry_images[:12]],
                            "raw_path": str(model_dir / f"{stem}.raw.txt"),
                            "image_paths": image_paths,
                        }
                    )
            except Exception:
                pass

        part = _json_loads_loose(raw)
        if not isinstance(part, dict) or part.get("schema_version") != DATA_PACK_SCHEMA_VERSION:
            return merged

        if debug_dir is not None:
            try:
                safe_model = "".join(c if c.isalnum() or c in {"-", "_", "."} else "_" for c in extractor_model_id) or "model"
                model_dir = debug_dir / safe_model
                (model_dir / "retry-summary-page-1.parsed.json").write_text(
                    json.dumps(part, indent=2, sort_keys=True), encoding="utf-8"
                )
            except Exception:
                pass

        merged2 = self._merge_data_pack_parts([merged, part], meta={k: v for k, v in merged.items() if k != "derived"})
        merged2["derived"] = self._derive_data_pack_views(merged2, expected_sessions=expected_sessions)
        return merged2

    async def run_pipeline(self, run_id: str, on_event: OnEvent = None) -> None:
        async def emit(payload: dict[str, Any]) -> None:
            if on_event is not None:
                await on_event(payload)

        with session_scope() as session:
            run = get_run(session, run_id)
            if run is None:
                return
            report = get_report(session, run.report_id)
            if report is None:
                update_run_status(session, run_id, status="failed", error_message="Report not found")
                return

            council_model_ids = _loads_json_list(run.council_model_ids_json)
            if not council_model_ids:
                update_run_status(
                    session,
                    run_id,
                    status="failed",
                    error_message="No council models selected for run",
                )
                return

            update_run_status(session, run_id, status="running")

        await emit({"run_id": run_id, "status": "running"})

        try:
            await self._stage1(run_id, council_model_ids, report, emit)
            await self._stage2(run_id, council_model_ids, emit)
            await self._stage3(run_id, council_model_ids, emit)
            await self._stage4(run_id, emit)
            await self._stage5(run_id, council_model_ids, emit)
            await self._stage6(run_id, council_model_ids, emit)
        except _NeedsAuth as e:
            with session_scope() as session:
                update_run_status(session, run_id, status="needs_auth", error_message=str(e))
            await emit({"run_id": run_id, "status": "needs_auth", "error": str(e)})
            return
        except Exception as e:
            with session_scope() as session:
                update_run_status(session, run_id, status="failed", error_message=str(e))
            await emit({"run_id": run_id, "status": "failed", "error": str(e)})
            return

        with session_scope() as session:
            update_run_status(session, run_id, status="complete")
        await emit({"run_id": run_id, "status": "complete"})

    async def _call_model_chat(
        self,
        *,
        model_id: str,
        prompt_text: str,
        temperature: float,
        max_tokens: int,
    ) -> str:
        attempts = 0
        while True:
            try:
                return await self._llm.chat_completions(
                    model_id=model_id,
                    messages=[{"role": "user", "content": prompt_text}],
                    temperature=temperature,
                    max_tokens=max_tokens,
                    stream=False,
                )
            except UpstreamError as e:
                if e.status_code == 401:
                    raise _NeedsAuth(str(e)) from e
                if (e.status_code in {429, 500, 502, 503, 504} or e.status_code is None) and attempts < 4:
                    await _sleep_backoff(attempts)
                    attempts += 1
                    continue
                raise

    async def _call_model_multimodal(
        self,
        *,
        model_id: str,
        prompt_text: str,
        images: list[PageImage],
        temperature: float,
        max_tokens: int,
        allow_text_fallback: bool = True,
    ) -> str:
        """Call a vision-capable model with text and images."""
        # Build multimodal content array
        content: list[dict] = [{"type": "text", "text": prompt_text}]

        # Add images (page-tagged, in order).
        for img in images:
            tag = f"[PAGE {img.page}]"
            if img.label:
                tag = f"{tag} [{img.label}]"
            content.append({"type": "text", "text": tag})
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{img.base64_png}",
                    "detail": "high"  # Use high detail for clinical data
                }
            })

        messages = [{"role": "user", "content": content}]

        attempts = 0
        while True:
            try:
                return await self._llm.chat_completions(
                    model_id=model_id,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    stream=False,
                )
            except UpstreamError as e:
                if e.status_code == 401:
                    raise _NeedsAuth(str(e)) from e
                if (e.status_code in {429, 500, 502, 503, 504} or e.status_code is None) and attempts < 4:
                    await _sleep_backoff(attempts)
                    attempts += 1
                    continue
                # If multimodal fails, optionally fall back to text-only (NOT suitable for strict data capture).
                if allow_text_fallback and attempts == 0:
                    return await self._call_model_chat(
                        model_id=model_id,
                        prompt_text=prompt_text,
                        temperature=temperature,
                        max_tokens=max_tokens,
                    )
                raise

    async def _write_artifact(
        self,
        *,
        run_id: str,
        stage: StageDef,
        model_id: str,
        text: str,
    ) -> Artifact:
        out_dir = _stage_dir(run_id, stage.num)
        out_dir.mkdir(parents=True, exist_ok=True)
        path = _artifact_path(run_id, stage.num, model_id, stage.ext)
        path.write_text(text, encoding="utf-8")
        with session_scope() as session:
            artifact = create_artifact(
                session,
                run_id=run_id,
                stage_num=stage.num,
                stage_name=stage.name,
                model_id=model_id,
                kind=stage.kind,
                content_path=path,
                content_type=stage.content_type,
            )
        return artifact

    async def _stage1(
        self,
        run_id: str,
        council_model_ids: list[str],
        report: Report,
        emit: Callable[[dict[str, Any]], Awaitable[None]],
    ) -> None:
        stage = STAGES[0]
        prompt = _load_prompt("stage1_analysis.md")
        report_dir = _derive_report_dir(report)
        report_text = _load_best_report_text(report, report_dir)

        # Load images from the report folder (preferred), then fallback lookup.
        page_images = _load_page_images(report, report_dir)

        needs_images = any(is_vision_capable(m) for m in council_model_ids)
        expected_page_count = _page_count_from_markers(report_text)
        pages_present = {img.page for img in page_images}
        missing_pages: list[int] = []
        if expected_page_count:
            missing_pages = [p for p in range(1, expected_page_count + 1) if p not in pages_present]

        # If a vision-capable model is selected but images are missing/incomplete, generate on the fly.
        if needs_images and Path(report.stored_path).suffix.lower() == ".pdf" and (not page_images or missing_pages):
            try:
                full = extract_pdf_full(Path(report.stored_path))
                enhanced_text = full.enhanced_text
                if enhanced_text and enhanced_text.strip():
                    report_text = enhanced_text
                    try:
                        (report_dir / "extracted_enhanced.txt").write_text(enhanced_text, encoding="utf-8")
                        # Keep extracted.txt aligned so the UI preview and any verification tooling never
                        # shows only a single OCR engine.
                        (report_dir / "extracted.txt").write_text(enhanced_text, encoding="utf-8")
                    except Exception:
                        pass

                page_images = []
                for img in full.page_images:
                    if not isinstance(img, dict):
                        continue
                    page = img.get("page")
                    b64 = img.get("base64_png")
                    if isinstance(page, int) and isinstance(b64, str):
                        page_images.append(PageImage(page=page, base64_png=b64))

                # Best-effort persist generated images + per-page sources/metadata for later stages/debugging.
                if page_images:
                    try:
                        pages_dir = report_dir / "pages"
                        pages_dir.mkdir(parents=True, exist_ok=True)
                        for img in page_images:
                            out = pages_dir / f"page-{img.page}.png"
                            out.write_bytes(base64.b64decode(img.base64_png))
                    except Exception:
                        pass

                try:
                    sources_dir = report_dir / "sources"
                    sources_dir.mkdir(parents=True, exist_ok=True)
                    for p in full.per_page_sources:
                        page_num = p.get("page")
                        if not isinstance(page_num, int):
                            continue
                        (sources_dir / f"page-{page_num}.pypdf.txt").write_text(p.get("pypdf_text", ""), encoding="utf-8")
                        (sources_dir / f"page-{page_num}.pymupdf.txt").write_text(p.get("pymupdf_text", ""), encoding="utf-8")
                        (sources_dir / f"page-{page_num}.apple_vision.txt").write_text(p.get("vision_ocr_text", ""), encoding="utf-8")
                        (sources_dir / f"page-{page_num}.tesseract.txt").write_text(p.get("tesseract_ocr_text", ""), encoding="utf-8")

                    meta = dict(full.metadata)
                    meta.update(
                        {
                            "has_enhanced_ocr": True,
                            "has_page_images": True,
                            "page_images_written": len(page_images),
                            "sources_dir": "sources",
                        }
                    )
                    (report_dir / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
                except Exception:
                    pass
            except Exception:
                page_images = []

        is_pdf = Path(report.stored_path).suffix.lower() == ".pdf"
        strict_data = _truthy_env("QEEG_STRICT_DATA_AVAILABILITY", True)
        # Non-PDF uploads can't be validated via page images.
        if not is_pdf:
            strict_data = False
        # Prevent accidental non-strict PDF runs unless explicitly allowed.
        if is_pdf and not strict_data and not _truthy_env("QEEG_ALLOW_NONSTRICT_DATA_AVAILABILITY", False):
            strict_data = True
        # Tests/mocks: don't hard-fail on missing multimodal extraction.
        if all(mid.startswith("mock-") for mid in council_model_ids):
            strict_data = False

        # In strict mode, enforce multi-source extraction coverage (PDF-native + Apple Vision OCR + Tesseract OCR).
        enforce_all_sources = _truthy_env("QEEG_ENFORCE_ALL_SOURCES", True)
        if strict_data and is_pdf and enforce_all_sources:
            meta_path = report_dir / "metadata.json"
            meta: dict[str, Any] | None = None
            if meta_path.exists():
                try:
                    loaded = json.loads(meta_path.read_text(encoding="utf-8"))
                    if isinstance(loaded, dict):
                        meta = loaded
                except Exception:
                    meta = None

            engines: dict[str, Any] = {}
            if isinstance(meta, dict) and meta.get("schema_version") == 2 and isinstance(meta.get("engines"), dict):
                engines = meta["engines"]

            if not engines:
                # Metadata missing/outdated: regenerate report assets in-place (best effort) so strict runs always have
                # a full audit trail (sources/ + metadata.json).
                try:
                    full = extract_pdf_full(Path(report.stored_path))
                    enhanced_text = full.enhanced_text
                    if enhanced_text and enhanced_text.strip():
                        report_text = enhanced_text
                        (report_dir / "extracted_enhanced.txt").write_text(enhanced_text, encoding="utf-8")
                        (report_dir / "extracted.txt").write_text(enhanced_text, encoding="utf-8")

                    page_images = []
                    for img in full.page_images:
                        page = img.get("page") if isinstance(img, dict) else None
                        b64 = img.get("base64_png") if isinstance(img, dict) else None
                        if isinstance(page, int) and isinstance(b64, str):
                            page_images.append(PageImage(page=page, base64_png=b64))

                    pages_dir = report_dir / "pages"
                    pages_dir.mkdir(parents=True, exist_ok=True)
                    for img in page_images:
                        (pages_dir / f"page-{img.page}.png").write_bytes(base64.b64decode(img.base64_png))

                    sources_dir = report_dir / "sources"
                    sources_dir.mkdir(parents=True, exist_ok=True)
                    for p in full.per_page_sources:
                        page_num = p.get("page")
                        if not isinstance(page_num, int):
                            continue
                        (sources_dir / f"page-{page_num}.pypdf.txt").write_text(p.get("pypdf_text", ""), encoding="utf-8")
                        (sources_dir / f"page-{page_num}.pymupdf.txt").write_text(p.get("pymupdf_text", ""), encoding="utf-8")
                        (sources_dir / f"page-{page_num}.apple_vision.txt").write_text(p.get("vision_ocr_text", ""), encoding="utf-8")
                        (sources_dir / f"page-{page_num}.tesseract.txt").write_text(p.get("tesseract_ocr_text", ""), encoding="utf-8")

                    meta2 = dict(full.metadata)
                    meta2.update(
                        {
                            "has_enhanced_ocr": True,
                            "has_page_images": True,
                            "page_images_written": len(page_images),
                            "sources_dir": "sources",
                        }
                    )
                    meta_path.write_text(json.dumps(meta2, indent=2), encoding="utf-8")
                    engines = meta2.get("engines") if isinstance(meta2.get("engines"), dict) else {}
                except Exception:
                    engines = engines or {}

            required = ["pypdf", "pymupdf", "apple_vision", "tesseract"]
            missing_engines = [k for k in required if not engines.get(k)]
            if missing_engines:
                raise RuntimeError(
                    "Strict data availability requested, but required extraction sources are unavailable.\n"
                    f"Missing sources: {', '.join(missing_engines)}\n"
                    f"Report: {report.filename} ({report.id})\n"
                    f"Metadata: {meta_path}\n"
                    "Fix: ensure Apple Vision OCR + Tesseract are available, then re-run: POST /api/reports/{report_id}/reextract"
                )

        extractor_models = [m for m in council_model_ids if is_vision_capable(m)]
        if strict_data and not extractor_models:
            raise RuntimeError(
                "Strict data availability requested, but no vision-capable models were selected.\n"
                "Stage 1 requires at least one vision-capable model to process ALL PDF pages.\n"
                "Select a vision-capable model in the run's council_model_ids (see /api/models)."
            )

        data_pack = await self._ensure_data_pack(
            run_id=run_id,
            report=report,
            report_text=report_text,
            page_images=page_images,
            candidate_extractor_model_ids=extractor_models,
            strict=strict_data,
        )

        transcript_model_id: str | None = None
        if isinstance(data_pack, dict) and isinstance(data_pack.get("extraction_model_id"), str):
            transcript_model_id = data_pack["extraction_model_id"]
        elif extractor_models:
            transcript_model_id = extractor_models[0]

        vision_transcript_text = await self._ensure_vision_transcript(
            run_id=run_id,
            report=report,
            page_images=page_images,
            transcript_model_id=transcript_model_id,
            strict=strict_data,
        )

        data_pack_block = ""
        if data_pack:
            derived_tables: list[str] = []
            derived = data_pack.get("derived")
            if isinstance(derived, dict):
                for key in (
                    "summary_performance_table_markdown",
                    "summary_evoked_table_markdown",
                    "summary_state_table_markdown",
                    "peak_frequency_table_markdown",
                    "p300_cp_table_markdown",
                    "n100_central_frontal_table_markdown",
                ):
                    val = derived.get(key)
                    if isinstance(val, str) and val.strip():
                        derived_tables.append(val.strip())
            dp_json = json.dumps(data_pack, indent=2, sort_keys=True)
            data_pack_block = (
                "STRUCTURED DATA PACK (authoritative transcription from ALL PDF pages, including graphics):\n\n"
                + ("\n\n".join(derived_tables) + "\n\n" if derived_tables else "")
                + "```json\n"
                + dp_json
                + "\n```\n\n"
            )

        vision_transcript_block = ""
        if isinstance(vision_transcript_text, str) and vision_transcript_text.strip():
            vision_transcript_block = (
                "MULTIMODAL VISION TRANSCRIPT (page-grounded transcription from ALL PDF page images):\n\n"
                f"{vision_transcript_text.strip()}\n\n---\n\n"
            )

        workflow_context = _workflow_context_block(stage_num=stage.num, stage_name=stage.name)

        base_prompt_text = (
            f"{prompt}\n\n---\n\n"
            f"{workflow_context}\n\n---\n\n"
            f"{data_pack_block}"
            f"{vision_transcript_block}"
            "FULL qEEG REPORT OCR TEXT (all pages; may include OCR artifacts):\n\n"
            f"{report_text}\n"
        )

        await emit({"run_id": run_id, "stage_num": stage.num, "stage_name": stage.name, "status": "start"})

        async def one(model_id: str) -> tuple[str, str] | None:
            try:
                # Multi-pass multimodal ingestion for vision models: build page-grounded notes in chunks, then write
                # the final long-form report using the notes + full OCR + data pack.
                notes_text = ""
                per_model_notes = _truthy_env("QEEG_STAGE1_PER_MODEL_VISION_NOTES", False)
                if is_vision_capable(model_id) and page_images and (per_model_notes or not vision_transcript_block):
                    chunk_size = int(os.getenv("QEEG_VISION_PAGES_PER_CALL", "8") or "8")
                    if chunk_size <= 0:
                        chunk_size = 8
                    # Hard requirement: PDFs >10 pages must be ingested in 2+ multimodal passes.
                    if len(page_images) > 10 and chunk_size > 10:
                        chunk_size = 10
                    notes_parts: list[str] = []
                    for chunk in _chunked(page_images, chunk_size):
                        pages = [img.page for img in chunk]
                        ingest_prompt = (
                            "Stage 1 multimodal ingestion pass (do NOT write the final report yet).\n"
                            f"Pages in this pass: {', '.join(str(p) for p in pages)}\n\n"
                            "Task:\n"
                            "- For each provided page image, produce a page-by-page markdown transcript with headings "
                            "\"## Page <n>\".\n"
                            "- Enumerate every table/figure/metric visible on that page and transcribe any clearly "
                            "printed numeric values that are likely clinically relevant.\n"
                            "- Do not interpret or diagnose. Do not invent numbers.\n"
                        )
                        notes = await self._call_model_multimodal(
                            model_id=model_id,
                            prompt_text=ingest_prompt,
                            images=chunk,
                            temperature=0.0,
                            max_tokens=2500,
                            allow_text_fallback=not strict_data,
                        )
                        notes_parts.append(notes)
                    notes_text = "\n\n".join(notes_parts).strip()

                final_prompt = base_prompt_text
                if notes_text:
                    final_prompt = (
                        f"{final_prompt}\n\n---\n\n"
                        "MULTIMODAL INGESTION NOTES (generated from ALL PDF pages in multiple passes):\n\n"
                        f"{notes_text}\n"
                    )
                text = await self._call_model_chat(
                    model_id=model_id,
                    prompt_text=final_prompt,
                    temperature=0.2,
                    max_tokens=8000,
                )
                return model_id, text
            except Exception:
                return None

        results = await asyncio.gather(*(one(m) for m in council_model_ids))
        successes = [r for r in results if r is not None]

        for model_id, text in successes:
            await self._write_artifact(run_id=run_id, stage=stage, model_id=model_id, text=text)

        await emit(
            {
                "run_id": run_id,
                "stage_num": stage.num,
                "stage_name": stage.name,
                "status": "complete",
                "success_count": len(successes),
                "requested_count": len(council_model_ids),
            }
        )

    async def _stage2(
        self,
        run_id: str,
        council_model_ids: list[str],
        emit: Callable[[dict[str, Any]], Awaitable[None]],
    ) -> None:
        stage = STAGES[1]
        prompt = _load_prompt("stage2_peer_review.md")

        with session_scope() as session:
            artifacts = _stage_artifacts(session, run_id, 1, kind="analysis")
            run = get_run(session, run_id)
            report = get_report(session, run.report_id) if run else None

        report_text = ""
        if report and report.extracted_text_path:
            report_dir = _derive_report_dir(report)
            report_text = _load_best_report_text(report, report_dir)

        data_pack_text = ""
        dp_path = _data_pack_path(run_id)
        if dp_path.exists():
            try:
                data_pack_text = dp_path.read_text(encoding="utf-8", errors="replace")
            except Exception:
                data_pack_text = ""

        vision_transcript_text = ""
        vt_path = _vision_transcript_path(run_id)
        if vt_path.exists():
            try:
                vision_transcript_text = vt_path.read_text(encoding="utf-8", errors="replace")
            except Exception:
                vision_transcript_text = ""

        workflow_context = _workflow_context_block(stage_num=stage.num, stage_name=stage.name)
        data_pack_block = ""
        if data_pack_text.strip():
            data_pack_block = (
                "STRUCTURED DATA PACK (authoritative transcription from ALL PDF pages, including graphics):\n\n"
                f"```json\n{data_pack_text.strip()}\n```\n\n---\n\n"
            )

        vision_transcript_block = ""
        if vision_transcript_text.strip():
            vision_transcript_block = (
                "MULTIMODAL VISION TRANSCRIPT (page-grounded transcription from ALL PDF page images):\n\n"
                f"{vision_transcript_text.strip()}\n\n---\n\n"
            )

        analyses_by_model: dict[str, str] = {}
        for a in artifacts:
            analyses_by_model[a.model_id] = Path(a.content_path).read_text(encoding="utf-8", errors="replace")

        available_models = [m for m in council_model_ids if m in analyses_by_model]
        if len(available_models) < 2:
            await emit(
                {
                    "run_id": run_id,
                    "stage_num": stage.num,
                    "stage_name": stage.name,
                    "status": "complete",
                    "skipped": True,
                    "reason": "Not enough Stage 1 analyses for peer review",
                }
            )
            return

        label_map = _stable_label_map(run_id, available_models)
        with session_scope() as session:
            set_run_label_map(session, run_id, label_map)

        analysis_blocks: list[str] = []
        for label, model_id in label_map.items():
            analysis_blocks.append(f"Analysis {label}:\n{analyses_by_model[model_id]}".strip())
        all_analyses_text = "\n\n".join(analysis_blocks)

        await emit({"run_id": run_id, "stage_num": stage.num, "stage_name": stage.name, "status": "start"})

        async def one(reviewer_model_id: str) -> tuple[str, str] | None:
            # Reviewer sees all analyses except its own, still labeled A/B/C...
            reviewer_label = next((lbl for lbl, mid in label_map.items() if mid == reviewer_model_id), None)
            filtered: list[str] = []
            for label, mid in label_map.items():
                if mid == reviewer_model_id:
                    continue
                filtered.append(f"Analysis {label}:\n{analyses_by_model[mid]}".strip())
            if not filtered:
                return None
            filtered_text = "\n\n".join(filtered)
            prompt_text = (
                f"{prompt}\n\n---\n\n"
                f"{workflow_context}\n\n---\n\n"
                f"{data_pack_block}"
                f"{vision_transcript_block}"
                f"ORIGINAL qEEG REPORT (for verification - cross-reference all claims against this):\n\n{report_text}\n\n---\n\n"
                f"Reviewer Model ID: {reviewer_model_id}\n"
                f"Your own analysis label (do not review yourself): {reviewer_label}\n\n"
                f"ANALYSES TO REVIEW:\n\n{filtered_text}\n"
            )
            try:
                text = await self._call_model_chat(
                    model_id=reviewer_model_id,
                    prompt_text=prompt_text,
                    temperature=0.1,
                    max_tokens=4000,
                )
                _json_loads_loose(text)  # sanity
                return reviewer_model_id, _strip_to_json(text)
            except Exception:
                return None

        results = await asyncio.gather(*(one(m) for m in available_models))
        successes = [r for r in results if r is not None]

        for model_id, text in successes:
            await self._write_artifact(run_id=run_id, stage=stage, model_id=model_id, text=text)

        await emit(
            {
                "run_id": run_id,
                "stage_num": stage.num,
                "stage_name": stage.name,
                "status": "complete",
                "success_count": len(successes),
                "requested_count": len(available_models),
                "label_map": label_map,
            }
        )

    async def _stage3(
        self,
        run_id: str,
        council_model_ids: list[str],
        emit: Callable[[dict[str, Any]], Awaitable[None]],
    ) -> None:
        stage = STAGES[2]
        prompt = _load_prompt("stage3_revision.md")

        with session_scope() as session:
            s1 = _stage_artifacts(session, run_id, 1, kind="analysis")
            s2 = _stage_artifacts(session, run_id, 2, kind="peer_review")
            run = get_run(session, run_id)
            label_map = json.loads(run.label_map_json or "{}") if run is not None else {}
            report = get_report(session, run.report_id) if run else None

        report_text = ""
        if report and report.extracted_text_path:
            report_dir = _derive_report_dir(report)
            report_text = _load_best_report_text(report, report_dir)

        data_pack_text = ""
        dp_path = _data_pack_path(run_id)
        if dp_path.exists():
            try:
                data_pack_text = dp_path.read_text(encoding="utf-8", errors="replace")
            except Exception:
                data_pack_text = ""

        vision_transcript_text = ""
        vt_path = _vision_transcript_path(run_id)
        if vt_path.exists():
            try:
                vision_transcript_text = vt_path.read_text(encoding="utf-8", errors="replace")
            except Exception:
                vision_transcript_text = ""

        workflow_context = _workflow_context_block(stage_num=stage.num, stage_name=stage.name)
        data_pack_block = ""
        if data_pack_text.strip():
            data_pack_block = (
                "STRUCTURED DATA PACK (authoritative transcription from ALL PDF pages, including graphics):\n\n"
                f"```json\n{data_pack_text.strip()}\n```\n\n---\n\n"
            )

        vision_transcript_block = ""
        if vision_transcript_text.strip():
            vision_transcript_block = (
                "MULTIMODAL VISION TRANSCRIPT (page-grounded transcription from ALL PDF page images):\n\n"
                f"{vision_transcript_text.strip()}\n\n---\n\n"
            )

        analyses_by_model = {
            a.model_id: Path(a.content_path).read_text(encoding="utf-8", errors="replace") for a in s1
        }
        peer_reviews = [
            (a.model_id, Path(a.content_path).read_text(encoding="utf-8", errors="replace")) for a in s2
        ]

        available_models = [m for m in council_model_ids if m in analyses_by_model]
        if not available_models:
            raise RuntimeError("No Stage 1 analyses available for revision")

        await emit({"run_id": run_id, "stage_num": stage.num, "stage_name": stage.name, "status": "start"})

        async def one(model_id: str) -> tuple[str, str] | None:
            analysis = analyses_by_model.get(model_id)
            if not analysis:
                return None
            my_label = next((lbl for lbl, mid in label_map.items() if mid == model_id), None)
            pr_text = "\n\n".join([f"Peer review by {mid}:\n{txt}" for mid, txt in peer_reviews]).strip()
            prompt_text = (
                f"{prompt}\n\n---\n\n"
                f"{workflow_context}\n\n---\n\n"
                f"{data_pack_block}"
                f"{vision_transcript_block}"
                f"ORIGINAL qEEG REPORT (for fact-checking your revision):\n\n{report_text}\n\n---\n\n"
                f"Your Model ID: {model_id}\n"
                f"Your analysis label (if present): {my_label}\n\n"
                f"Your original analysis:\n\n{analysis}\n\n"
                f"Peer review JSON artifacts:\n\n{pr_text}\n"
            )
            try:
                text = await self._call_model_chat(
                    model_id=model_id, prompt_text=prompt_text, temperature=0.2, max_tokens=8000
                )
                return model_id, text
            except Exception:
                return None

        results = await asyncio.gather(*(one(m) for m in available_models))
        successes = [r for r in results if r is not None]
        if not successes:
            raise RuntimeError("All models failed during Stage 3 revision")

        for model_id, text in successes:
            await self._write_artifact(run_id=run_id, stage=stage, model_id=model_id, text=text)

        await emit(
            {
                "run_id": run_id,
                "stage_num": stage.num,
                "stage_name": stage.name,
                "status": "complete",
                "success_count": len(successes),
                "requested_count": len(available_models),
            }
        )

    async def _stage4(
        self,
        run_id: str,
        emit: Callable[[dict[str, Any]], Awaitable[None]],
    ) -> None:
        stage = STAGES[3]
        prompt = _load_prompt("stage4_consolidation.md")

        with session_scope() as session:
            run = get_run(session, run_id)
            if run is None:
                raise RuntimeError("Run not found")
            consolidator = run.consolidator_model_id
            revisions = _stage_artifacts(session, run_id, 3, kind="revision")
            report = get_report(session, run.report_id) if run else None

        report_text = ""
        if report and report.extracted_text_path:
            report_dir = _derive_report_dir(report)
            report_text = _load_best_report_text(report, report_dir)

        data_pack_text = ""
        dp_path = _data_pack_path(run_id)
        if dp_path.exists():
            try:
                data_pack_text = dp_path.read_text(encoding="utf-8", errors="replace")
            except Exception:
                data_pack_text = ""

        vision_transcript_text = ""
        vt_path = _vision_transcript_path(run_id)
        if vt_path.exists():
            try:
                vision_transcript_text = vt_path.read_text(encoding="utf-8", errors="replace")
            except Exception:
                vision_transcript_text = ""

        workflow_context = _workflow_context_block(stage_num=stage.num, stage_name=stage.name)
        data_pack_block = ""
        if data_pack_text.strip():
            data_pack_block = (
                "STRUCTURED DATA PACK (authoritative transcription from ALL PDF pages, including graphics):\n\n"
                f"```json\n{data_pack_text.strip()}\n```\n\n---\n\n"
            )

        vision_transcript_block = ""
        if vision_transcript_text.strip():
            vision_transcript_block = (
                "MULTIMODAL VISION TRANSCRIPT (page-grounded transcription from ALL PDF page images):\n\n"
                f"{vision_transcript_text.strip()}\n\n---\n\n"
            )

        if not revisions:
            raise RuntimeError("Consolidation requires at least one revision artifact")

        revision_text = "\n\n".join(
            [
                f"Revision by {a.model_id}:\n{Path(a.content_path).read_text(encoding='utf-8', errors='replace')}"
                for a in revisions
            ]
        )
        prompt_text = (
            f"{prompt}\n\n---\n\n"
            f"{workflow_context}\n\n---\n\n"
            f"{data_pack_block}"
            f"{vision_transcript_block}"
            f"ORIGINAL qEEG REPORT (the source of truth - verify all claims against this):\n\n{report_text}\n\n---\n\n"
            f"REVISED ANALYSES TO CONSOLIDATE:\n\n{revision_text}\n"
        )

        await emit({"run_id": run_id, "stage_num": stage.num, "stage_name": stage.name, "status": "start"})
        text = await self._call_model_chat(
            model_id=consolidator, prompt_text=prompt_text, temperature=0.2, max_tokens=8000
        )
        await self._write_artifact(run_id=run_id, stage=stage, model_id=consolidator, text=text)
        await emit({"run_id": run_id, "stage_num": stage.num, "stage_name": stage.name, "status": "complete"})

    async def _stage5(
        self,
        run_id: str,
        council_model_ids: list[str],
        emit: Callable[[dict[str, Any]], Awaitable[None]],
    ) -> None:
        stage = STAGES[4]
        prompt = _load_prompt("stage5_final_review.md")

        with session_scope() as session:
            s4 = _stage_artifacts(session, run_id, 4, kind="consolidation")
            run = get_run(session, run_id)
            report = get_report(session, run.report_id) if run else None

        report_text = ""
        if report and report.extracted_text_path:
            report_dir = _derive_report_dir(report)
            report_text = _load_best_report_text(report, report_dir)

        data_pack_text = ""
        dp_path = _data_pack_path(run_id)
        if dp_path.exists():
            try:
                data_pack_text = dp_path.read_text(encoding="utf-8", errors="replace")
            except Exception:
                data_pack_text = ""

        vision_transcript_text = ""
        vt_path = _vision_transcript_path(run_id)
        if vt_path.exists():
            try:
                vision_transcript_text = vt_path.read_text(encoding="utf-8", errors="replace")
            except Exception:
                vision_transcript_text = ""

        workflow_context = _workflow_context_block(stage_num=stage.num, stage_name=stage.name)
        data_pack_block = ""
        if data_pack_text.strip():
            data_pack_block = (
                "STRUCTURED DATA PACK (authoritative transcription from ALL PDF pages, including graphics):\n\n"
                f"```json\n{data_pack_text.strip()}\n```\n\n---\n\n"
            )

        vision_transcript_block = ""
        if vision_transcript_text.strip():
            vision_transcript_block = (
                "MULTIMODAL VISION TRANSCRIPT (page-grounded transcription from ALL PDF page images):\n\n"
                f"{vision_transcript_text.strip()}\n\n---\n\n"
            )

        if not s4:
            raise RuntimeError("Stage 5 requires Stage 4 consolidation artifact")

        consolidated = Path(s4[0].content_path).read_text(encoding="utf-8", errors="replace")

        await emit({"run_id": run_id, "stage_num": stage.num, "stage_name": stage.name, "status": "start"})

        async def one(model_id: str) -> tuple[str, str] | None:
            prompt_text = (
                f"{prompt}\n\n---\n\n"
                f"{workflow_context}\n\n---\n\n"
                f"{data_pack_block}"
                f"{vision_transcript_block}"
                f"ORIGINAL qEEG REPORT (for verification):\n\n{report_text}\n\n---\n\n"
                f"CONSOLIDATED REPORT TO REVIEW:\n\n{consolidated}\n"
            )
            try:
                text = await self._call_model_chat(
                    model_id=model_id, prompt_text=prompt_text, temperature=0.1, max_tokens=2500
                )
                payload = _json_loads_loose(text)
                _validate_stage5(payload)
                return model_id, json.dumps(payload, indent=2, sort_keys=True)
            except Exception:
                return None

        results = await asyncio.gather(*(one(m) for m in council_model_ids))
        successes = [r for r in results if r is not None]
        if not successes:
            raise RuntimeError("All models failed during Stage 5 final review")

        for model_id, text in successes:
            await self._write_artifact(run_id=run_id, stage=stage, model_id=model_id, text=text)

        await emit(
            {
                "run_id": run_id,
                "stage_num": stage.num,
                "stage_name": stage.name,
                "status": "complete",
                "success_count": len(successes),
                "requested_count": len(council_model_ids),
            }
        )

    async def _stage6(
        self,
        run_id: str,
        council_model_ids: list[str],
        emit: Callable[[dict[str, Any]], Awaitable[None]],
    ) -> None:
        stage = STAGES[5]
        prompt = _load_prompt("stage6_final_draft.md")

        with session_scope() as session:
            s4 = _stage_artifacts(session, run_id, 4, kind="consolidation")
            s5 = _stage_artifacts(session, run_id, 5, kind="final_review")
            run = get_run(session, run_id)
            report = get_report(session, run.report_id) if run else None

        report_text = ""
        if report and report.extracted_text_path:
            report_dir = _derive_report_dir(report)
            report_text = _load_best_report_text(report, report_dir)

        data_pack_text = ""
        dp_path = _data_pack_path(run_id)
        if dp_path.exists():
            try:
                data_pack_text = dp_path.read_text(encoding="utf-8", errors="replace")
            except Exception:
                data_pack_text = ""

        vision_transcript_text = ""
        vt_path = _vision_transcript_path(run_id)
        if vt_path.exists():
            try:
                vision_transcript_text = vt_path.read_text(encoding="utf-8", errors="replace")
            except Exception:
                vision_transcript_text = ""

        workflow_context = _workflow_context_block(stage_num=stage.num, stage_name=stage.name)
        data_pack_block = ""
        if data_pack_text.strip():
            data_pack_block = (
                "STRUCTURED DATA PACK (authoritative transcription from ALL PDF pages, including graphics):\n\n"
                f"```json\n{data_pack_text.strip()}\n```\n\n---\n\n"
            )

        vision_transcript_block = ""
        if vision_transcript_text.strip():
            vision_transcript_block = (
                "MULTIMODAL VISION TRANSCRIPT (page-grounded transcription from ALL PDF page images):\n\n"
                f"{vision_transcript_text.strip()}\n\n---\n\n"
            )

        if not s4:
            raise RuntimeError("Stage 6 requires Stage 4 consolidation artifact")
        consolidated = Path(s4[0].content_path).read_text(encoding="utf-8", errors="replace")

        required_changes = _aggregate_required_changes(s5)

        await emit({"run_id": run_id, "stage_num": stage.num, "stage_name": stage.name, "status": "start"})

        async def one(model_id: str) -> tuple[str, str] | None:
            changes = "\n".join([f"- {c}" for c in required_changes]) if required_changes else "(none)"
            prompt_text = (
                f"{prompt}\n\n---\n\n"
                f"{workflow_context}\n\n---\n\n"
                f"{data_pack_block}"
                f"{vision_transcript_block}"
                f"ORIGINAL qEEG REPORT (for any needed verification):\n\n{report_text}\n\n---\n\n"
                f"Required changes to apply:\n{changes}\n\n"
                f"CONSOLIDATED REPORT:\n\n{consolidated}\n"
            )
            try:
                text = await self._call_model_chat(
                    model_id=model_id, prompt_text=prompt_text, temperature=0.2, max_tokens=8000
                )
                return model_id, text
            except Exception:
                return None

        results = await asyncio.gather(*(one(m) for m in council_model_ids))
        successes = [r for r in results if r is not None]
        if not successes:
            raise RuntimeError("All models failed during Stage 6 final draft")

        for model_id, text in successes:
            await self._write_artifact(run_id=run_id, stage=stage, model_id=model_id, text=text)

        await emit(
            {
                "run_id": run_id,
                "stage_num": stage.num,
                "stage_name": stage.name,
                "status": "complete",
                "success_count": len(successes),
                "requested_count": len(council_model_ids),
            }
        )


class _NeedsAuth(RuntimeError):
    pass


def _loads_json_list(text: str) -> list[str]:
    try:
        data = json.loads(text)
    except Exception:
        return []
    if not isinstance(data, list):
        return []
    return [x for x in data if isinstance(x, str)]


def _stage_artifacts(session, run_id: str, stage_num: int, *, kind: str) -> list[Artifact]:
    from sqlalchemy import select

    return list(
        session.scalars(
            select(Artifact)
            .where(Artifact.run_id == run_id, Artifact.stage_num == stage_num, Artifact.kind == kind)
            .order_by(Artifact.created_at.asc())
        )
    )


def _validate_stage5(payload: Any) -> None:
    if not isinstance(payload, dict):
        raise ValueError("Stage 5 payload must be an object")
    vote = payload.get("vote")
    if vote not in {"APPROVE", "REVISE"}:
        raise ValueError("Stage 5 vote must be APPROVE or REVISE")
    for key in ("required_changes", "optional_changes"):
        val = payload.get(key)
        if not isinstance(val, list) or not all(isinstance(x, str) for x in val):
            raise ValueError(f"Stage 5 {key} must be a string[]")
    score = payload.get("quality_score_1to10")
    if not isinstance(score, int) or not (1 <= score <= 10):
        raise ValueError("Stage 5 quality_score_1to10 must be int 1..10")


def _aggregate_required_changes(stage5_artifacts: list[Artifact]) -> list[str]:
    changes: list[str] = []
    seen: set[str] = set()
    for art in stage5_artifacts:
        try:
            payload = json.loads(Path(art.content_path).read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(payload, dict):
            continue
        for c in payload.get("required_changes", []) or []:
            if isinstance(c, str):
                c2 = c.strip()
                if c2 and c2 not in seen:
                    seen.add(c2)
                    changes.append(c2)
    return changes
