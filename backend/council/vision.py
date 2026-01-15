from __future__ import annotations

import base64
from pathlib import Path

from .types import PageImage


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
    crops.append(
        PageImage(page=page_image.page, base64_png=base64.b64encode(buf.getvalue()).decode("utf-8"), label="legend")
    )

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

