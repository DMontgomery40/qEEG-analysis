from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.reports import extract_pdf_full


def main() -> int:
    parser = argparse.ArgumentParser(description="qEEG Council PDF extraction diagnostics (per-page source lengths).")
    parser.add_argument("pdf_path", type=Path, help="Path to a PDF report")
    parser.add_argument("--write-meta", type=Path, default=None, help="Optional path to write metadata JSON")
    args = parser.parse_args()

    pdf_path: Path = args.pdf_path
    if not pdf_path.exists():
        raise SystemExit(f"PDF not found: {pdf_path}")

    extracted = extract_pdf_full(pdf_path)
    meta = extracted.metadata

    print(f"pdf: {pdf_path}")
    print(f"page_count: {meta.get('page_count')}")
    print(f"render_zoom: {meta.get('render_zoom')}")
    print(f"engines: {json.dumps(meta.get('engines', {}), indent=2, sort_keys=True)}")
    print("")

    pages = meta.get("pages") or []
    for p in pages:
        if not isinstance(p, dict):
            continue
        page_num = p.get("page")
        if not isinstance(page_num, int):
            continue
        pypdf_chars = p.get("pypdf_chars", 0)
        pymupdf_chars = p.get("pymupdf_chars", 0)
        apple_chars = p.get("apple_vision_chars", 0)
        tess_chars = p.get("tesseract_chars", 0)
        has_png = bool(p.get("has_png"))
        print(
            f"page {page_num:>3}: "
            f"pypdf={pypdf_chars:>5}  pymupdf={pymupdf_chars:>5}  "
            f"apple_vision={apple_chars:>5}  tesseract={tess_chars:>5}  "
            f"png={'yes' if has_png else 'no'}"
        )

    if args.write_meta is not None:
        args.write_meta.parent.mkdir(parents=True, exist_ok=True)
        args.write_meta.write_text(json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")
        print(f"\nWrote metadata: {args.write_meta}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
