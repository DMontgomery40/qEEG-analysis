 You are the next Cursor/Codex coding agent working in:

    /Users/davidmontgomery/qEEG-analysis

  Mission (hard requirements, do not compromise):
  - Make Stage 1 (and therefore the whole 6-stage pipeline) reliably use ALL data in the PDF(s),
    including metrics that only appear inside graphical tables/figures (image-only content).
  - No “data not provided” claims when the data exists in the PDF.
  - No “graphical data not captured” excuses; if multimodal/OCR is available, it must be used.
  - If a PDF is >10 pages, you MUST do 2+ passes (or N passes) to cover all pages / needed pages.
  - Expensive/high-token is acceptable; quality and completeness are the priority.
  - If required numeric tables cannot be extracted reliably, the run must HARD-FAIL with a specific error
    explaining what was missing and what was tried (no silent omission).

  High-level repo overview
  =======================
  qEEG Council is a FastAPI + React app that runs a 6-stage multi-model deliberation workflow on qEEG
  PDF reports via a local OpenAI-compatible proxy (CLIProxyAPI).

  Topology:
  - Frontend: React+Vite on http://localhost:5173
  - Backend: FastAPI on http://localhost:8000
  - Upstream: CLIProxyAPI on http://127.0.0.1:8317
    (wraps Claude/Gemini/Codex CLIs, exposes /v1/models, /v1/chat/completions, /v1/responses)

  Core directories:
  - backend/
    - main.py: FastAPI endpoints
    - council.py: QEEGCouncilWorkflow (6 stages)
    - reports.py: PDF text extraction + OCR + image rendering helpers
    - storage.py: SQLite + file-based artifacts
    - prompts/: stage prompts (markdown/json)
  - frontend/

  Key operational rule:
  - DO NOT run with QEEG_MOCK_LLM=1 for real quality runs.
    That mode swaps in hardcoded mock responses and produces “trash” artifacts in seconds.

  6-stage workflow (backend/council.py)
  =====================================
  Stage 1: Initial analyses (parallel across models)
  - The ONLY stage that directly uses page images (multimodal).
  - Creates per-model stage-1 analysis artifacts (markdown).

  Stage 2: Peer review (parallel, anonymized A/B/C…)
  - Each model reviews other models’ Stage-1 analyses.
  - Output is JSON artifacts.

  Stage 3: Revision (parallel)
  - Each model revises based on peer review.
  - Output is markdown artifacts.

  Stage 4: Consolidation (single consolidator model)
  - Synthesizes Stage-3 revisions into one consolidated report (markdown).

  Stage 5: Final review (parallel)
  - JSON votes/required edits.

  Stage 6: Final drafts (parallel)
  - Markdown drafts applying required edits; UI selects final.

  Critical architecture point:
  - Models MUST have access to source material to avoid hallucinated peer review cascades.
  - The pipeline now provides “full report OCR text + structured data pack” to ALL stages (not just Stage 1).
    This is central to preventing hallucinated peer review based on missing source context.

  Data ingestion and storage model
  ================================
  Reports are stored on disk under data/reports/<patient>/<folder>/...
  Some historical reports had a mismatch:
  - DB report.id != on-disk folder name
  So any code that assumes folder path uses report.id can fail to find pages/images/text.

  Current reliable source of truth for report_dir:
  - report.extracted_text_path.parent (preferred)
  - else Path(report.stored_path).parent fallback

  Re-extraction endpoint
  ======================
  There is a backend endpoint to regenerate OCR/text/images for an existing report:

  - POST /api/reports/{report_id}/reextract

  This re-generates:
  - extracted.txt (page markers; OCR fallback for empty-text pages)
  - extracted_enhanced.txt (best-effort full coverage)
  - pages/page-*.png
  - metadata.json

  This is needed to ensure “data availability” when pypdf misses tables/figures.

  Key change (lossless OCR strategy)
  ==================================
  backend/reports.py: extract_text_enhanced(pdf_path)
  - Now runs pypdf extraction AND OCR on ALL pages (if OCR is available).
  - When both sources have non-empty content, it does NOT “pick one” and drop the other.
    It preserves both with markers:
      --- PYPDF TEXT ---
      --- OCR TEXT ---

  This prevents losing image-embedded tables/figures that pypdf can’t see, while still retaining
  any born-digital text pypdf extracts.

  Stage 1: how multimodal + “data pack” works now
  ===============================================
  Stage 1 is designed around a strict “data pack” concept:

  - Before writing narrative Stage-1 reports, Stage 1 builds a STRUCTURED DATA PACK (JSON)
    that is supposed to be an authoritative transcription of image-only numeric data
    (tables/figures) across ALL pages.

  - The data pack is stored at:
    data/artifacts/<run_id>/stage-1/_data_pack.json

  - That data pack is then included (verbatim JSON) in the prompts for:
    Stage 1 (final narrative), Stage 2, Stage 3, Stage 4, Stage 5, Stage 6.

  This is how we ensure “fresh session models” can verify claims against source facts,
  rather than only seeing another model’s narrative.

  Multi-pass page coverage:
  - Vision models do NOT just get the first N pages anymore.
  - The pipeline chunks pages by QEEG_VISION_PAGES_PER_CALL (default 8) and makes multiple
    calls as needed.

  Key env vars:
  - QEEG_VISION_PAGES_PER_CALL (default "8"): images per multimodal call
  - QEEG_STRICT_DATA_AVAILABILITY (default true): hard-fail if required fields missing

  Hard fail behavior:
  - If strict mode is on (default), Stage 1 will fail the run if required fields are missing
    after retries. Non-PDF uploads and mock runs relax strict mode.

  Recent fixes already implemented (DO NOT RE-DO; verify they remain)
  ==================================================================
  1) Lossless OCR:
  - backend/reports.py extract_text_enhanced now combines pypdf + OCR output.
    It stops dropping content via “choose longer”.

  2) Stage 1 assets pathing:
  - Stage 1 derives report_dir from report.extracted_text_path.parent (or stored_path.parent),
    so folder mismatch no longer breaks page image discovery.

  3) Stage 1 uses best text:
  - Stage 1 prefers extracted_enhanced.txt when present (more complete than extracted.txt).

  4) Multi-pass multimodal ingestion:
  - Stage 1 uses chunked multimodal passes to build “page-grounded notes” for vision models,
    then writes the final long-form Stage 1 report based on notes + full OCR + data pack.

  5) Targeted extraction for “hard tables/figures”:
  - Stage 1 has a targeted retry mechanism to re-run multimodal extraction on the pages that
    actually contain the needed tables (not random pages).

  Specific WAVi / qEEG figure that caused failures (context)
  =========================================================
  The “P300 Rare Comparison” page (often Page 2 of 16) contains per-electrode plots and tiny colored
  numeric triplets for each session:
  - yield count (#)
  - uV
  - ms
  The central-parietal electrodes of special interest: C3, CZ, C4, P3, PZ, P4.

  This was previously being reported by the model as “garbled/unreadable”.
  We now treat these values as REQUIRED in strict mode and fail if not extracted.

  Deterministic N100 extraction added (important)
  ==============================================
  backend/council.py now includes a deterministic parser:
  - _facts_from_report_text_n100_central_frontal(report_text, expected_sessions=[1,2,3])

  It extracts CENTRAL-FRONTAL AVERAGE “N100-UV MS” from OCR text on the P300 Rare Comparison page
  and inserts these into the data pack as fact_type:
  - n100_central_frontal_average

  This makes the pipeline less dependent on an LLM correctly transcribing that tiny block.

  Strict requirements:
  - QEEGCouncilWorkflow._missing_required_fields now considers N100 central-frontal required
    (per session) in strict mode.
    If a future report template truly omits N100, you may need conditional logic, but do not relax
    this without replacing it with an equally strict alternative.

  Stage 2–6 context problem (hallucinated peer review) is addressed
  =================================================================
  The pipeline already passes full source context into every stage prompt:

  - Stage 2 prompt includes:
    - structured data pack JSON
    - full report OCR text
    - the analyses being reviewed (excluding self)
    => this prevents “peer review with no source material”.

  - Stage 3 prompt includes:
    - structured data pack JSON
    - full report OCR text
    - own prior analysis + peer reviews

  - Stage 4 prompt includes:
    - structured data pack JSON
    - full report OCR text
    - Stage 3 revisions

  - Stage 5 and Stage 6 include:
    - structured data pack JSON
    - full report OCR text
    - consolidation and/or required changes

  Verify this pattern remains in backend/council.py.

  Known-good reference run (real, non-mock)
  =========================================
  A full successful run exists demonstrating CP per-site P300 + N100 extraction:

  - run_id: e69e1c2b-9d2f-40cf-b384-233bf2f07f5a
  - data pack:
    data/artifacts/e69e1c2b-9d2f-40cf-b384-233bf2f07f5a/stage-1/_data_pack.json
  - export:
    data/exports/e69e1c2b-9d2f-40cf-b384-233bf2f07f5a/final.md
    data/exports/e69e1c2b-9d2f-40cf-b384-233bf2f07f5a/final.pdf

  If the UI shows a run stuck “running”, one older run that timed out was:
  - run_id: d04db42d-fbec-4b7a-8ba4-6879969ad026
  (This is a UI/state artifact; don’t waste time unless it blocks workflows.)

  Acceptance criteria checklist (must be met; otherwise fail run)
  ==============================================================
  For the “TBI-Male_20_treatments...” report, Stage 1 output and downstream stages MUST include:

  A) Performance metrics table (all sessions, actual numbers):
  - Physical Reaction Time (ms)
  - Trail Making Test A (sec)
  - Trail Making Test B (sec)

  B) Evoked potentials:
  - Audio P300 Delay (ms)
  - Audio P300 Voltage (uV)

  C) State / background EEG metrics:
  - CZ Eyes Closed Theta/Beta (Power)
  - F3/F4 Eyes Closed Alpha (Power)
  - Peak Frequency (Frontal, Central-Parietal, Occipital)

  D) P300 Rare Comparison per-electrode table for the central-parietal sites:
  - C3, CZ, C4, P3, PZ, P4
  For each session:
  - yield (#)
  - amplitude (uV)
  - latency (ms)
  No “unreadable” hand-waving; if you can’t get it, hard-fail with specifics.

  E) CENTRAL-FRONTAL AVERAGE N100:
  For each session:
  - yield (#)
  - N100 uV (can be negative)
  - N100 ms
  Allow N/A if report indicates it, but do not omit.

  F) Stage 4 consolidation must preserve these numbers and restate them correctly.
  G) The selected export final.md must contain the numbers.

  If any of these are missing:
  - the run must be marked failed with a clear message including:
    - what specific fields were missing
    - which pages were attempted
    - what OCR/multimodal retries occurred
    - where artifacts are saved for debugging

  Key PDFs for “real tests”
  ========================
  /Users/davidmontgomery/Downloads/five/TBI-Male_20_treatments_ final qeeg_Redacted.pdf (primary)
  /Users/davidmontgomery/Downloads/five/BB_male_Long COVID_20 treatments_final qeeg_Redacted.pdf
  /Users/davidmontgomery/Downloads/five/LJ_TBI_male_20 treatments_ final qeeg_Redacted.pdf
  /Users/davidmontgomery/Downloads/five/PG_femal_TBI+Parkinsons_20 treatments_final qeeg_Redacted.pdf
  /Users/davidmontgomery/Downloads/five/RR_Male_Long_COVID_20 treatments_Final qeeg_Redacted.pdf

  How to run (manual)
  ===================
  1) Start CLIProxyAPI and ensure you can list models:
     - GET http://127.0.0.1:8317/v1/models

  2) Start backend:
     - uvicorn backend.main:app --reload --port 8000

  3) Upload report via UI or API, create a run.

  4) Call re-extraction for the report if needed:
     - POST /api/reports/{report_id}/reextract
     Verify on disk the report folder contains:
     - extracted.txt
     - extracted_enhanced.txt
     - pages/page-1.png ... page-N.png

  5) Start run:
     - POST /api/runs/{run_id}/start
     Watch stream:
     - GET /api/runs/{run_id}/stream

  6) Verify artifacts:
     - data/artifacts/<run_id>/stage-1/_data_pack.json (must contain required facts)
     - stage-4 consolidation artifact
     - exported final.md includes numeric tables

  Apple Vision OCR (exploratory future improvement; not currently integrated)
  ==========================================================================
  Current qEEG-analysis OCR engine: Tesseract (pytesseract) + PyMuPDF rendering.

  There is Apple Vision OCR code in:
  - /Users/davidmontgomery/secondbrain/src/second_brain/ocr/apple_vision_ocr.py
  - /Users/davidmontgomery/secondbrain-ds/brain_scans/ocr_pdfs.py
  - parsing examples in /Users/davidmontgomery/secondbrain-ds/brain_scans/parse_wavi_reports.py
    and parse_wavi_complete.py

  Important notes before adopting:
  - This repo’s current Python env does NOT have pyobjc Vision installed (import Vision fails).
  - If adopting Apple Vision OCR:
    - Use VNRecognizeTextRequest (Accurate), not Live Text private APIs unless willing to accept fragility.
    - Disable language correction for numeric-heavy ROIs.
    - Capture real confidence + bounding boxes (for audit/debug and deterministic parsing).
    - Keep Tesseract as a backstop/cross-check (no single OCR engine is perfect).

  Next task suggestions (likely priorities)
  ========================================
  - Improve deterministic extraction coverage for remaining “hard” pages (e.g., coherence heatmap tables)
    using ROI-based OCR (and/or Apple Vision OCR once integrated), with strict schema validation.
  - Add explicit per-page/ROI debug artifacts on failure (store crops + OCR tokens + confidences).
  - Ensure Stage 1’s data pack is “the authoritative facts” so Stage 2–6 never hallucinate missing data.
  - Consider PDF-native extraction (PyMuPDF text with coordinates) when numbers are vector text instead of raster,
    but still keep OCR/multimodal for image-only content.

  Do not:
  - Do not enable QEEG_MOCK_LLM for “real” runs.
  - Do not relax strictness to make runs “complete” if data is missing; missing must fail loudly.
  - Do not allow the model to claim “not provided” if OCR/data pack contains the data.

  End of handoff.