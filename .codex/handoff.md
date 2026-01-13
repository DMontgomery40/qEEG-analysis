# qEEG Council — Agent Handoff (Stage 1 Data Completeness + Strictness)

You are the next Cursor/Codex coding agent working in:

    /Users/davidmontgomery/qEEG-analysis

This handoff is intentionally detailed. It is meant to be the “source of truth” for the most recent
Stage‑1 data-ingestion hardening work, what to verify next, and what to watch out for.

---

## Mission (hard requirements — do not compromise)

- Make Stage 1 (and therefore the whole 6-stage pipeline) reliably use **ALL** data in the PDF(s),
  including metrics that only appear inside graphical tables/figures (image-only content).
- No “data not provided” claims when the data exists in the PDF.
- No “graphical data not captured” excuses; if multimodal/OCR is available, it must be used.
- If a PDF is **>10 pages**, you MUST do **2+ passes** (or N passes) to cover all pages / needed pages.
- Expensive/high-token is acceptable; **quality and completeness are the priority**.
- If required numeric tables cannot be extracted reliably, the run must **HARD-FAIL** with a specific error
  explaining what was missing and what was tried (no silent omission).

---

## High-level repo overview

qEEG Council is a FastAPI + React app that runs a 6-stage multi-model deliberation workflow on qEEG
PDF reports via a local OpenAI-compatible proxy (CLIProxyAPI).

Topology:
- Frontend: React+Vite on `http://localhost:5173`
- Backend: FastAPI on `http://localhost:8000`
- Upstream: CLIProxyAPI on `http://127.0.0.1:8317`
  - wraps Claude/Gemini/Codex CLIs
  - exposes `/v1/models`, `/v1/chat/completions`, `/v1/responses`

Core directories:
- `backend/`
  - `main.py`: FastAPI endpoints + SSE
  - `council.py`: `QEEGCouncilWorkflow` (6-stage pipeline + Stage-1 data pack)
  - `reports.py`: PDF extraction (pypdf + PyMuPDF render + Apple Vision OCR (macOS) + Tesseract backstop)
  - `apple_vision_ocr.py`: Apple Vision OCR wrapper (PyObjC; macOS only)
  - `storage.py`: SQLite + file artifacts
  - `prompts/`: stage prompts
- `frontend/`

Critical operational rule:
- DO NOT run with `QEEG_MOCK_LLM=1` for real quality runs.
  That mode uses canned mock outputs; it is tests-only.

---

## Current Stage 1 model: “lossless OCR + multimodal data pack + multi-pass”

Stage 1 is the only stage that directly uses images. Stage 2–6 now also receive source material via:
- `extracted_enhanced.txt` (lossless OCR strategy; includes OCR even when pypdf has text)
- `stage-1/_data_pack.json` (structured facts transcribed from images; intended to be authoritative)
- `stage-1/_vision_transcript.md` (page-grounded multimodal transcript so later stages can “see” image-only tables/figures)

### Lossless OCR strategy (already implemented; do not undo)

This repo’s extraction model assumes you must preserve redundancy, not “choose one” text source:
- `backend/reports.py` `extract_text_enhanced(pdf_path)`:
  - runs pypdf text extraction AND PyMuPDF text extraction AND OCR on ALL pages (if any OCR engine is available)
  - Apple Vision OCR (macOS) is used when available (can be disabled via env var); Tesseract remains as backstop
  - if sources have unique content, it keeps multiple with markers (deduping trivial duplicates/subsets):
    - `--- PYPDF TEXT ---`
    - `--- PYMUPDF TEXT ---`
    - `--- TESSERACT OCR ---`
    - `--- APPLE VISION OCR ---`
  - this prevents losing image-embedded tables/figures
  - per-page source outputs are also saved under `sources/` in each report directory for audit/debug

Operationally, the “repair” button is:
- `POST /api/reports/{report_id}/reextract`
  - regenerates `extracted.txt`, `extracted_enhanced.txt`, `pages/page-*.png`, `metadata.json`

### 6-stage workflow context (why Stage 1 strictness matters)

Stage 1 (Initial analyses):
- only stage that directly sees page images (multimodal)
- builds `_data_pack.json` first (structured facts), then writes narrative analyses per model

Stage 2–6 (Peer review → Revision → Consolidation → Final review → Final draft):
- do NOT see images
- **must** receive full source context to prevent “hallucinated peer review cascades”
- pipeline passes to all later-stage prompts:
  - full report OCR text (prefers `extracted_enhanced.txt`)
  - the Stage‑1 data pack JSON (including derived markdown tables)
  - the Stage‑1 vision transcript markdown (broad image transcription, used when OCR misses tables/figures)

Key file paths:
- Reports live under `data/reports/<patient_id>/<upload_id>/...`
  - **DB report.id is NOT guaranteed to equal upload_id**
  - Always locate report assets via:
    - `report.extracted_text_path.parent` (preferred report_dir)
    - else `Path(report.stored_path).parent`
- Artifacts live under `data/artifacts/<run_id>/stage-<n>/...`
  - Stage 1 data pack: `data/artifacts/<run_id>/stage-1/_data_pack.json`
  - Stage 1 vision transcript: `data/artifacts/<run_id>/stage-1/_vision_transcript.md`

---

## What changed in this pass (the important deltas)

Most Stage‑1 strictness work is centered in `backend/council.py`. OCR + extraction improvements also touched
`backend/reports.py` and added `backend/apple_vision_ocr.py` (plus dependency updates in `pyproject.toml` / `uv.lock`).

### Code-level map (how Stage 1 now works end-to-end)

The Stage‑1 sequence inside `QEEGCouncilWorkflow._stage1()` is:
1) Determine `report_dir` (folder-mismatch safe) and load best report text:
   - `_derive_report_dir(report)`
   - `_load_best_report_text(report, report_dir)` (now always prefers enhanced OCR)
2) Load `pages/page-*.png` and auto-heal missing pages from `original.pdf` when needed:
   - `_load_page_images(report, report_dir)`
   - `_page_count_from_markers(report_text)` + missing page detection
   - regeneration via `extract_pdf_with_images(Path(report.stored_path))`
3) Build/upgrade the Stage‑1 structured data pack (strict mode):
   - `_ensure_data_pack(run_id, report, report_text, page_images, extractor_models, strict)`
   - multi-pass across ALL pages (chunked)
   - deterministic facts injected from OCR text (summary + N100 + peak frequency)
   - missing-required-fields check
   - targeted retries (P300/N100 crops; summary-table crops)
   - on failure: write `_data_pack_failure.json` + debug artifacts, then raise RuntimeError
4) Generate a run-level multimodal vision transcript (markdown) across ALL pages:
   - `_ensure_vision_transcript(...)`
   - saved to `stage-1/_vision_transcript.md` and passed to all later stages
5) For each model:
   - write final narrative analysis using:
     - stage prompt + workflow context + data pack block (tables + JSON) + vision transcript + full OCR text
   - (fallback) if the run-level transcript is unavailable and the model is vision-capable, Stage 1 may still
     do multi-pass multimodal ingestion notes across ALL pages

### 1) “Best report text” now truly prefers enhanced OCR (no length heuristic)

Problem:
- Some reports have critical information that appears only in OCR text (especially tables/figures),
  and a “choose longer text” heuristic can accidentally drop OCR coverage.

Fix:
- `_load_best_report_text()` now uses `extracted_enhanced.txt` whenever it exists and is non-empty,
  regardless of length. This aligns with “no compromise on data availability”.

Where:
- `backend/council.py` function `_load_best_report_text()`

### 2) Stage 1 auto-heals missing/incomplete page images (no silent partial-page runs)

Problem:
- Historical runs or partial extractions can leave `pages/page-*.png` incomplete (missing pages),
  which breaks the “ALL pages available” guarantee even if the PDF is present.

Fix:
- Stage 1 now determines expected page count from page markers in report text and checks whether the
  on-disk `pages/page-*.png` set is complete.
- If images are missing/incomplete and the report is a PDF, Stage 1 regenerates images from `original.pdf`
  using `extract_pdf_with_images()` and writes them into `report_dir/pages/page-<n>.png`.
- It also refreshes `extracted_enhanced.txt` on the fly if needed.

Where:
- `backend/council.py` `_stage1()` logic (search for `expected_page_count` / `missing_pages`)
- Helpers:
  - `_page_count_from_markers(report_text)`
  - `_load_page_images()`

Operational note:
- This regeneration writes images even if files already exist (intentionally “repair first”).
  If someone manually edited page PNGs for debugging, rerunning Stage 1 may overwrite them.

### 3) Hard requirement: PDFs >10 pages => 2+ multimodal passes (guaranteed)

Problem:
- Even if the system chunks pages, a user could set `QEEG_VISION_PAGES_PER_CALL` too high,
  resulting in a single pass for a >10 page PDF (violating the hard requirement).

Fix:
- Both data-pack extraction and Stage‑1 multimodal ingestion clamp the chunk size:
  - if `len(page_images) > 10` then `chunk_size = min(chunk_size, 10)`
  - This guarantees 2+ multimodal calls for PDFs >10 pages.

Where:
- `backend/council.py` `_ensure_data_pack()` chunk sizing
- `backend/council.py` `_stage1()` multimodal notes chunk sizing

### 4) Peak Frequency is now first-class + required (deterministic + strict)

Problem:
- Acceptance criteria required Peak Frequency by region (frontal, central-parietal, occipital).
  It previously existed in some data packs via LLM extraction, but it wasn’t enforced by strict-mode checks,
  and Stage 1 could still degrade into “not provided” narratives if upstream transcription missed it.

Fix:
- Data pack prompt now explicitly defines canonical identifiers for peak frequency:
  - fact_type: `peak_frequency`
  - metric: `frontal_peak_frequency_ec | central_parietal_peak_frequency_ec | occipital_peak_frequency_ec`
- Deterministic extraction from OCR text (PAGE 1) now extracts these values into the data pack:
  - `_facts_from_report_text_summary()` now emits `peak_frequency` facts.
- Strict required-field validation now requires these values per session:
  - `_missing_required_fields()` now asserts peak frequency facts exist for every expected session.

Where:
- `backend/council.py`:
  - `_data_pack_prompt()`
  - `_facts_from_report_text_summary()`
  - `QEEGCouncilWorkflow._missing_required_fields()`

Important:
- This may cause strict failures on report templates that genuinely omit Peak Frequency. Do not relax
  strictness without adding an equally strict alternative condition (template detection + explicit “not present” proof).

### 5) Debug artifacts + failure artifacts are now explicit, saved, and actionable

Problem:
- Strict-mode failures need to explain what was missing and what was tried, and leave artifacts for debugging.
  Previously, it was possible to fail without enough forensic detail.

Fix:
- When strict mode is on (or when `QEEG_SAVE_DATA_PACK_DEBUG=1`), Stage 1 saves:
  - per-multimodal-call prompts + raw model outputs + parsed JSON + the exact PNGs sent to the model (including crops)
  - stored under: `data/artifacts/<run_id>/stage-1/_data_pack_debug/<model_id>/...`
- On strict failure, Stage 1 writes:
  - `data/artifacts/<run_id>/stage-1/_data_pack_failure.json`
    - includes missing fields
    - pages processed
    - pass chunking
    - targeted retry kinds
    - attempt log (with paths to raw outputs and the saved images per call)
    - partial data pack snapshot
- The exception message raised on failure includes paths to the above artifacts and suggests re-extraction.

Where:
- `backend/council.py` `_ensure_data_pack()`

New env var:
- `QEEG_SAVE_DATA_PACK_DEBUG=1` (default false)
  - Forces debug artifact saving even if strict mode is off.

### 6) Targeted retries are expanded (and now logged)

Existing behavior (kept):
- There is a targeted retry for P300 Rare Comparison pages:
  - tries to locate likely pages deterministically from OCR text
  - uses `_try_build_p300_cp_site_crops()` to produce larger crops (legend + per-site panels + central-frontal block)
  - re-runs multimodal extraction with a focused prompt

New behavior:
- A targeted retry for PAGE 1 summary metrics now runs when summary required fields are missing:
  - `_try_build_summary_table_crops()` generates OCR-friendly crops for the PAGE 1 summary table + peak frequency block
  - re-runs multimodal extraction with a focused summary prompt
- Both targeted retries save debug artifacts (prompt + raw + parsed JSON) when debug is enabled.

Where:
- `backend/council.py`:
  - `_try_build_summary_table_crops()`
  - `QEEGCouncilWorkflow._targeted_retry_summary_missing()`
  - `QEEGCouncilWorkflow._targeted_retry_missing()` (P300/N100) extended with debug logging

### 7) Data pack now includes “derived markdown tables” for direct downstream use

Problem:
- Models sometimes “refuse” to build clean tables even when JSON facts are present.
- Humans need fast verification that required metrics are present without manually digging through JSON.

Fix:
- `data_pack["derived"]` now includes ready-to-paste markdown tables:
  - `summary_performance_table_markdown`
  - `summary_evoked_table_markdown`
  - `summary_state_table_markdown`
  - `peak_frequency_table_markdown`
  - `p300_cp_table_markdown`
  - `n100_central_frontal_table_markdown`
- Stage 1 prompt now includes these derived tables at the top of the “STRUCTURED DATA PACK” block.

Where:
- `backend/council.py`:
  - `QEEGCouncilWorkflow._derive_data_pack_views()`
  - `_stage1()` prompt composition

### 8) Docs updated to reflect real behavior

Prior docs claimed Stage 1 was limited to the first 10 pages. That is no longer true.

Updated:
- `CLAUDE.md`
- `backend/AGENTS.md`
- `.codex/skills/qeeg-council-backend/SKILL.md`

---

## Environment variables that matter (Stage 1)

- `QEEG_STRICT_DATA_AVAILABILITY` (default true)
  - When true, Stage 1 MUST produce a complete data pack with all required fields, otherwise the run fails.
  - Automatically disabled for:
    - non-PDF uploads
    - all-mock model runs

- `QEEG_ALLOW_NONSTRICT_DATA_AVAILABILITY` (default false)
  - When true, allows setting `QEEG_STRICT_DATA_AVAILABILITY=0` for PDFs (not recommended for real report quality runs).

- `QEEG_ENFORCE_ALL_SOURCES` (default true)
  - When true (recommended), strict runs require **all** extraction layers: pypdf text, PyMuPDF text, Apple Vision OCR, and Tesseract OCR.

- `QEEG_VISION_PAGES_PER_CALL` (default `"8"`)
  - Number of page images per multimodal call.
  - If the PDF has >10 pages, this value is clamped to 10 to guarantee 2+ passes.

- `QEEG_SAVE_DATA_PACK_DEBUG` (default false)
  - If true, saves debug artifacts even when strict mode is off.
  - In strict mode, debug saving is enabled automatically.

- `QEEG_PDF_RENDER_ZOOM` (default `"3.0"`)
  - Render zoom for page PNGs used by OCR and Stage‑1 multimodal (higher = larger images/cost; also can improve readability).

- `QEEG_VISION_TRANSCRIPT_PAGES_PER_CALL` (default `"2"`)
  - Pages per multimodal call when generating `stage-1/_vision_transcript.md`.
  - If the PDF has >10 pages, this value is clamped to 10 (still guaranteeing 2+ passes).

- `QEEG_VISION_TRANSCRIPT_MAX_TOKENS` (default `"4000"`)
  - Max output tokens per multimodal call when generating `stage-1/_vision_transcript.md`.

- `QEEG_LLM_TIMEOUT_S` (default `"600"`)
  - HTTP timeout (seconds) for CLIProxyAPI calls; large multimodal passes can require higher timeouts.

- `QEEG_STAGE1_PER_MODEL_VISION_NOTES` (default false)
  - If true, Stage 1 will also run per-model multimodal “page notes” ingestion even when the run-level
    vision transcript exists (expensive; mostly useful for debugging/model comparisons).

- `QEEG_DISABLE_APPLE_VISION_OCR` (default false)
  - Force-disable Apple Vision OCR even on macOS (falls back to Tesseract if available).

- `QEEG_APPLE_VISION_RECOGNITION_LEVEL` (default `"accurate"`)
  - Apple Vision OCR quality/speed tradeoff (`fast` | `accurate`).

- `QEEG_APPLE_VISION_LANGUAGE_CORRECTION` (default false)
  - If true, Apple Vision applies language correction (often harmful for numeric-heavy tables/figures).

---

## Data pack: required facts (canonical identifiers)

The data pack is the strict contract that prevents downstream “data not provided” hallucinations.

Fact types and identifiers currently treated as REQUIRED in strict mode:

1) `performance_metric` (per session)
   - `metric`: `physical_reaction_time` (ms) (often includes `sd_plus_minus`)
   - `metric`: `trail_making_test_a` (sec)
   - `metric`: `trail_making_test_b` (sec)

2) `evoked_potential` (per session)
   - `metric`: `audio_p300_delay` (ms)
   - `metric`: `audio_p300_voltage` (µV)

3) `state_metric` (per session)
   - `metric`: `cz_theta_beta_ratio_ec` (ratio)
   - `metric`: `f3_f4_alpha_ratio_ec` (ratio)

4) `peak_frequency` (per session)
   - `metric`: `frontal_peak_frequency_ec` (Hz)
   - `metric`: `central_parietal_peak_frequency_ec` (Hz)
   - `metric`: `occipital_peak_frequency_ec` (Hz)

5) `p300_cp_site` (per session, per site)
   - `site`: `C3 | CZ | C4 | P3 | PZ | P4`
   - required numeric fields (unless explicitly `shown_as: "N/A"`):
     - `uv` (µV)
     - `ms` (latency)
   - also captured:
     - `yield` (#)

6) `n100_central_frontal_average` (per session)
   - required numeric fields (unless explicitly `shown_as: "N/A"`):
     - `uv` (µV; can be negative)
     - `ms`
   - also captured:
     - `yield` (#)

Session mapping:
- `expected_session_indices` is inferred from OCR text (`Session 1`, `Session 2`, ...)
- Strict mode expects all required facts for every session index in that list.

---

## Validation performed in this pass (sanity checks)

- Unit tests: `uv run pytest -q` (passed locally in this repo state).
- Apple Vision OCR sanity:
  - Verified `backend.apple_vision_ocr.apple_vision_available()` returns true on this machine.
  - Verified `backend.reports.extract_text_enhanced()` includes `--- APPLE VISION OCR ---` blocks on a real 16-page WAVi PDF where page 1 has no pypdf text (PDF-native extraction failure).
- Manual verification against the known-good run:
  - loaded `data/artifacts/e69e1c2b-9d2f-40cf-b384-233bf2f07f5a/stage-1/_data_pack.json`
  - verified `_missing_required_fields()` returns empty for sessions `[1,2,3]`
  - verified derived tables render correctly for peak frequency and summaries

---

## “All data must be available” preflight checklist (do this before blaming models)

For a given report directory: `data/reports/<patient_id>/<upload_id>/`

Verify it contains:
- `original.pdf`
- `extracted.txt`
- `extracted_enhanced.txt` (may include `--- APPLE VISION OCR ---` / `--- TESSERACT OCR ---` blocks when OCR adds unique content)
- `pages/page-1.png ... page-N.png`
- `metadata.json` (nice-to-have, not required for Stage 1 to run)

If any are missing/garbled:
- Use `POST /api/reports/{report_id}/reextract`
  - This re-generates text + enhanced OCR + page images into the correct folder
  - Note: report folder is derived from `report.extracted_text_path.parent` to avoid DB-folder mismatches

---

## Hard-fail behavior: what to expect when strict-mode breaks

If required numeric facts cannot be extracted:
- The run status should become `failed`.
- Error message includes:
  - missing fields
  - pages processed
  - multimodal passes (page chunks)
  - targeted retries used (if any)
  - extractor model id
  - paths to failure artifacts

Artifacts on failure:
- `data/artifacts/<run_id>/stage-1/_data_pack_failure.json`
  - This is the fastest place to debug. It includes:
    - missing_fields
    - attempt_log (paths to raw outputs)
    - partial_data_pack snapshot (facts gathered so far)
- `data/artifacts/<run_id>/stage-1/_data_pack_debug/<model_id>/...`
  - Per-pass `*.prompt.txt`, `*.raw.txt`, `*.parsed.json`
  - Retry attempts also create `retry-*` files

---

## Acceptance criteria checklist (must be met; otherwise fail run)

For the WAVi “TBI-Male_20_treatments...” style reports, Stage 1 output and downstream stages MUST include:

A) Performance metrics table (all sessions, actual numbers):
- Physical Reaction Time (ms)
- Trail Making Test A (sec)
- Trail Making Test B (sec)

B) Evoked potentials:
- Audio P300 Delay (ms)
- Audio P300 Voltage (µV)

C) State / background EEG metrics:
- CZ Eyes Closed Theta/Beta (Power)
- F3/F4 Eyes Closed Alpha (Power)
- Peak Frequency (Frontal, Central-Parietal, Occipital) (Hz)

D) P300 Rare Comparison per-electrode table for central-parietal sites:
- C3, CZ, C4, P3, PZ, P4
For each session:
- yield (#)
- amplitude (µV)
- latency (ms)
No “unreadable” hand-waving; if you can’t get it, hard-fail with specifics.

E) CENTRAL-FRONTAL AVERAGE N100 (deterministic extraction exists; still required):
For each session:
- yield (#)
- N100 µV (can be negative)
- N100 ms
Allow N/A only if report indicates it explicitly (do not omit).

F) Stage 4 consolidation must preserve these numbers and restate them correctly.
G) The selected export final.md must contain the numbers.

---

## Known-good reference run (real, non-mock)

There is a known-good run demonstrating complete extraction:
- run_id: `e69e1c2b-9d2f-40cf-b384-233bf2f07f5a`
- data pack:
  - `data/artifacts/e69e1c2b-9d2f-40cf-b384-233bf2f07f5a/stage-1/_data_pack.json`
- export:
  - `data/exports/e69e1c2b-9d2f-40cf-b384-233bf2f07f5a/final.md`
  - `data/exports/e69e1c2b-9d2f-40cf-b384-233bf2f07f5a/final.pdf`

Notes:
- Older runs won’t automatically have the newer derived table keys in `data_pack["derived"]` until Stage 1 is re-run.
  New runs will include them.

---

## Key PDFs for “real tests” (local paths)

Primary:
- `/Users/davidmontgomery/Downloads/five/TBI-Male_20_treatments_ final qeeg_Redacted.pdf`

Additional:
- `/Users/davidmontgomery/Downloads/five/BB_male_Long COVID_20 treatments_final qeeg_Redacted.pdf`
- `/Users/davidmontgomery/Downloads/five/LJ_TBI_male_20 treatments_ final qeeg_Redacted.pdf`
- `/Users/davidmontgomery/Downloads/five/PG_femal_TBI+Parkinsons_20 treatments_final qeeg_Redacted.pdf`
- `/Users/davidmontgomery/Downloads/five/RR_Male_Long_COVID_20 treatments_Final qeeg_Redacted.pdf`

---

## Known “UI looks stuck” artifact (historical)

If the UI shows a run stuck “running”, one older run that timed out was:
- run_id: `d04db42d-fbec-4b7a-8ba4-6879969ad026`

This is a UI/state artifact; don’t chase it unless it blocks the workflow.

---

## How to run (manual)

1) Start CLIProxyAPI and ensure you can list models:
   - `GET http://127.0.0.1:8317/v1/models`

2) Start backend:
   - `uv run python -m backend.main`
   - or `./start.sh` to start everything

3) Upload report via UI or API, create a run.

4) If needed, trigger re-extraction for the report:
   - `POST /api/reports/{report_id}/reextract`
   Verify on disk the report folder contains `extracted_enhanced.txt` and complete `pages/page-*.png`.

5) Start run:
   - `POST /api/runs/{run_id}/start`
   Watch stream:
   - `GET /api/runs/{run_id}/stream`

6) Verify artifacts:
   - `data/artifacts/<run_id>/stage-1/_data_pack.json` (must contain required facts)
   - `data/artifacts/<run_id>/stage-1/_vision_transcript.md` (broad multimodal transcript for image-only tables/figures)
   - `data/artifacts/<run_id>/stage-4/<consolidator>.md` (must preserve facts)
   - exported `final.md` includes numeric tables

---

## Debugging playbook (when a run fails strict Stage 1)

1) Read the run error message (UI or `GET /api/runs/{run_id}`).
   - It should include `Failure artifact: data/artifacts/<run_id>/stage-1/_data_pack_failure.json`.

2) Open `_data_pack_failure.json`:
   - Confirm `missing_fields` and `expected_session_indices`.
   - Inspect `attempt_log` entries: open the referenced `*.raw.txt` and `*.parsed.json`.
   - Inspect `stage-1/_vision_transcript.md` to confirm the relevant page tables/figures were transcribed.

3) Confirm report assets:
   - Check `data/reports/<patient_id>/<upload_id>/pages/` has every page.
   - Check `extracted_enhanced.txt` includes the relevant page markers and the expected sections.

4) If page images or OCR look incomplete:
   - Re-run `POST /api/reports/{report_id}/reextract`.
   - Re-run the pipeline.

5) If failures are specific to the P300 Rare Comparison page:
   - Confirm the report actually has “P300 Rare Comparison” text in OCR on the correct page.
   - Check the debug artifacts for the `retry-p300-*` attempt.
   - Review the saved crop PNGs under `_data_pack_debug/<model_id>/retry-p300-*.images/` to confirm what the model saw.
   - If needed, enhance ROI crops (or add additional crop strategies) for readability.

---

## Next steps (high-value improvements)

These are likely the next “big wins” for completeness and auditability:

1) (Done) Save visual debug artifacts (images/crops), not just text outputs:
   - `_data_pack_debug/` now includes `*.images/` directories with the exact PNGs/crops used per multimodal call.
   - Attempt logs in `_data_pack_failure.json` include image paths when available.

2) Expand deterministic extraction coverage beyond the current “required fields”:
   - Many reports have additional clinically relevant numeric tables (band magnitudes, coherence summaries, etc.).
   - Strategy:
     - Add ROI-based OCR extraction + deterministic parsing for the hardest pages
     - Keep the LLM multimodal transcription as a second source, not the only source

3) (Done) Make multimodal ingestion notes stricter in strict mode:
   - Stage‑1 “page notes” ingestion now disables text-only fallback when `QEEG_STRICT_DATA_AVAILABILITY` is enabled.

4) Template awareness without relaxing strictness:
   - If some report templates omit specific required fields (e.g., Peak Frequency absent), don’t disable strict mode globally.
   - Implement template detection + explicit “not present” rules + alternative required facts.

5) (Done) Apple Vision OCR (macOS):
   - `backend/reports.py` now uses Apple Vision OCR (VNRecognizeTextRequest) on macOS when available, with
     Tesseract retained as a backstop/cross-check.
   - Env vars:
     - `QEEG_DISABLE_APPLE_VISION_OCR=1` to force-disable
     - `QEEG_APPLE_VISION_RECOGNITION_LEVEL=fast|accurate` (default `accurate`)
     - `QEEG_APPLE_VISION_LANGUAGE_CORRECTION=1` (default false; numeric-heavy pages usually prefer false)
   - Implementation reference source of truth lives on disk in:
     - `/Users/davidmontgomery/secondbrain/src/second_brain/ocr/apple_vision_ocr.py`
     - `/Users/davidmontgomery/secondbrain-ds/brain_scans/ocr_all_pages.py`

---

## Do not (still non-negotiable)

- Do not enable `QEEG_MOCK_LLM=1` for “real” runs.
- Do not relax strictness to make runs “complete” if data is missing; missing must fail loudly.
- Do not allow models to claim “not provided” if OCR/data pack contains the data.
- Do not reintroduce heuristics that drop OCR content (e.g., “pick one source by length”).

---

## Quick pointers (where to look in code)

- Stage 1 orchestration: `backend/council.py` `QEEGCouncilWorkflow._stage1()`
- Data pack build/validate/fail: `backend/council.py` `QEEGCouncilWorkflow._ensure_data_pack()`
- Required fields enforcement: `backend/council.py` `QEEGCouncilWorkflow._missing_required_fields()`
- Deterministic parsers:
  - `backend/council.py` `_facts_from_report_text_summary()` (summary metrics + peak frequency)
  - `backend/council.py` `_facts_from_report_text_n100_central_frontal()` (N100)
- Targeted retries + crops:
  - `backend/council.py` `_try_build_p300_cp_site_crops()`
  - `backend/council.py` `_try_build_summary_table_crops()`
