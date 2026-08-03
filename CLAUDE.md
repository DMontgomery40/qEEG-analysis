# CLAUDE.md — Technical Notes for qEEG Council

This file captures the architectural “source of truth” for this repo so future agents don’t misread report content (e.g., “WAVi”) as code, and don’t accidentally run in mock mode when evaluating real report quality.

## Project overview

qEEG Council is a **6-stage deliberation workflow** where multiple LLMs collaboratively analyze **redacted qEEG/ERP reports** (commonly WAVi PDF exports):

- Stage 1: initial analyses (parallel)
- Stage 2: peer review (parallel, anonymized A/B/C… labels)
- Stage 3: revision (parallel)
- Stage 4: consolidation (single consolidator)
- Stage 5: final review (parallel vote JSON)
- Stage 6: final drafts (parallel)
- Selection/export happens after Stage 6

## System topology

```
┌─────────────────────────────────────────────────────────────────────┐
│                     qEEG Council Frontend                           │
│                    (React + Vite, localhost:5173)                   │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
┌──────────────────────────────▼──────────────────────────────────────┐
│                     qEEG Council Backend                            │
│                  (FastAPI, localhost:8000)                          │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
┌──────────────────────────────▼──────────────────────────────────────┐
│                      CLIProxyAPI                                    │
│                   (http://127.0.0.1:8317)                           │
│  OpenAI-compatible: /v1/models, /v1/chat/completions, /v1/responses  │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
           ┌───────────────────┼───────────────────┐
           ▼                   ▼                   ▼
      Claude (OAuth)       OpenAI (OAuth)       Gemini (OAuth)
```

## Key non-negotiables

- **CLIProxyAPI is the only upstream**. No direct provider SDK calls.
- **“WAVi” is vendor/report content**, not a code module.
- For “real report quality” evaluation, you must run in **real mode** (see below), not mock mode.

## Ports

- CLIProxyAPI: `http://127.0.0.1:8317`
- Backend: `http://127.0.0.1:8000`
- Frontend: `http://127.0.0.1:5173`

## Run modes

### Real mode (default)

- Backend calls CLIProxyAPI and uses real model IDs from `GET /v1/models`.

### Mock mode (tests only)

- Set `QEEG_MOCK_LLM=1` before starting the backend.
- This swaps in a deterministic mocked transport for LLM calls.
- **Do not use mock mode to judge report quality** (it will produce canned content quickly).

## Patient identity

Every patient has one canonical clinic ID: `XX_MM-DD-YYYY` — two initials, an
underscore, the date of birth — with a collision ordinal only when two real
people share both (`ZZ_01-01-1900`, `ZZ_01-01-1900_2`; ordinal 1 is the
unsuffixed form, so `_1` never exists). `backend/patient_identity.py` holds the
authoritative regex and the real-calendar-date check.

- **This engine is the only allocator.** `allocate_canonical_patient_id` runs
  under a SQLite write transaction against the durable `patient_id_reservations`
  table, so an ordinal is issued once and never recomputed or reused. Ordinals
  are scoped to one initials-and-birthdate pair, not global: whatever happens to
  `ZZ_01-01-1900` leaves `QX_02-29-1904` untouched. The workbench, the hub, and
  the renderers read the ID; none of them mint one.
- **The ID is the clinic-visible key.** It is the patient's `label` column, the
  `data/portal_patients/` folder name, the prefix on every published filename,
  the hub's blob key, and the `patient_id` field in API responses.
- **The SQLite UUID is the invisible relational key.** It is the `patients.id`
  primary key and the join key for reports, runs, and files — and it is also the
  `{patient_uuid}` path parameter on `/api/patients/...` routes and the directory
  name under `data/reports/` and `data/patient_files/`. It never reaches a clinic
  screen, a portal folder, a published filename, or a chat message. Responses
  carry both: `id` is the UUID, `patient_id` is the canonical clinic ID.
- **Full names are stored.** `first_name`, `last_name`, `birthdate`,
  `first_initial`, and `last_initial` are normalized columns on `patients`. Names
  are ordinary working data for the clinic; the ID just carries the initials and
  the date of birth in a form a person can read at a glance.
- **Identity comes before creation.** The patient-neutral report preview endpoint
  runs extraction/OCR without creating a patient, report row, portal folder, or
  paid run. Create or find the patient from what it read, then register the
  report.
- **Name conflicts are answered, never guessed.** When incoming identity matches
  an existing patient's initials and birthdate but not their stored name, create
  returns `409 identity_name_mismatch` with the candidate patients and the
  incoming name. The caller resolves it explicitly with `attach_to` (same person;
  the stored name is not overwritten) or `force_new` (next ordinal). Never
  silently split a chart and never silently merge one.
- **`notes` is agent-managed free text.** Video/analogy preferences and informal
  context the clinic mentions once. No taxonomy, no parsing. A `PUT` that omits
  `notes` keeps what is stored — updates never blank it.

## Persistence and filesystem layout

- SQLite: `data/app.db`
- Reports:
  - `data/reports/<patient_uuid>/<upload_id>/original.pdf`
  - `data/reports/<patient_uuid>/<upload_id>/extracted.txt`
  - `data/reports/<patient_uuid>/<upload_id>/extracted_enhanced.txt` (OCR/table-friendly)
  - `data/reports/<patient_uuid>/<upload_id>/pages/page-<n>.png` (for multimodal Stage 1)
  - `data/reports/<patient_uuid>/<upload_id>/metadata.json`
- Patient files:
  - `data/patient_files/<patient_uuid>/<file_id>/original.<ext>`
- Portal sync folder:
  - `data/portal_patients/<PATIENT_ID>/...` (best-effort local publish copies, watched by `thrylen`)
- Artifacts: `data/artifacts/<run_id>/stage-<n>/<model_id>.(md|json)`
- Exports: `data/exports/<run_id>/final.(md|pdf)`

Important gotcha:
- **`report_id` (DB id) is not guaranteed to equal `<upload_id>` (folder name).**
- Always locate the report folder via the DB fields `stored_path` / `extracted_text_path`.

## Backend structure (`backend/`)

- `config.py`
  - `CLIPROXY_BASE_URL`, `CLIPROXY_API_KEY`
  - Model config + “vision-capable” detection
  - `DATA_DIR`, `REPORTS_DIR`, `ARTIFACTS_DIR`, `EXPORTS_DIR`
- `llm_client.py`
  - `AsyncOpenAICompatClient` (OpenAI-compatible client)
  - Prefer chat completions; fallback once to Responses when needed
- `reports.py`
  - PDF text extraction, enhanced OCR, and page image rendering (`extract_pdf_with_images`)
- `council/`
  - `QEEGCouncilWorkflow` orchestrates stages and writes artifacts
  - Stage 1 supports multimodal prompts for vision-capable models
  - Workflow core lives in `backend/council/workflow/core.py`
- `storage.py`
  - SQLite (patients/reports/runs/artifacts) + file paths for artifacts/exports
- `patient_files.py`
  - Patient file upload storage helpers (stored under `data/patient_files/`)
- `main.py`
  - FastAPI app + SSE broker + orchestration endpoints

## API surface (see `backend/main.py`)

- Health/models
  - `GET /api/health`
  - `GET /api/models`
- CLIProxy helpers (local convenience)
  - `POST /api/cliproxy/start`
  - `POST /api/cliproxy/login`
  - `POST /api/cliproxy/install`
- Patients
  - `GET/POST /api/patients`
  - `POST /api/patients/bulk_upload`
  - `GET/PUT /api/patients/{patient_uuid}`
  - `GET /api/patients/{patient_uuid}/reports`
  - `GET /api/patients/{patient_uuid}/runs`
  - `GET /api/patients/{patient_uuid}/files`
  - `POST /api/patients/{patient_uuid}/files`
  - `GET /api/patient_files/{file_id}`
  - `DELETE /api/patient_files/{file_id}`
- Reports
  - `POST /api/patients/{patient_uuid}/reports` (upload)
  - `GET /api/reports/{report_id}/extracted`
  - `POST /api/reports/{report_id}/reextract` (regenerate extracted/enhanced/pages)
  - `GET /api/reports/{report_id}/original`
  - `GET /api/reports/{report_id}/pages`
  - `GET /api/reports/{report_id}/pages/{page_num}`
  - `GET /api/reports/{report_id}/metadata`
- Runs
  - `POST /api/runs`
  - `POST /api/runs/{run_id}/start`
  - `GET /api/runs/{run_id}`
  - `GET /api/runs/{run_id}/artifacts`
  - `GET /api/runs/{run_id}/stream` (SSE)
  - `POST /api/runs/{run_id}/select`
- Exports
  - `POST /api/runs/{run_id}/export`
  - `GET /api/runs/{run_id}/export/final.md`
  - `GET /api/runs/{run_id}/export/final.pdf`

## Multimodal + extraction (critical for “all data available”)

- Stage 1 uses extracted text plus page images for vision-capable models.
- `POST /api/reports/{report_id}/reextract` is the “repair” button:
  - regenerates `extracted.txt`
  - regenerates `extracted_enhanced.txt`
  - regenerates `pages/page-*.png`
- Stage 1 processes page images in **multiple multimodal passes** as needed to cover **ALL pages**. Chunk size is controlled by `QEEG_VISION_PAGES_PER_CALL` (default 8) and is clamped to 10 pages per call; PDFs >10 pages will always run 2+ passes.
- Stage 1 writes run-level artifacts for downstream stages:
  - `data/artifacts/<run_id>/stage-1/_data_pack.json` (structured required facts)
  - `data/artifacts/<run_id>/stage-1/_vision_transcript.md` (broad transcription of image-only tables/figures)

## Commands

- Start everything (recommended): `./start.sh`
- Backend only: `uv run python -m backend.main`
- Backend tests: `uv run pytest -q`
- Frontend dev: `cd frontend && npm run dev`
- Frontend tests: `cd frontend && npm test` (Playwright)

## Common gotchas

1. Run the backend as a module (`python -m backend.main`) from the repo root to avoid import issues.
2. Don’t assume `report_id` == report folder name; always use `stored_path` / `extracted_text_path`.
3. Mock mode (`QEEG_MOCK_LLM=1`) is for deterministic tests only; it will not generate realistic clinical-quality reports.

## Explainer video integration

This repo is the source of truth for the explainer-video QC gate:
- Narrative truth: Stage 4 consolidation markdown (`kind='consolidation'`, `stage_num=4`)
- Numeric truth: Stage 1 data pack JSON (`kind='data_pack'`, `stage_num=1`, `model_id='_data_pack'`)

Publishing targets:
- Portal sync folder: `data/portal_patients/<PATIENT_ID>/` (configurable via `QEEG_PORTAL_PATIENTS_DIR`)
- DB-tracked upload: `POST /api/patients/{patient_uuid}/files` (also publishes a best-effort copy into the portal folder)

The “generate narrative + slides” pipeline lives in `../local-explainer-video`. Its **QC + Publish** step reads
`data/app.db` + artifacts from this repo, then writes the final MP4 into `data/portal_patients/` so `thrylen` can sync it.

Visual QC note:
- By default, the explainer repo runs visual QC in **check-only** mode (no automated image edits). When issues are found it writes:
  - `../local-explainer-video/projects/<PROJECT>/qc_visual_issues.json`
- Image models (in the explainer repo): generate via `qwen/qwen-image-2512`, edit via `qwen/qwen-image-edit-2511` (or DashScope `qwen-image-edit-max` when configured).

<!-- gitnexus:start -->
# GitNexus — Code Intelligence

This project is indexed by GitNexus as **qEEG-analysis** (5903 symbols, 14667 relationships, 300 execution flows). Use the GitNexus MCP tools to understand code, assess impact, and navigate safely.

> If any GitNexus tool warns the index is stale, run `npx gitnexus analyze` in terminal first.

## Always Do

- **MUST run impact analysis before editing any symbol.** Before modifying a function, class, or method, run `gitnexus_impact({target: "symbolName", direction: "upstream"})` and report the blast radius (direct callers, affected processes, risk level) to the user.
- **MUST run `gitnexus_detect_changes()` before committing** to verify your changes only affect expected symbols and execution flows.
- **MUST warn the user** if impact analysis returns HIGH or CRITICAL risk before proceeding with edits.
- When exploring unfamiliar code, use `gitnexus_query({query: "concept"})` to find execution flows instead of grepping. It returns process-grouped results ranked by relevance.
- When you need full context on a specific symbol — callers, callees, which execution flows it participates in — use `gitnexus_context({name: "symbolName"})`.

## Never Do

- NEVER edit a function, class, or method without first running `gitnexus_impact` on it.
- NEVER ignore HIGH or CRITICAL risk warnings from impact analysis.
- NEVER rename symbols with find-and-replace — use `gitnexus_rename` which understands the call graph.
- NEVER commit changes without running `gitnexus_detect_changes()` to check affected scope.

## Resources

| Resource | Use for |
|----------|---------|
| `gitnexus://repo/qEEG-analysis/context` | Codebase overview, check index freshness |
| `gitnexus://repo/qEEG-analysis/clusters` | All functional areas |
| `gitnexus://repo/qEEG-analysis/processes` | All execution flows |
| `gitnexus://repo/qEEG-analysis/process/{name}` | Step-by-step execution trace |

## CLI

| Task | Read this skill file |
|------|---------------------|
| Understand architecture / "How does X work?" | `.claude/skills/gitnexus/gitnexus-exploring/SKILL.md` |
| Blast radius / "What breaks if I change X?" | `.claude/skills/gitnexus/gitnexus-impact-analysis/SKILL.md` |
| Trace bugs / "Why is X failing?" | `.claude/skills/gitnexus/gitnexus-debugging/SKILL.md` |
| Rename / extract / split / refactor | `.claude/skills/gitnexus/gitnexus-refactoring/SKILL.md` |
| Tools, resources, schema reference | `.claude/skills/gitnexus/gitnexus-guide/SKILL.md` |
| Index, status, clean, wiki CLI commands | `.claude/skills/gitnexus/gitnexus-cli/SKILL.md` |

<!-- gitnexus:end -->
