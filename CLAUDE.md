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

## Persistence and filesystem layout

- SQLite: `data/app.db`
- Reports:
  - `data/reports/<patient_id>/<upload_id>/original.pdf`
  - `data/reports/<patient_id>/<upload_id>/extracted.txt`
  - `data/reports/<patient_id>/<upload_id>/extracted_enhanced.txt` (OCR/table-friendly)
  - `data/reports/<patient_id>/<upload_id>/pages/page-<n>.png` (for multimodal Stage 1)
  - `data/reports/<patient_id>/<upload_id>/metadata.json`
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
- `council.py`
  - `QEEGCouncilWorkflow` orchestrates stages and writes artifacts
  - Stage 1 supports multimodal prompts for vision-capable models
- `storage.py`
  - SQLite (patients/reports/runs/artifacts) + file paths for artifacts/exports
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
  - `GET/PUT /api/patients/{patient_id}`
  - `GET /api/patients/{patient_id}/reports`
  - `GET /api/patients/{patient_id}/runs`
- Reports
  - `POST /api/patients/{patient_id}/reports` (upload)
  - `GET /api/reports/{report_id}/extracted`
  - `POST /api/reports/{report_id}/reextract` (regenerate extracted/enhanced/pages)
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
- Current implementation limits a single multimodal Stage 1 call to the **first 10 page images** (see `backend/council.py`). For PDFs >10 pages, Stage 1 must be extended to **multi-pass** to guarantee full coverage.

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
