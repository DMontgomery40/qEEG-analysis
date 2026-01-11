# CLAUDE.md - Technical Notes for qEEG Council

This file contains technical details, architectural decisions, and implementation notes to keep future sessions consistent.

## Project Overview

qEEG Council is a 6-stage deliberation workflow where multiple LLMs collaboratively analyze qEEG reports.

Key features

* Subscription-based model access via CLIProxyAPI (no per-token API keys in this app)
* Multi-round peer review plus revision cycle
* Patient tracking plus run history

## Architecture

### System topology

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
│     Wraps Claude Code CLI, Gemini CLI, and OpenAI Codex CLI         │
│     Exposes OpenAI-compatible endpoints (/v1/models, /v1/...)        │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
           ┌───────────────────┼───────────────────┐
           ▼                   ▼                   ▼
      Claude (OAuth)       Codex (OAuth)        Gemini (OAuth)
```

### Backend structure (`backend/`)

**`config.py`**

* `CLIPROXY_BASE_URL`: CLIProxyAPI base URL (default `http://127.0.0.1:8317`)
* `CLIPROXY_API_KEY`: optional key for CLIProxyAPI (empty means no auth header)
* `COUNCIL_MODELS`: list of model configs, each with:

  * `id`: model id sent to CLIProxyAPI
  * `name`: UI display name
  * `source`: UI badge text
  * `endpoint_preference`: `chat` or `responses`
* `DEFAULT_CONSOLIDATOR`: model id used for Stage 4 synthesis
* Data directories:

  * `DATA_DIR`
  * `REPORTS_DIR`
  * `ARTIFACTS_DIR`
  * `EXPORTS_DIR`

**`llm_client.py`**

* `AsyncOpenAICompatClient`: generic OpenAI-compatible async client

  * `list_models()` calls `GET /v1/models`
  * `chat_completions()` calls `POST /v1/chat/completions`
  * `responses()` calls `POST /v1/responses`
* Fallback behavior

  * Prefer `chat_completions`
  * If the upstream indicates the model only supports Responses, retry once using `/v1/responses`
* Auth header behavior

  * If `CLIPROXY_API_KEY` is non-empty, send `Authorization: Bearer <key>`

**`storage.py`**

* SQLite metadata plus file storage for large text
* Store in SQLite

  * patients
  * reports
  * runs
  * artifacts
* Store on disk

  * uploaded report files
  * extracted text
  * stage artifacts (`.md` or `.json`)
  * exports (`final.md`, `final.pdf`)

**`patients.py`**

* Patient CRUD
* Minimal schema (no PHI assumptions)

  * `{id, label, notes, created_at, updated_at}`

**`reports.py`**

* Report handling

  * accept PDF or text
  * extract text for PDFs (pypdf)
  * store original file plus extracted text
* Reports link to patients

**`council.py`** (core orchestration)

* `QEEGCouncilWorkflow` orchestrates 6 stages
* Stages

  * Stage 1: initial analyses (parallel)
  * Stage 2: peer review (parallel, anonymized)
  * Stage 3: revision (parallel)
  * Stage 4: consolidation (single consolidator)
  * Stage 5: final review (parallel vote and required edits)
  * Stage 6: final drafts (parallel)
* Selection happens in the UI after Stage 6

**`main.py`**

* FastAPI app with CORS for `http://localhost:5173`
* Core endpoints

  * `GET /api/health`
  * `GET /api/models` (configured plus discovered via CLIProxyAPI)
  * `GET/POST/PUT /api/patients`
  * `POST /api/patients/{id}/reports`
  * `POST /api/runs` (create a run)
  * `POST /api/runs/{run_id}/start` (start pipeline)
  * `GET /api/runs/{run_id}` (run status plus artifact references)
  * `GET /api/runs/{run_id}/stream` (SSE for stage progress)
  * `GET /api/patients/{id}/runs` (history)
  * `POST /api/runs/{run_id}/export` (generate md and pdf)

**`prompts/`**

* Stage prompts live in `backend/prompts/`

  * `stage1_analysis.md`
  * `stage2_peer_review.md` (JSON output)
  * `stage3_revision.md`
  * `stage4_consolidation.md`
  * `stage5_final_review.md` (JSON output)
  * `stage6_final_draft.md`

### Frontend structure (`frontend/src/`)

**Pages**

* `Dashboard` (patient list and quick actions)
* `PatientPage` (patient detail, reports, runs)
* `RunPage` (stage progress, artifacts, export)

**Components**

* `PatientList` (filterable list)
* `PatientDetail` (header card)
* `ReportUpload` (drag and drop)
* `StageProgress` (6-stage progress)
* `ArtifactTabs` (tab view for per-model outputs)
* `PeerReviewView` (anonymized labels plus de-anonymized display)
* `ConsolidatedReport` (Stage 4)
* `FinalDraftCompare` (Stage 6 plus selection)
* `ModelBadge` (shows model source label)

## The 6-stage workflow

```
                    ┌─────────────────┐
                    │   qEEG Report   │
                    └────────┬────────┘
                             ▼
┌──────────────────────────────────────────────────────────────────┐
│ STAGE 1: Initial Analysis                                        │
│ All models independently analyze the qEEG report                  │
│ Output: N analyses                                                │
└────────────────────────────┬─────────────────────────────────────┘
                             ▼
┌──────────────────────────────────────────────────────────────────┐
│ STAGE 2: Peer Review                                             │
│ Each model reviews the other analyses (anonymized A/B/C...)       │
│ Output: N peer review JSON artifacts                              │
└────────────────────────────┬─────────────────────────────────────┘
                             ▼
┌──────────────────────────────────────────────────────────────────┐
│ STAGE 3: Revision                                                │
│ Each model revises its analysis based on feedback                 │
│ Output: N revised analyses                                        │
└────────────────────────────┬─────────────────────────────────────┘
                             ▼
┌──────────────────────────────────────────────────────────────────┐
│ STAGE 4: Consolidation                                           │
│ Consolidator synthesizes revised analyses                         │
│ Output: 1 consolidated report                                     │
└────────────────────────────┬─────────────────────────────────────┘
                             ▼
┌──────────────────────────────────────────────────────────────────┐
│ STAGE 5: Final Review                                            │
│ All models review consolidation and vote APPROVE or REVISE        │
│ Output: N final review JSON artifacts                             │
└────────────────────────────┬─────────────────────────────────────┘
                             ▼
┌──────────────────────────────────────────────────────────────────┐
│ STAGE 6: Final Drafts                                            │
│ All models produce a polished draft applying required edits       │
│ Output: N final drafts                                            │
└────────────────────────────┬─────────────────────────────────────┘
                             ▼
┌──────────────────────────────────────────────────────────────────┐
│ SELECTION (UI)                                                    │
│ User selects a final draft or keeps the consolidated report       │
│ Output: Saved and exportable                                      │
└──────────────────────────────────────────────────────────────────┘
```

## Key design decisions

### CLIProxyAPI as the auth and routing layer

* Goal

  * Use flat subscription access via the official CLIs, but expose them as a single OpenAI-compatible endpoint.
* Result

  * qEEG Council talks to `CLIPROXY_BASE_URL` only.
* Model routing

  * Use the exact model ids returned by `GET /v1/models` from CLIProxyAPI.
  * Many model ids follow prefixes like `claude-*`, `gpt-*`, `gemini-*`, but the real source of truth is `/v1/models`.

### Anonymization in Stage 2

* Models receive analyses labeled A, B, C.
* Backend stores mapping of labels to model ids per run.
* UI de-anonymizes only for display.

### Error handling

* Treat upstream errors as recoverable when possible

  * 401 typically means re-auth is needed in CLIProxyAPI
  * 429 means rate limiting
  * 5xx means upstream unavailable
* Degrade gracefully

  * Stage 1 can complete with partial models
  * Consolidation requires at least one revision

### Persistence

* SQLite for metadata
* Files on disk for report text, artifacts, and exports

## Important implementation details

### CLIProxyAPI setup

CLIProxyAPI can be installed via Homebrew as `cliproxyapi`, or built from source as `cli-proxy-api`.

Homebrew install

```bash
brew tap router-for-me/tap
brew install cliproxyapi
```

Login flows

```bash
cliproxyapi -claude-login
cliproxyapi -codex-login
cliproxyapi -login
```

Gemini often works better with a project id

```bash
cliproxyapi -login -project_id YOUR_PROJECT_ID
```

Start the proxy

```bash
cliproxyapi -config /opt/homebrew/etc/cliproxyapi.conf
```

### Port configuration

* CLIProxyAPI: 8317
* Backend: 8000
* Frontend: 5173

### Streaming

SSE needs a GET endpoint, so the app uses a pattern like:

* `POST /api/runs` to create
* `POST /api/runs/{run_id}/start` to run
* `GET /api/runs/{run_id}/stream` to watch progress

Example client

```javascript
const runId = "...";
const es = new EventSource(`/api/runs/${runId}/stream`);
es.onmessage
```
