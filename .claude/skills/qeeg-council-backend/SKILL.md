---
name: qeeg-council-backend
description: Build and maintain the qEEG Council FastAPI backend, CLIProxyAPI upstream client, SQLite persistence, and the 6-stage workflow artifacts and exports.
license: MIT
compatibility: Designed for Claude Code and Codex CLI. Requires python 3.11+, uv, and CLIProxyAPI reachable at CLIPROXY_BASE_URL (default http://127.0.0.1:8317).
metadata:
  author: dmontgomery
  version: "2.0"
---

# qEEG Council Backend Skill

## Use this Skill for
- implementing or refactoring FastAPI endpoints
- implementing the 6-stage workflow orchestration
- adding or changing artifact formats and storage layout
- integrating CLIProxyAPI model discovery and request routing
- implementing SSE progress streaming

## Non-negotiables
- CLIProxyAPI is the only model upstream. No direct provider SDK calls.
- “WAVi” is vendor/report content inside PDFs, not a code module.
- For report-quality evaluation: **do not** use mock mode (`QEEG_MOCK_LLM=1` is tests-only).

## Modes
- Real mode (default): backend calls CLIProxyAPI and uses real discovered model IDs.
- Mock mode (tests only): set `QEEG_MOCK_LLM=1` before starting the backend. Mock mode is not valid for evaluating report quality.

## Where the code lives (post-refactor)
- `backend/main.py`: FastAPI app + endpoints + SSE stream
- `backend/council/`: council package (import as `backend.council`)
  - `backend/council/workflow/core.py`: `QEEGCouncilWorkflow` orchestration
  - `backend/council/workflow/stages.py`: per-stage prompt + artifact logic
  - `backend/council/workflow/data_pack.py`: Stage 1 structured `_data_pack.json` + required-field validation
- `backend/reports.py`: PDF text extraction + enhanced OCR + page image rendering
- `backend/llm_client.py`: OpenAI-compatible async client to CLIProxyAPI (`/v1/models`, `/v1/chat/completions`, `/v1/responses`)
- `backend/storage.py`: SQLite + filesystem paths for reports/artifacts/exports

## Commands
- Install deps: `uv sync`
- Run backend: `uv run python -m backend.main`
- Run tests: `uv run pytest -q`
- Format: `uv run ruff format`
- Lint: `uv run ruff check --fix`

## Quick start checks
- CLIProxyAPI reachability + discovered models: `uv run python -m backend.cliproxy_status`
- Backend smoke test (requires backend running): `uv run python backend/scripts/smoke_api.py`
- CLIProxyAPI model list (direct): `uv run python backend/scripts/cliproxy_models.py`

## Canonical references
- Workflow spec: [references/workflow.md](references/workflow.md)
- API contract: [references/api-contract.md](references/api-contract.md)
- JSON shapes:
  - Stage 2 peer review schema: [assets/schemas/stage2_peer_review.schema.json](assets/schemas/stage2_peer_review.schema.json)
  - Stage 5 final review schema: [assets/schemas/stage5_final_review.schema.json](assets/schemas/stage5_final_review.schema.json)

## Stage 1 “data availability” rule of thumb
Before blaming models, verify the backend has actually extracted and stored:
- `extracted.txt`
- `extracted_enhanced.txt`
- `pages/page-*.png`

If missing/garbled, repair via:
- `POST /api/reports/{report_id}/reextract`

## Report storage gotcha (don’t miss)
- Report files live under `data/reports/<patient_id>/<upload_id>/...`
- The DB `report_id` is not guaranteed to equal `<upload_id>` (folder name).
- Always locate report assets via `stored_path` / `extracted_text_path`, not by constructing paths from ids.

## Validation utilities
- Stage 2 JSON validation: `python scripts/validate_stage2_peer_review.py path/to/file.json`
- Stage 5 JSON validation: `python scripts/validate_stage5_final_review.py path/to/file.json`
